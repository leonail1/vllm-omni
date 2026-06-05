# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import inspect
import json
import logging
import math
import os
import time
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
import torch.distributed
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import DistributedAutoencoderKLQwenImage
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import prefetch_subfolders
from vllm_omni.diffusion.models.dmd2 import DMD2PipelineMixin
from vllm_omni.diffusion.models.qwen_image.cfg_parallel import (
    QwenImageCFGParallelMixin,
)
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenImageTransformer2DModel,
)
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.diffusion.utils.prompt_utils import (
    validate_prompt_sequence_lengths,
)
from vllm_omni.diffusion.utils.size_utils import (
    normalize_min_aligned_size,
)
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import DiffusionRequestState

from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

logger = logging.getLogger(__name__)

QWEN_IMAGE_STAGE_PAYLOAD_KEY = "qwen_image_stage_payload"
QWEN_IMAGE_STAGE_KIND_KEY = "qwen_image_stage_kind"
QWEN_IMAGE_STAGE_TRACE_KEY = "qwen_image_stage_trace"
QWEN_IMAGE_DENOISE_BATCH_TRACE_KEY = "qwen_image_denoise_batch_trace"


def _get_qwen_image_model_path(model_name: str) -> str:
    if os.path.exists(model_name):
        return model_name
    return download_weights_from_hf_specific(model_name, None, ["vae/config.json"])


def _load_qwen_image_vae_config(model_name: str) -> dict[str, Any]:
    model_path = _get_qwen_image_model_path(model_name)
    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        return json.load(f)


def _get_qwen_image_vae_scale_factor(model_name: str) -> int:
    vae_config = _load_qwen_image_vae_config(model_name)
    return 2 ** len(vae_config["temporal_downsample"]) if "temporal_downsample" in vae_config else 8


def _extract_qwen_image_stage_payload(
    prompts: list[Any] | None,
    expected_kind: str,
) -> dict[str, Any]:
    if not prompts:
        raise ValueError(f"Qwen-Image {expected_kind} stage requires an upstream stage payload.")
    prompt = prompts[0]
    if not isinstance(prompt, dict):
        raise ValueError(
            f"Qwen-Image {expected_kind} stage expects a dict prompt carrying "
            f"{QWEN_IMAGE_STAGE_PAYLOAD_KEY!r}, got {type(prompt).__name__}."
        )
    payload = prompt.get(QWEN_IMAGE_STAGE_PAYLOAD_KEY)
    kind = prompt.get(QWEN_IMAGE_STAGE_KIND_KEY)
    if not isinstance(payload, dict) or kind != expected_kind:
        raise ValueError(
            f"Qwen-Image {expected_kind} stage received invalid payload kind {kind!r}."
        )
    return payload


def _get_qwen_image_stage_payload_from_prompt(prompt: Any) -> dict[str, Any] | None:
    if not isinstance(prompt, dict):
        return None
    payload = prompt.get(QWEN_IMAGE_STAGE_PAYLOAD_KEY)
    if not isinstance(payload, dict):
        return None
    return payload


def get_qwen_image_stage_payload_do_true_cfg(prompt: Any) -> bool | None:
    payload = _get_qwen_image_stage_payload_from_prompt(prompt)
    if payload is None:
        return None
    return bool(payload.get("do_true_cfg", False))


def _to_cpu_stage_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _to_cpu_stage_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_stage_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_stage_value(item) for item in value)
    return value


def _qwen_image_tensor_nbytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sum(_qwen_image_tensor_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_qwen_image_tensor_nbytes(item) for item in value)
    return 0


def _qwen_image_payload_nbytes(payload: Mapping[str, Any]) -> int:
    return sum(
        _qwen_image_tensor_nbytes(value)
        for key, value in payload.items()
        if key != QWEN_IMAGE_STAGE_TRACE_KEY
    )


def _qwen_image_trace_request_id(payload: Mapping[str, Any]) -> str:
    trace = payload.get(QWEN_IMAGE_STAGE_TRACE_KEY)
    if isinstance(trace, list):
        for event in reversed(trace):
            if isinstance(event, Mapping):
                request_id = event.get("request_id")
                if request_id:
                    return str(request_id)
    return "unknown"


def _qwen_image_runtime_trace_context() -> dict[str, Any]:
    """Collect stable runtime identifiers for cross-stage trace analysis."""
    context: dict[str, Any] = {}
    for env_name, trace_key in (
        ("VLLM_OMNI_STAGE_ID", "stage_id"),
        ("VLLM_OMNI_REPLICA_ID", "replica_id"),
        ("ASCEND_RT_VISIBLE_DEVICES", "ascend_visible_devices"),
        ("CUDA_VISIBLE_DEVICES", "cuda_visible_devices"),
    ):
        value = os.environ.get(env_name)
        if value is None or value == "":
            continue
        if trace_key in {"stage_id", "replica_id"}:
            try:
                context[trace_key] = int(value)
                continue
            except ValueError:
                pass
        context[trace_key] = value
    return context


def _append_qwen_image_stage_trace(
    payload: dict[str, Any],
    *,
    stage: str,
    request_id: str,
    start_s: float,
    end_s: float,
    payload_bytes: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    trace = list(payload.get(QWEN_IMAGE_STAGE_TRACE_KEY) or [])
    event = {
        "stage": stage,
        "request_id": request_id,
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": max(0.0, end_s - start_s),
        "payload_bytes": _qwen_image_payload_nbytes(payload) if payload_bytes is None else int(payload_bytes),
    }
    event.update(_qwen_image_runtime_trace_context())
    if extra:
        for key, value in extra.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                event[str(key)] = value
    trace.append(event)
    payload[QWEN_IMAGE_STAGE_TRACE_KEY] = trace
    return payload


def _qwen_image_stage_env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _qwen_image_stage_timing_sync(enabled: bool) -> None:
    if not enabled:
        return
    npu = getattr(torch, "npu", None)
    if npu is not None:
        is_available = getattr(npu, "is_available", None)
        synchronize = getattr(npu, "synchronize", None)
        if callable(is_available) and is_available() and callable(synchronize):
            synchronize()
            return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _qwen_image_record_denoise_batch_trace(
    instance: Any,
    input_batch: "InputBatch",
    *,
    start_s: float,
    end_s: float,
) -> None:
    if not _qwen_image_stage_env_flag("QWEN_IMAGE_DENOISE_BATCH_TRACE"):
        return
    if all(request_id == DUMMY_DIFFUSION_REQUEST_ID for request_id in input_batch.request_ids):
        return

    trace_by_request = getattr(instance, "_qwen_image_denoise_batch_trace_by_request", None)
    if trace_by_request is None:
        trace_by_request = {}
        setattr(instance, "_qwen_image_denoise_batch_trace_by_request", trace_by_request)

    token_counts = {span.request_id: int(span.token_count) for span in input_batch.request_spans}
    row_counts = {span.request_id: int(span.row_count) for span in input_batch.request_spans}
    total_image_tokens = int(sum(token_counts.values()))
    total_rows = int(sum(row_counts.values()))
    common = {
        "stage": "denoise_step",
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": max(0.0, end_s - start_s),
        "batch_size": int(input_batch.num_reqs),
        "num_reqs_after_padding": int(input_batch.num_reqs_after_padding),
        "total_rows": total_rows,
        "total_image_tokens": total_image_tokens,
        "is_dynamic": bool(input_batch.is_dynamic),
    }
    common.update(_qwen_image_runtime_trace_context())
    for request_id in input_batch.request_ids:
        request_trace = trace_by_request.setdefault(request_id, [])
        event = dict(common)
        event["request_id"] = request_id
        event["step_index"] = len(request_trace)
        event["request_image_tokens"] = int(token_counts.get(request_id, 0))
        event["request_rows"] = int(row_counts.get(request_id, 0))
        request_trace.append(event)


def _qwen_image_pop_denoise_batch_trace(instance: Any, request_id: str) -> list[dict[str, Any]]:
    trace_by_request = getattr(instance, "_qwen_image_denoise_batch_trace_by_request", None)
    if not isinstance(trace_by_request, dict):
        return []
    trace = trace_by_request.pop(request_id, [])
    return trace if isinstance(trace, list) else []


def _is_qwen_image_shape_triplet(value: Any) -> bool:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(item, (int, np.integer)) for item in value)
    )


def _as_qwen_image_shape_tuple(value: Any) -> tuple[int, int, int]:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if not _is_qwen_image_shape_triplet(value):
        raise ValueError(f"Invalid Qwen-Image shape entry: {value!r}.")
    return (int(value[0]), int(value[1]), int(value[2]))


def _normalize_qwen_image_img_shapes(img_shapes: Any) -> list[list[tuple[int, int, int]]]:
    """Normalize stage-transported image shapes to the request-state layout."""
    if torch.is_tensor(img_shapes):
        img_shapes = img_shapes.detach().cpu().tolist()
    if _is_qwen_image_shape_triplet(img_shapes):
        return [[_as_qwen_image_shape_tuple(img_shapes)]]
    if not isinstance(img_shapes, (list, tuple)) or not img_shapes:
        raise ValueError(f"Invalid Qwen-Image img_shapes payload: {img_shapes!r}.")

    normalized: list[list[tuple[int, int, int]]] = []
    for sample_shapes in img_shapes:
        if torch.is_tensor(sample_shapes):
            sample_shapes = sample_shapes.detach().cpu().tolist()
        if _is_qwen_image_shape_triplet(sample_shapes):
            normalized.append([_as_qwen_image_shape_tuple(sample_shapes)])
            continue
        if not isinstance(sample_shapes, (list, tuple)) or not sample_shapes:
            raise ValueError(f"Invalid Qwen-Image sample img_shapes entry: {sample_shapes!r}.")
        normalized.append([_as_qwen_image_shape_tuple(shape) for shape in sample_shapes])
    return normalized


def _make_dummy_encode_payload(
    *,
    height: int,
    width: int,
    num_inference_steps: int,
    true_cfg_scale: float = 4.0,
) -> dict[str, Any]:
    timesteps = torch.arange(num_inference_steps, 0, -1, dtype=torch.float32)
    return {
        "prompt_embeds": torch.zeros(1, 1, 1),
        "prompt_embeds_mask": torch.ones(1, 1, dtype=torch.long),
        "negative_prompt_embeds": None,
        "negative_prompt_embeds_mask": None,
        "latents": torch.zeros(1, 1, 4),
        "img_shapes": [[(1, 1, 1)]],
        "timesteps": timesteps,
        "num_inference_steps": num_inference_steps,
        "sigmas": None,
        "do_true_cfg": False,
        "guidance": None,
        "txt_seq_lens": [1],
        "negative_txt_seq_lens": None,
        "true_cfg_scale": true_cfg_scale,
        "cfg_normalize": True,
        "height": height,
        "width": width,
        "output_type": "latent",
        "attention_kwargs": {},
    }


def _make_dummy_denoise_payload(
    *,
    height: int,
    width: int,
) -> dict[str, Any]:
    return {
        "latents": torch.zeros(1, 1, 4),
        "height": height,
        "width": width,
        "output_type": "dummy_image",
    }


def _stage_output(payload: dict[str, Any], kind: str) -> DiffusionOutput:
    payload_bytes = _qwen_image_payload_nbytes(payload)
    export_start_s = time.time()
    converted_payload = _to_cpu_stage_value(payload)
    export_end_s = time.time()
    if _qwen_image_stage_env_flag("QWEN_IMAGE_STAGE_TRANSFER_TRACE"):
        _append_qwen_image_stage_trace(
            converted_payload,
            stage=f"{kind}_payload_export",
            request_id=_qwen_image_trace_request_id(payload),
            start_s=export_start_s,
            end_s=export_end_s,
            payload_bytes=payload_bytes,
            extra={
                "source_device": "stage_device",
                "target_device": "cpu",
                "payload_kind": kind,
            },
        )
    return DiffusionOutput(
        output=[],
        custom_output={
            QWEN_IMAGE_STAGE_PAYLOAD_KEY: converted_payload,
            QWEN_IMAGE_STAGE_KIND_KEY: kind,
        },
    )


def get_qwen_image_post_process_func(
    od_config: OmniDiffusionConfig,
):
    model_name = od_config.model
    vae_scale_factor = _get_qwen_image_vae_scale_factor(model_name)

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    def post_process_func(
        images: torch.Tensor,
    ):
        return image_processor.postprocess(images)

    return post_process_func


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, int]:
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: torch.Tensor | tuple[torch.Tensor],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class QwenImagePipeline(nn.Module, QwenImageCFGParallelMixin, DiffusionPipelineProfilerMixin):
    supports_step_execution: ClassVar[bool] = True
    EXTRA_OUTPUT_PARAMS: ClassVar[frozenset[str]] = frozenset(
        {QWEN_IMAGE_STAGE_TRACE_KEY, QWEN_IMAGE_DENOISE_BATCH_TRACE_KEY}
    )

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        self.device = get_local_device()
        model = od_config.model
        # Check if model is a local path
        local_files_only = os.path.exists(model)

        # See pipeline_qwen_image_edit_plus: guard against transformers v5
        # multi-worker race on partial subfolder shard sets (Buildkite #1043).
        prefetch_subfolders(
            model,
            ["scheduler", "text_encoder", "vae", "tokenizer"],
            local_files_only=local_files_only,
        )

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, subfolder="text_encoder", local_files_only=local_files_only
        )
        # Qwen2.5-VL ships a vision tower that text-to-image does not use.
        # Drop it while the model is still on CPU, before moving to GPU, so
        # the vision tower never consumes GPU memory. Handle both transformers
        # layouts: newer puts visual under .model, older puts it directly on
        # the model.
        visual_owner = None
        if hasattr(self.text_encoder, "model") and hasattr(self.text_encoder.model, "visual"):
            visual_owner = self.text_encoder.model
        elif hasattr(self.text_encoder, "visual"):
            visual_owner = self.text_encoder
        if visual_owner is not None:
            del visual_owner.visual
        else:
            logger.warning("Qwen-Image: vision tower not found on text encoder; skipping drop")
        self.text_encoder = self.text_encoder.to(self.device)
        self.vae = DistributedAutoencoderKLQwenImage.from_pretrained(
            model, subfolder="vae", local_files_only=local_files_only
        ).to(self.device)
        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, QwenImageTransformer2DModel)
        self.transformer = QwenImageTransformer2DModel(
            od_config=od_config, quant_config=od_config.quantization_config, **transformer_kwargs
        )

        self.tokenizer = Qwen2Tokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)

        self.stage = None

        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        # QwenImage latents are turned into 2x2 patches and packed.
        # This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        # self.image_processor = VaeImageProcessor(
        #     vae_scale_factor=self.vae_scale_factor * 2
        # )
        self.tokenizer_max_length = 1024
        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"  # noqa: E501
        self.prompt_template_encode_start_idx = 34
        self.default_sample_size = 128

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} "
                f"but are {height} and {width}. Dimensions will be resized accordingly"
            )

        # if callback_on_step_end_tensor_inputs is not None and not all(
        #     k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        # ):
        #     raise ValueError(
        #         f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs},
        # but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
        #     )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `prompt_embeds_mask` also have to be passed. "
                "Make sure to generate `prompt_embeds_mask` from the same text encoder "
                "that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` also have to be passed. "
                "Make sure to generate `negative_prompt_embeds_mask` from the same text encoder "
                "that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > self.tokenizer_max_length:
            raise ValueError(
                f"`max_sequence_length` cannot be greater than {self.tokenizer_max_length} but is {max_sequence_length}"
            )

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    def _get_qwen_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        dtype: torch.dtype | None = None,
        max_sequence_length: int | None = None,
        prompt_name: str = "prompt",
    ):
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        txt_tokens = self.tokenizer(
            txt,
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(self.device)
        # Validate only the user prompt contribution. The Qwen template also
        # adds a fixed suffix after the user text, so subtracting only
        # prompt_template_encode_start_idx would overcount near-limit prompts.
        template_tokens = self.tokenizer(
            [template.format("")],
            padding=True,
            truncation=False,
            return_tensors="pt",
        ).to(self.device)
        validate_prompt_sequence_lengths(
            txt_tokens.attention_mask,
            max_sequence_length=max_sequence_length or self.tokenizer_max_length,
            supported_max_sequence_length=self.tokenizer_max_length,
            prompt_name=prompt_name,
            baseline_attention_mask=template_tokens.attention_mask,
            error_context="after applying the Qwen prompt template",
        )
        encoder_hidden_states = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype)

        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
        prompt_name: str = "prompt",
    ):
        r"""

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt,
                max_sequence_length=max_sequence_length,
                prompt_name=prompt_name,
            )

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

        return latents

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ) -> torch.Tensor:
        # generator=torch.Generator(device="cuda").manual_seed(42)
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, 1, num_channels_latents, height, width)

        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        return latents

    def prepare_timesteps(self, num_inference_steps, sigmas, image_seq_len):
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        # image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
            mu=mu,
        )
        return timesteps, num_inference_steps

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def _extract_prompts(self, prompts):
        """Extract prompt and negative_prompt from OmniPromptType list."""
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in prompts] or None
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in prompts):
            negative_prompt = None
        elif prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in prompts]
        else:
            negative_prompt = None
        return prompt, negative_prompt

    def _prepare_generation_context(
        self,
        *,
        prompt,
        negative_prompt,
        height,
        width,
        num_inference_steps,
        sigmas,
        guidance_scale,
        num_images_per_prompt,
        generator,
        true_cfg_scale,
        max_sequence_length,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        latents=None,
        attention_kwargs=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        """Shared preparation logic for forward() and prepare_encode().

        Validates inputs, encodes prompts, prepares latents, computes timesteps,
        and returns all intermediate values as a dict.
        """
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs,
            max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs or {}
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            batch_size = 1

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        self.check_cfg_parallel_validity(true_cfg_scale, has_neg_prompt)

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                prompt_name="negative_prompt",
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None

        num_channels_latents = self.transformer.in_channels // 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )

        img_shapes = [[(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)]] * batch_size

        timesteps, num_inference_steps = self.prepare_timesteps(
            num_inference_steps,
            sigmas,
            latents.shape[1],
        )
        self._num_timesteps = len(timesteps)

        if self.transformer.guidance_embeds:
            guidance = torch.full([1], guidance_scale, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        return {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
            "latents": latents,
            "img_shapes": img_shapes,
            "timesteps": timesteps,
            "do_true_cfg": do_true_cfg,
            "guidance": guidance,
            "txt_seq_lens": txt_seq_lens,
            "negative_txt_seq_lens": negative_txt_seq_lens,
        }

    def prepare_encode(
        self,
        state: "DiffusionRequestState",
        **kwargs: Any,
    ) -> "DiffusionRequestState":
        """Populate *state* with encoded prompts, latents, timesteps, and CFG config."""
        stage_start_s = time.time()
        sampling = state.sampling
        prompt, negative_prompt = self._extract_prompts(state.prompts or [])
        height = sampling.height or self.default_sample_size * self.vae_scale_factor
        width = sampling.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = sampling.num_inference_steps or 50

        ctx = self._prepare_generation_context(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            sigmas=sampling.sigmas,
            guidance_scale=sampling.guidance_scale if sampling.guidance_scale_provided else 1.0,
            num_images_per_prompt=sampling.num_outputs_per_prompt if sampling.num_outputs_per_prompt > 0 else 1,
            generator=sampling.generator,
            true_cfg_scale=sampling.true_cfg_scale or 4.0,
            max_sequence_length=sampling.max_sequence_length or self.tokenizer_max_length,
            attention_kwargs=kwargs.get("attention_kwargs"),
        )

        # prepare_timesteps() has already materialized request-specific timestep
        # state on self.scheduler, so deepcopy preserves dynamic-shifting state
        # without replaying set_timesteps() on the per-request scheduler.
        # Per-request scheduler (must not share state with self.scheduler)
        req_scheduler = copy.deepcopy(self.scheduler)
        req_scheduler.set_begin_index(0)

        # Populate state from generation context
        state.prompt_embeds = ctx["prompt_embeds"]
        state.prompt_embeds_mask = ctx["prompt_embeds_mask"]
        state.negative_prompt_embeds = ctx["negative_prompt_embeds"]
        state.negative_prompt_embeds_mask = ctx["negative_prompt_embeds_mask"]
        state.latents = ctx["latents"]
        state.timesteps = ctx["timesteps"]
        state.step_index = 0
        state.scheduler = req_scheduler
        state.do_true_cfg = ctx["do_true_cfg"]
        state.guidance = ctx["guidance"]
        state.img_shapes = ctx["img_shapes"]
        state.txt_seq_lens = ctx["txt_seq_lens"]
        state.negative_txt_seq_lens = ctx["negative_txt_seq_lens"]
        # QwenImage always normalizes CFG output (matching forward())
        state.sampling.cfg_normalize = True
        stage_end_s = time.time()
        trace_payload = {
            QWEN_IMAGE_STAGE_TRACE_KEY: [],
            "prompt_embeds": ctx["prompt_embeds"],
            "negative_prompt_embeds": ctx["negative_prompt_embeds"],
            "latents": ctx["latents"],
        }
        _append_qwen_image_stage_trace(
            trace_payload,
            stage="encode",
            request_id=state.request_id,
            start_s=stage_start_s,
            end_s=stage_end_s,
            extra={
                "height": int(height),
                "width": int(width),
                "num_inference_steps": int(num_inference_steps),
            },
        )
        state.extra["qwen_image_full_trace"] = {
            "height": int(height),
            "width": int(width),
            "stage_trace": list(trace_payload.get(QWEN_IMAGE_STAGE_TRACE_KEY) or []),
            "denoise_start_s": stage_end_s,
            "num_inference_steps": int(num_inference_steps),
            "timestep_count": len(ctx["timesteps"]),
        }

        return state

    def _build_denoise_kwargs(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        img_shapes: list,
        txt_seq_lens: list[int] | None,
        do_true_cfg: bool,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds_mask: torch.Tensor | None,
        negative_txt_seq_lens: list[int] | None,
        image_latents: torch.Tensor | None = None,
        extra_transformer_kwargs: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, int | None]:
        """Build positive/negative kwargs and output_slice for one denoise step.

        Returns:
            (positive_kwargs, negative_kwargs, output_slice)
        """
        extra_transformer_kwargs = extra_transformer_kwargs or {}

        # Broadcast timestep to match batch size
        t_for_model = timestep.expand(latents.shape[0]).to(
            device=latents.device,
            dtype=latents.dtype,
        )

        # Concatenate image latents if available (editing pipelines)
        latent_model_input = latents
        if image_latents is not None:
            latent_model_input = torch.cat([latents, image_latents], dim=1)

        positive_kwargs = {
            "hidden_states": latent_model_input,
            "timestep": t_for_model / 1000,
            "guidance": guidance,
            "encoder_hidden_states_mask": prompt_embeds_mask,
            "encoder_hidden_states": prompt_embeds,
            "img_shapes": img_shapes,
            "txt_seq_lens": txt_seq_lens,
            **extra_transformer_kwargs,
        }
        if do_true_cfg:
            negative_kwargs = {
                "hidden_states": latent_model_input,
                "timestep": t_for_model / 1000,
                "guidance": guidance,
                "encoder_hidden_states_mask": negative_prompt_embeds_mask,
                "encoder_hidden_states": negative_prompt_embeds,
                "img_shapes": img_shapes,
                "txt_seq_lens": negative_txt_seq_lens,
                **extra_transformer_kwargs,
            }
        else:
            negative_kwargs = None

        output_slice = latents.size(1) if image_latents is not None else None
        return positive_kwargs, negative_kwargs, output_slice

    def _decode_latents(
        self,
        latents: torch.Tensor,
        height: int,
        width: int,
        output_type: str = "pil",
        *,
        timing: dict[str, float] | None = None,
        sync_timing: bool = False,
    ) -> DiffusionOutput:
        """Unpack, normalize, and VAE-decode latents into a DiffusionOutput."""
        timing_start_s = time.time()
        last_s = timing_start_s

        def mark_timing(name: str) -> None:
            nonlocal last_s
            if timing is None:
                return
            _qwen_image_stage_timing_sync(sync_timing)
            now_s = time.time()
            timing[name] = max(0.0, now_s - last_s)
            last_s = now_s

        if timing is not None:
            _qwen_image_stage_timing_sync(sync_timing)
            timing_start_s = time.time()
            last_s = timing_start_s

        if output_type == "latent":
            if timing is not None:
                mark_timing("latent_passthrough_s")
                timing["total_s"] = max(0.0, time.time() - timing_start_s)
            return DiffusionOutput(
                output=latents,
                stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
            )

        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        mark_timing("unpack_s")
        latents = latents.to(self.vae.dtype)
        mark_timing("to_dtype_s")
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        mark_timing("stats_tensor_s")
        latents = latents / latents_std + latents_mean
        mark_timing("normalize_s")
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        mark_timing("vae_decode_s")
        if timing is not None:
            timing["total_s"] = max(0.0, time.time() - timing_start_s)
        return DiffusionOutput(
            output=image,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def _build_dynamic_denoise_kwargs(
        self,
        input_batch: "InputBatch",
        *,
        branch: str,
    ) -> dict[str, Any]:
        if input_batch.dynamic_latents is None:
            raise ValueError("Dynamic Qwen-Image denoise requires request-local latents.")

        latents = input_batch.dynamic_latents
        device = latents[0].device
        dtype = latents[0].dtype
        timestep = input_batch.timesteps.to(device=device, dtype=dtype).reshape(input_batch.num_reqs) / 1000
        guidance = None if input_batch.guidance is None else input_batch.guidance.to(device=device, dtype=dtype)

        if branch == "negative":
            encoder_hidden_states = input_batch.negative_prompt_embeds
            encoder_hidden_states_mask = input_batch.negative_prompt_embeds_mask
            txt_seq_lens = input_batch.negative_txt_seq_lens
        else:
            encoder_hidden_states = input_batch.prompt_embeds
            encoder_hidden_states_mask = input_batch.prompt_embeds_mask
            txt_seq_lens = input_batch.txt_seq_lens

        if encoder_hidden_states is None:
            raise ValueError(f"Dynamic Qwen-Image {branch} branch is missing prompt embeddings.")

        attention_kwargs = dict(self.attention_kwargs)
        attention_kwargs["attention_id"] = f"qwen_image.joint.{branch}"
        return {
            "hidden_states": latents,
            "timestep": timestep,
            "guidance": guidance,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "img_shapes": input_batch.img_shapes,
            "txt_seq_lens": txt_seq_lens,
            "attention_kwargs": attention_kwargs,
            "return_dict": False,
        }

    @staticmethod
    def _expect_dynamic_noise(noise_pred: Any, *, branch: str) -> list[torch.Tensor]:
        if not isinstance(noise_pred, list) or not all(torch.is_tensor(value) for value in noise_pred):
            raise ValueError(f"Dynamic Qwen-Image {branch} branch must return one tensor per request.")
        return noise_pred

    def _combine_dynamic_cfg(
        self,
        positive_noise_pred: list[torch.Tensor],
        negative_noise_pred: list[torch.Tensor],
        true_cfg_scales: list[float],
        cfg_normalize_flags: list[bool],
    ) -> list[torch.Tensor]:
        if len(positive_noise_pred) != len(negative_noise_pred):
            raise ValueError("Dynamic Qwen-Image CFG branch outputs have different request counts.")
        if len(true_cfg_scales) != len(positive_noise_pred):
            raise ValueError("Dynamic Qwen-Image CFG scales do not match request count.")

        combined: list[torch.Tensor] = []
        for positive, negative, scale, normalize in zip(
            positive_noise_pred,
            negative_noise_pred,
            true_cfg_scales,
            cfg_normalize_flags,
            strict=True,
        ):
            noise_pred = negative + float(scale) * (positive - negative)
            if normalize:
                noise_pred = self.cfg_normalize_function(positive, noise_pred)
            combined.append(noise_pred)
        return combined

    def _denoise_step_dynamic_batch(self, input_batch: "InputBatch") -> list[torch.Tensor]:
        positive_noise_pred = self._expect_dynamic_noise(
            self.predict_noise(**self._build_dynamic_denoise_kwargs(input_batch, branch="positive")),
            branch="positive",
        )
        if not input_batch.do_true_cfg:
            return positive_noise_pred

        negative_noise_pred = self._expect_dynamic_noise(
            self.predict_noise(**self._build_dynamic_denoise_kwargs(input_batch, branch="negative")),
            branch="negative",
        )
        return self._combine_dynamic_cfg(
            positive_noise_pred,
            negative_noise_pred,
            input_batch.true_cfg_scales,
            input_batch.cfg_normalize_flags,
        )

    def denoise_step(
        self,
        input_batch: "InputBatch",
        **kwargs: Any,
    ) -> torch.Tensor | None:
        """One denoise step: read from *input_batch*, delegate to CFGParallelMixin.

        Reuses ``predict_noise_maybe_with_cfg`` so that CFG-parallel,
        sequential-CFG, and no-CFG paths are handled identically to
        ``diffuse()``.
        """
        del kwargs
        trace_start_s = time.time()
        try:
            if self.interrupt:
                return None

            t = input_batch.timesteps
            self._current_timestep = t
            self.transformer.do_true_cfg = input_batch.do_true_cfg

            if input_batch.is_dynamic:
                return self._denoise_step_dynamic_batch(input_batch)

            positive_kwargs, negative_kwargs, output_slice = self._build_denoise_kwargs(
                latents=input_batch.latents,
                timestep=t,
                guidance=input_batch.guidance,
                prompt_embeds=input_batch.prompt_embeds,
                prompt_embeds_mask=input_batch.prompt_embeds_mask,
                img_shapes=input_batch.img_shapes,
                txt_seq_lens=input_batch.txt_seq_lens,
                do_true_cfg=input_batch.do_true_cfg,
                negative_prompt_embeds=input_batch.negative_prompt_embeds,
                negative_prompt_embeds_mask=input_batch.negative_prompt_embeds_mask,
                negative_txt_seq_lens=input_batch.negative_txt_seq_lens,
                image_latents=input_batch.image_latents,
                extra_transformer_kwargs={
                    "attention_kwargs": self.attention_kwargs,
                    "return_dict": False,
                },
            )

            return self.predict_noise_maybe_with_cfg(
                input_batch.do_true_cfg,
                input_batch.true_cfg_scale,
                positive_kwargs,
                negative_kwargs,
                input_batch.cfg_normalize,
                output_slice,
            )
        finally:
            _qwen_image_record_denoise_batch_trace(
                self,
                input_batch,
                start_s=trace_start_s,
                end_s=time.time(),
            )

    def step_scheduler(
        self,
        state: "DiffusionRequestState",
        noise_pred: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """One scheduler step: update ``state.latents`` and advance ``step_index``."""
        if self.interrupt:
            return

        t = state.current_timestep
        state.latents = self.scheduler_step_maybe_with_cfg(
            noise_pred,
            t,
            state.latents,
            state.do_true_cfg,
            per_request_scheduler=state.scheduler,
        )

        state.step_index += 1

    def post_decode(
        self,
        state: "DiffusionRequestState",
        **kwargs: Any,
    ) -> DiffusionOutput:
        """Decode final latents from *state*."""
        self._current_timestep = None

        height = state.sampling.height or self.default_sample_size * self.vae_scale_factor
        width = state.sampling.width or self.default_sample_size * self.vae_scale_factor
        output_type = kwargs.get("output_type", "pil")
        stage_meta = state.extra.get("qwen_image_full_trace", {})
        denoise_end_s = time.time()
        payload = {
            QWEN_IMAGE_STAGE_TRACE_KEY: list(stage_meta.get("stage_trace") or []),
            QWEN_IMAGE_DENOISE_BATCH_TRACE_KEY: _qwen_image_pop_denoise_batch_trace(self, state.request_id),
        }
        _append_qwen_image_stage_trace(
            payload,
            stage="denoise",
            request_id=state.request_id,
            start_s=float(stage_meta.get("denoise_start_s") or denoise_end_s),
            end_s=denoise_end_s,
            extra={
                "height": int(height),
                "width": int(width),
                "num_inference_steps": int(stage_meta.get("num_inference_steps") or 50),
                "timestep_count": int(stage_meta.get("timestep_count") or state.step_index),
            },
        )
        decode_start_s = time.time()
        fine_timing = _qwen_image_stage_env_flag("QWEN_IMAGE_STAGE_FINE_TIMING")
        sync_timing = _qwen_image_stage_env_flag("QWEN_IMAGE_STAGE_TIMING_SYNC")
        decode_timing: dict[str, float] = {}
        output = self._decode_latents(
            state.latents,
            height,
            width,
            output_type,
            timing=decode_timing if fine_timing else None,
            sync_timing=sync_timing,
        )
        trace_extra: dict[str, Any] = {
            "height": int(height),
            "width": int(width),
            "fine_timing": fine_timing,
            "sync_timing": sync_timing,
        }
        for name, value in decode_timing.items():
            trace_extra[f"decode_{name}"] = value
        _append_qwen_image_stage_trace(
            payload,
            stage="decode",
            request_id=state.request_id,
            start_s=decode_start_s,
            end_s=time.time(),
            extra=trace_extra,
        )
        output.custom_output = dict(output.custom_output or {})
        output.custom_output[QWEN_IMAGE_STAGE_TRACE_KEY] = list(payload.get(QWEN_IMAGE_STAGE_TRACE_KEY) or [])
        output.custom_output[QWEN_IMAGE_DENOISE_BATCH_TRACE_KEY] = list(
            payload.get(QWEN_IMAGE_DENOISE_BATCH_TRACE_KEY) or []
        )
        return output

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        true_cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 1024,
    ) -> DiffusionOutput:
        extracted_prompt, negative_prompt = self._extract_prompts(req.prompts)
        prompt = extracted_prompt or prompt

        height = req.sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = req.sampling_params.width or self.default_sample_size * self.vae_scale_factor
        height, width = normalize_min_aligned_size(height, width, self.vae_scale_factor * 2)
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        sigmas = req.sampling_params.sigmas or sigmas
        max_sequence_length = req.sampling_params.max_sequence_length or max_sequence_length
        generator = req.sampling_params.generator or generator
        true_cfg_scale = req.sampling_params.true_cfg_scale or true_cfg_scale
        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale
        num_images_per_prompt = (
            req.sampling_params.num_outputs_per_prompt
            if req.sampling_params.num_outputs_per_prompt > 0
            else num_images_per_prompt
        )

        ctx = self._prepare_generation_context(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            true_cfg_scale=true_cfg_scale,
            max_sequence_length=max_sequence_length,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            latents=latents,
            attention_kwargs=attention_kwargs,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        latents = self.diffuse(
            ctx["prompt_embeds"],
            ctx["prompt_embeds_mask"],
            ctx["negative_prompt_embeds"],
            ctx["negative_prompt_embeds_mask"],
            ctx["latents"],
            ctx["img_shapes"],
            ctx["txt_seq_lens"],
            ctx["negative_txt_seq_lens"],
            ctx["timesteps"],
            ctx["do_true_cfg"],
            ctx["guidance"],
            true_cfg_scale,
            image_latents=None,
            cfg_normalize=True,
            additional_transformer_kwargs={
                "return_dict": False,
                "attention_kwargs": self.attention_kwargs,
            },
        )

        self._current_timestep = None
        return self._decode_latents(latents, height, width, output_type)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


class QwenImageEncodePipeline(nn.Module, QwenImageCFGParallelMixin, DiffusionPipelineProfilerMixin):
    """Text/latent preparation stage for split Qwen-Image execution."""

    supports_step_execution: ClassVar[bool] = False
    EXTRA_OUTPUT_PARAMS: ClassVar[frozenset[str]] = frozenset(
        {QWEN_IMAGE_STAGE_TRACE_KEY, QWEN_IMAGE_DENOISE_BATCH_TRACE_KEY}
    )

    _extract_masked_hidden = QwenImagePipeline._extract_masked_hidden
    _get_qwen_prompt_embeds = QwenImagePipeline._get_qwen_prompt_embeds
    encode_prompt = QwenImagePipeline.encode_prompt
    _pack_latents = staticmethod(QwenImagePipeline._pack_latents)
    prepare_latents = QwenImagePipeline.prepare_latents
    prepare_timesteps = QwenImagePipeline.prepare_timesteps
    _extract_prompts = QwenImagePipeline._extract_prompts
    check_inputs = QwenImagePipeline.check_inputs

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        del prefix
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.weights_sources: list[DiffusersPipelineLoader.ComponentSource] = []
        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        prefetch_subfolders(
            model,
            ["scheduler", "text_encoder", "tokenizer"],
            local_files_only=local_files_only,
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, subfolder="text_encoder", local_files_only=local_files_only
        )
        visual_owner = None
        if hasattr(self.text_encoder, "model") and hasattr(self.text_encoder.model, "visual"):
            visual_owner = self.text_encoder.model
        elif hasattr(self.text_encoder, "visual"):
            visual_owner = self.text_encoder
        if visual_owner is not None:
            del visual_owner.visual
        else:
            logger.warning("Qwen-Image encode stage: vision tower not found on text encoder; skipping drop")
        self.text_encoder = self.text_encoder.to(self.device)
        self.tokenizer = Qwen2Tokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)

        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, QwenImageTransformer2DModel)
        self.num_channels_latents = int(transformer_kwargs.get("in_channels", 64)) // 4
        self.transformer_guidance_embeds = bool(transformer_kwargs.get("guidance_embeds", False))
        self.vae_scale_factor = _get_qwen_image_vae_scale_factor(model)
        self.tokenizer_max_length = 1024
        self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"  # noqa: E501
        self.prompt_template_encode_start_idx = 34
        self.default_sample_size = 128
        self._guidance_scale = 1.0
        self._attention_kwargs: dict[str, Any] = {}
        self._current_timestep = None
        self._interrupt = False
        self._num_timesteps = 0
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def _prepare_stage_payload(
        self,
        *,
        prompt,
        negative_prompt,
        height,
        width,
        num_inference_steps,
        sigmas,
        guidance_scale,
        num_images_per_prompt,
        generator,
        true_cfg_scale,
        max_sequence_length,
        output_type,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        latents=None,
        attention_kwargs=None,
    ) -> dict[str, Any]:
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds_mask,
            None,
            max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs or {}
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            batch_size = 1

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        self.check_cfg_parallel_validity(true_cfg_scale, has_neg_prompt)

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                prompt_name="negative_prompt",
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )

        img_shapes = [[(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)]] * batch_size
        timesteps, num_inference_steps = self.prepare_timesteps(
            num_inference_steps,
            sigmas,
            latents.shape[1],
        )
        self._num_timesteps = len(timesteps)

        if self.transformer_guidance_embeds:
            guidance = torch.full([1], guidance_scale, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        return {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
            "latents": latents,
            "img_shapes": img_shapes,
            "timesteps": timesteps,
            "num_inference_steps": num_inference_steps,
            "sigmas": sigmas,
            "do_true_cfg": do_true_cfg,
            "guidance": guidance,
            "txt_seq_lens": txt_seq_lens,
            "negative_txt_seq_lens": negative_txt_seq_lens,
            "true_cfg_scale": true_cfg_scale,
            "cfg_normalize": True,
            "height": height,
            "width": width,
            "output_type": output_type,
            "attention_kwargs": self._attention_kwargs,
        }

    def forward(
        self,
        req: OmniDiffusionRequest,
        **kwargs: Any,
    ) -> DiffusionOutput:
        stage_start_s = time.time()
        sampling = req.sampling_params
        if sampling.generator is None and sampling.seed is not None:
            gen_device = sampling.generator_device or ("cpu" if self.device.type == "cpu" else self.device)
            sampling.generator = torch.Generator(device=gen_device).manual_seed(sampling.seed)

        prompt, negative_prompt = self._extract_prompts(req.prompts)
        height = sampling.height or self.default_sample_size * self.vae_scale_factor
        width = sampling.width or self.default_sample_size * self.vae_scale_factor
        height, width = normalize_min_aligned_size(height, width, self.vae_scale_factor * 2)
        num_images_per_prompt = sampling.num_outputs_per_prompt if sampling.num_outputs_per_prompt > 0 else 1
        guidance_scale = sampling.guidance_scale if sampling.guidance_scale_provided else 1.0

        payload = self._prepare_stage_payload(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=sampling.num_inference_steps or 50,
            sigmas=sampling.sigmas,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=sampling.generator,
            true_cfg_scale=sampling.true_cfg_scale or 4.0,
            max_sequence_length=sampling.max_sequence_length or self.tokenizer_max_length,
            output_type=kwargs.get("output_type", "pil"),
            attention_kwargs=kwargs.get("attention_kwargs"),
        )
        stage_end_s = time.time()
        _append_qwen_image_stage_trace(
            payload,
            stage="encode",
            request_id=req.request_id,
            start_s=stage_start_s,
            end_s=stage_end_s,
            extra={
                "height": height,
                "width": width,
                "num_inference_steps": int(sampling.num_inference_steps or 50),
            },
        )
        return _stage_output(payload, "encode")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        del weights
        return None


class QwenImageDenoisePipeline(nn.Module, QwenImageCFGParallelMixin, DiffusionPipelineProfilerMixin):
    """DiT-only step execution stage for split Qwen-Image pipelines."""

    supports_step_execution: ClassVar[bool] = True
    EXTRA_OUTPUT_PARAMS: ClassVar[frozenset[str]] = frozenset(
        {QWEN_IMAGE_STAGE_TRACE_KEY, QWEN_IMAGE_DENOISE_BATCH_TRACE_KEY}
    )

    prepare_timesteps = QwenImagePipeline.prepare_timesteps
    _build_denoise_kwargs = QwenImagePipeline._build_denoise_kwargs
    _build_dynamic_denoise_kwargs = QwenImagePipeline._build_dynamic_denoise_kwargs
    _expect_dynamic_noise = staticmethod(QwenImagePipeline._expect_dynamic_noise)
    _combine_dynamic_cfg = QwenImagePipeline._combine_dynamic_cfg
    _denoise_step_dynamic_batch = QwenImagePipeline._denoise_step_dynamic_batch
    denoise_step = QwenImagePipeline.denoise_step
    step_scheduler = QwenImagePipeline.step_scheduler
    guidance_scale = QwenImagePipeline.guidance_scale
    attention_kwargs = QwenImagePipeline.attention_kwargs
    num_timesteps = QwenImagePipeline.num_timesteps
    current_timestep = QwenImagePipeline.current_timestep
    interrupt = QwenImagePipeline.interrupt

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        del prefix
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]
        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)
        prefetch_subfolders(model, ["scheduler"], local_files_only=local_files_only)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )
        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, QwenImageTransformer2DModel)
        self.transformer = QwenImageTransformer2DModel(
            od_config=od_config, quant_config=od_config.quantization_config, **transformer_kwargs
        )
        self.vae_scale_factor = _get_qwen_image_vae_scale_factor(model)
        self.default_sample_size = 128
        self._guidance_scale = 1.0
        self._attention_kwargs: dict[str, Any] = {}
        self._current_timestep = None
        self._interrupt = False
        self._num_timesteps = 0
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def prepare_encode(
        self,
        state: "DiffusionRequestState",
        **kwargs: Any,
    ) -> "DiffusionRequestState":
        del kwargs
        import_start_s = time.time()
        if state.request_id == DUMMY_DIFFUSION_REQUEST_ID:
            payload = _make_dummy_encode_payload(
                height=state.sampling.height or self.default_sample_size * self.vae_scale_factor,
                width=state.sampling.width or self.default_sample_size * self.vae_scale_factor,
                num_inference_steps=state.sampling.num_inference_steps or 1,
                true_cfg_scale=state.sampling.true_cfg_scale or 4.0,
            )
        else:
            payload = _extract_qwen_image_stage_payload(state.prompts, "encode")
        transfer_trace = _qwen_image_stage_env_flag("QWEN_IMAGE_STAGE_TRANSFER_TRACE")
        sync_timing = _qwen_image_stage_env_flag("QWEN_IMAGE_STAGE_TIMING_SYNC")
        if transfer_trace:
            _qwen_image_stage_timing_sync(sync_timing)
            import_start_s = time.time()
        state.prompt_embeds = payload["prompt_embeds"].to(self.device)
        state.prompt_embeds_mask = payload["prompt_embeds_mask"].to(self.device)
        negative_prompt_embeds = payload.get("negative_prompt_embeds")
        negative_prompt_embeds_mask = payload.get("negative_prompt_embeds_mask")
        state.negative_prompt_embeds = (
            None if negative_prompt_embeds is None else negative_prompt_embeds.to(self.device)
        )
        state.negative_prompt_embeds_mask = (
            None if negative_prompt_embeds_mask is None else negative_prompt_embeds_mask.to(self.device)
        )
        state.latents = payload["latents"].to(self.device)
        state.img_shapes = _normalize_qwen_image_img_shapes(payload["img_shapes"])
        state.do_true_cfg = bool(payload["do_true_cfg"])
        guidance = payload.get("guidance")
        state.guidance = None if guidance is None else guidance.to(self.device)
        state.txt_seq_lens = payload.get("txt_seq_lens")
        state.negative_txt_seq_lens = payload.get("negative_txt_seq_lens")
        if transfer_trace:
            _qwen_image_stage_timing_sync(sync_timing)
            _append_qwen_image_stage_trace(
                payload,
                stage="encode_payload_import",
                request_id=state.request_id,
                start_s=import_start_s,
                end_s=time.time(),
                extra={
                    "source_device": "cpu",
                    "target_device": str(self.device),
                    "payload_kind": "encode",
                    "sync_timing": sync_timing,
                },
            )
        state.step_index = 0
        state.sampling.true_cfg_scale = float(payload.get("true_cfg_scale", state.sampling.true_cfg_scale or 4.0))
        state.sampling.cfg_normalize = bool(payload.get("cfg_normalize", True))
        self._attention_kwargs = payload.get("attention_kwargs") or {}

        timesteps, _ = self.prepare_timesteps(
            int(payload.get("num_inference_steps") or 50),
            payload.get("sigmas"),
            state.latents.shape[1],
        )
        self._num_timesteps = len(timesteps)
        state.timesteps = timesteps
        req_scheduler = copy.deepcopy(self.scheduler)
        req_scheduler.set_begin_index(0)
        state.scheduler = req_scheduler
        state.extra["qwen_image_stage"] = {
            "height": int(payload["height"]),
            "width": int(payload["width"]),
            "output_type": payload.get("output_type", "pil"),
            "stage_trace": list(payload.get(QWEN_IMAGE_STAGE_TRACE_KEY) or []),
            "denoise_start_s": time.time(),
            "num_inference_steps": int(payload.get("num_inference_steps") or 50),
            "timestep_count": len(timesteps),
        }
        return state

    def denoise_step(
        self,
        input_batch: "InputBatch",
        **kwargs: Any,
    ) -> torch.Tensor | list[torch.Tensor] | None:
        if all(request_id == DUMMY_DIFFUSION_REQUEST_ID for request_id in input_batch.request_ids):
            if input_batch.is_dynamic:
                if input_batch.dynamic_latents is None:
                    raise ValueError("Dummy dynamic Qwen-Image denoise requires request-local latents.")
                return [torch.zeros_like(latent) for latent in input_batch.dynamic_latents]
            if input_batch.latents is None:
                raise ValueError("Dummy dense Qwen-Image denoise requires dense latents.")
            return torch.zeros_like(input_batch.latents)
        return QwenImagePipeline.denoise_step(self, input_batch, **kwargs)

    def post_decode(
        self,
        state: "DiffusionRequestState",
        **kwargs: Any,
    ) -> DiffusionOutput:
        del kwargs
        self._current_timestep = None
        stage_meta = state.extra.get("qwen_image_stage", {})
        payload = {
            "latents": state.latents,
            "height": int(
                stage_meta.get("height") or state.sampling.height or self.default_sample_size * self.vae_scale_factor
            ),
            "width": int(
                stage_meta.get("width") or state.sampling.width or self.default_sample_size * self.vae_scale_factor
            ),
            "output_type": stage_meta.get("output_type", "pil"),
            QWEN_IMAGE_STAGE_TRACE_KEY: list(stage_meta.get("stage_trace") or []),
            QWEN_IMAGE_DENOISE_BATCH_TRACE_KEY: _qwen_image_pop_denoise_batch_trace(self, state.request_id),
        }
        _append_qwen_image_stage_trace(
            payload,
            stage="denoise",
            request_id=state.request_id,
            start_s=float(stage_meta.get("denoise_start_s") or time.time()),
            end_s=time.time(),
            extra={
                "height": int(payload["height"]),
                "width": int(payload["width"]),
                "num_inference_steps": int(stage_meta.get("num_inference_steps") or 50),
                "timestep_count": int(stage_meta.get("timestep_count") or 0),
            },
        )
        return _stage_output(payload, "denoise")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


class QwenImageDecodePipeline(nn.Module, DiffusionPipelineProfilerMixin):
    """VAE decode stage for split Qwen-Image execution."""

    supports_step_execution: ClassVar[bool] = False
    EXTRA_OUTPUT_PARAMS: ClassVar[frozenset[str]] = frozenset(
        {QWEN_IMAGE_STAGE_TRACE_KEY, QWEN_IMAGE_DENOISE_BATCH_TRACE_KEY}
    )

    _unpack_latents = staticmethod(QwenImagePipeline._unpack_latents)
    _decode_latents = QwenImagePipeline._decode_latents

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        del prefix
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.weights_sources: list[DiffusersPipelineLoader.ComponentSource] = []
        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)
        prefetch_subfolders(model, ["vae"], local_files_only=local_files_only)
        self.vae = DistributedAutoencoderKLQwenImage.from_pretrained(
            model, subfolder="vae", local_files_only=local_files_only
        ).to(self.device)
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.default_sample_size = 128
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def forward(
        self,
        req: OmniDiffusionRequest,
        **kwargs: Any,
    ) -> DiffusionOutput:
        del kwargs
        stage_start_s = time.time()
        import_start_s = stage_start_s
        if req.request_id == DUMMY_DIFFUSION_REQUEST_ID:
            payload = _make_dummy_denoise_payload(
                height=req.sampling_params.height or self.default_sample_size * self.vae_scale_factor,
                width=req.sampling_params.width or self.default_sample_size * self.vae_scale_factor,
            )
        else:
            payload = _extract_qwen_image_stage_payload(req.prompts, "denoise")
        transfer_trace = _qwen_image_stage_env_flag("QWEN_IMAGE_STAGE_TRANSFER_TRACE")
        sync_timing = _qwen_image_stage_env_flag("QWEN_IMAGE_STAGE_TIMING_SYNC")
        if transfer_trace:
            _qwen_image_stage_timing_sync(sync_timing)
            import_start_s = time.time()
        latents = payload["latents"].to(self.device)
        if transfer_trace:
            _qwen_image_stage_timing_sync(sync_timing)
            _append_qwen_image_stage_trace(
                payload,
                stage="denoise_payload_import",
                request_id=req.request_id,
                start_s=import_start_s,
                end_s=time.time(),
                extra={
                    "source_device": "cpu",
                    "target_device": str(self.device),
                    "payload_kind": "denoise",
                    "sync_timing": sync_timing,
                },
            )
        if payload.get("output_type") == "dummy_image":
            image = torch.zeros(1, 3, 16, 16, device=self.device, dtype=self.vae.dtype)
            return DiffusionOutput(output=image)
        fine_timing = _qwen_image_stage_env_flag("QWEN_IMAGE_STAGE_FINE_TIMING")
        decode_timing: dict[str, float] = {}
        output = self._decode_latents(
            latents,
            int(payload["height"]),
            int(payload["width"]),
            payload.get("output_type", "pil"),
            timing=decode_timing if fine_timing else None,
            sync_timing=sync_timing,
        )
        trace_extra: dict[str, Any] = {
            "height": int(payload["height"]),
            "width": int(payload["width"]),
            "fine_timing": fine_timing,
            "sync_timing": sync_timing,
        }
        for name, value in decode_timing.items():
            trace_extra[f"decode_{name}"] = value
        _append_qwen_image_stage_trace(
            payload,
            stage="decode",
            request_id=req.request_id,
            start_s=stage_start_s,
            end_s=time.time(),
            extra=trace_extra,
        )
        output.custom_output = dict(output.custom_output or {})
        output.custom_output[QWEN_IMAGE_STAGE_TRACE_KEY] = list(payload.get(QWEN_IMAGE_STAGE_TRACE_KEY) or [])
        output.custom_output[QWEN_IMAGE_DENOISE_BATCH_TRACE_KEY] = list(
            payload.get(QWEN_IMAGE_DENOISE_BATCH_TRACE_KEY) or []
        )
        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        del weights
        return None


class QwenImageDMD2Pipeline(DMD2PipelineMixin, QwenImagePipeline):
    """QwenImage pipeline for FastGen DMD2-distilled models."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.__init_dmd2__()
