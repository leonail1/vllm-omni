# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import (
    QWEN_IMAGE_STAGE_KIND_KEY,
    QWEN_IMAGE_STAGE_PAYLOAD_KEY,
    QWEN_IMAGE_STAGE_TRACE_KEY,
    QwenImageDecodePipeline,
    QwenImageDenoisePipeline,
    QwenImageEncodePipeline,
    _append_qwen_image_stage_trace,
    _normalize_qwen_image_img_shapes,
    _qwen_image_payload_nbytes,
    _stage_output,
)
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import get_sampling_params_key
from vllm_omni.diffusion.sched.step_scheduler import StepScheduler
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.entrypoints.openai.api_server import _is_diffusion_only_stage_configs
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.entrypoints.utils import load_and_resolve_stage_configs
from vllm_omni.engine.stage_init_utils import build_diffusion_config, extract_stage_metadata
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.stage_input_processors.qwen_image_stage_pipeline import (
    denoise_to_decode,
    encode_to_denoise,
)
from vllm_omni.outputs import OmniRequestOutput
from tests.helpers.runtime import OmniServerStageCli


def test_qwen_image_stage_bridge_preserves_payload() -> None:
    payload = {"latents": torch.ones(1, 2), "height": 1024, "width": 1024}
    output = _stage_output(payload, "encode")
    source_output = SimpleNamespace(custom_output=output.custom_output)

    next_prompt = encode_to_denoise([source_output])

    assert next_prompt[QWEN_IMAGE_STAGE_KIND_KEY] == "encode"
    assert torch.equal(next_prompt[QWEN_IMAGE_STAGE_PAYLOAD_KEY]["latents"], payload["latents"])


def test_qwen_image_stage_trace_records_transport_size() -> None:
    payload = {"latents": torch.zeros(2, 3, dtype=torch.float16), "height": 256}

    _append_qwen_image_stage_trace(
        payload,
        stage="encode",
        request_id="req-trace",
        start_s=10.0,
        end_s=12.5,
    )

    assert _qwen_image_payload_nbytes(payload) == 12
    assert payload[QWEN_IMAGE_STAGE_TRACE_KEY] == [
        {
            "stage": "encode",
            "request_id": "req-trace",
            "start_s": 10.0,
            "end_s": 12.5,
            "duration_s": 2.5,
            "payload_bytes": 12,
        }
    ]


def test_qwen_image_stage_trace_records_scalar_extra_fields() -> None:
    payload = {"latents": torch.zeros(1, 1, dtype=torch.float16)}

    _append_qwen_image_stage_trace(
        payload,
        stage="decode",
        request_id="req-extra",
        start_s=1.0,
        end_s=2.0,
        extra={
            "height": 1024,
            "width": 1024,
            "sync_timing": True,
            "decode_vae_decode_s": 0.25,
            "non_scalar": {"ignored": True},
        },
    )

    event = payload[QWEN_IMAGE_STAGE_TRACE_KEY][0]
    assert event["height"] == 1024
    assert event["width"] == 1024
    assert event["sync_timing"] is True
    assert event["decode_vae_decode_s"] == 0.25
    assert "non_scalar" not in event


def test_omni_request_output_prefers_outer_custom_output() -> None:
    inner = OmniRequestOutput.from_diffusion(
        request_id="req-inner",
        images=[],
        custom_output={},
    )
    outer = OmniRequestOutput(
        request_id="req-outer",
        request_output=inner,
        _custom_output={QWEN_IMAGE_STAGE_TRACE_KEY: [{"stage": "decode"}]},
    )

    assert outer.custom_output == {QWEN_IMAGE_STAGE_TRACE_KEY: [{"stage": "decode"}]}


def test_chat_serving_resolves_extra_output_params_from_final_stage_config() -> None:
    serving = object.__new__(OmniOpenAIServingChat)
    serving.engine_client = None
    serving._diffusion_extra_output_params = None
    serving._diffusion_engine = SimpleNamespace(
        engine=SimpleNamespace(
            stage_configs=[
                SimpleNamespace(
                    final_output=False,
                    engine_args=SimpleNamespace(model_class_name="QwenImageEncodePipeline"),
                ),
                SimpleNamespace(
                    final_output=True,
                    engine_args=SimpleNamespace(model_class_name="QwenImageDecodePipeline"),
                ),
            ]
        )
    )

    exposed = OmniOpenAIServingChat._get_diffusion_extra_output_params(
        serving,
        {QWEN_IMAGE_STAGE_TRACE_KEY: [{"stage": "decode"}]},
    )

    assert exposed == {QWEN_IMAGE_STAGE_TRACE_KEY: [{"stage": "decode"}]}


def test_chat_serving_does_not_expose_stage_payload() -> None:
    serving = object.__new__(OmniOpenAIServingChat)
    serving.engine_client = None
    serving._diffusion_extra_output_params = None
    serving._diffusion_engine = SimpleNamespace(
        engine=SimpleNamespace(
            stage_configs=[
                SimpleNamespace(
                    final_output=True,
                    engine_args=SimpleNamespace(model_class_name="QwenImageDecodePipeline"),
                )
            ]
        )
    )

    exposed = OmniOpenAIServingChat._get_diffusion_extra_output_params(
        serving,
        {
            QWEN_IMAGE_STAGE_PAYLOAD_KEY: {"latents": torch.ones(1, 2)},
            QWEN_IMAGE_STAGE_TRACE_KEY: [{"stage": "decode"}],
        },
    )

    assert exposed == {QWEN_IMAGE_STAGE_TRACE_KEY: [{"stage": "decode"}]}


def test_qwen_image_stage_img_shapes_accept_transport_lists() -> None:
    assert _normalize_qwen_image_img_shapes([[1, 16, 16]]) == [[(1, 16, 16)]]
    assert _normalize_qwen_image_img_shapes([[[1, 16, 16]]]) == [[(1, 16, 16)]]
    assert _normalize_qwen_image_img_shapes((1, 16, 16)) == [[(1, 16, 16)]]


def test_qwen_image_stage_bridge_rejects_wrong_kind() -> None:
    output = _stage_output({"latents": torch.ones(1, 2)}, "encode")
    source_output = SimpleNamespace(custom_output=output.custom_output)

    try:
        denoise_to_decode([source_output])
    except RuntimeError as exc:
        assert "expected 'denoise' payload" in str(exc)
    else:
        raise AssertionError("denoise_to_decode accepted an encode payload")


def test_qwen_image_denoise_pipeline_keeps_dynamic_batch_support() -> None:
    runner = object.__new__(DiffusionModelRunner)
    runner.pipeline = object.__new__(QwenImageDenoisePipeline)
    runner.od_config = SimpleNamespace(step_execution=True, max_num_seqs=2)

    assert DiffusionModelRunner._supports_qwen_image_dynamic_step_batching(runner)


def test_qwen_image_stage_copied_static_methods_stay_unbound() -> None:
    encode = object.__new__(QwenImageEncodePipeline)
    denoise = object.__new__(QwenImageDenoisePipeline)
    decode = object.__new__(QwenImageDecodePipeline)

    assert encode._pack_latents is QwenImageEncodePipeline._pack_latents
    assert denoise._expect_dynamic_noise is QwenImageDenoisePipeline._expect_dynamic_noise
    assert decode._unpack_latents is QwenImageDecodePipeline._unpack_latents


def test_qwen_image_stage_deploy_uses_floating_default_dtype() -> None:
    repo = Path(__file__).resolve().parents[4]
    cfg = OmegaConf.load(repo / "vllm_omni/deploy/qwen_image_stage_pipeline.yaml")

    for stage in cfg.stage_args:
        assert stage.engine_args.dtype == "bfloat16"
        assert stage.engine_args.kv_cache_dtype is None


def test_qwen_image_stage_1x2x1_deploy_declares_shared_small_roles() -> None:
    repo = Path(__file__).resolve().parents[4]
    cfg = OmegaConf.load(repo / "vllm_omni/deploy/qwen_image_stage_pipeline_1x2x1.yaml")
    stages = {stage.stage_id: stage for stage in cfg.stage_args}

    assert stages[0].engine_args.model_class_name == "QwenImageEncodePipeline"
    assert stages[1].engine_args.model_class_name == "QwenImageDenoisePipeline"
    assert stages[1].num_replicas == 2
    assert stages[1].runtime.devices == "1,2"
    assert stages[2].engine_args.model_class_name == "QwenImageDecodePipeline"


def test_stage_cli_uses_local_dp_for_multi_replica_headless_stage() -> None:
    server = object.__new__(OmniServerStageCli)
    server.model = "Qwen/Qwen-Image"
    server.stage_config_path = "/tmp/qwen_image_stage_pipeline_1x2x1.yaml"
    server.host = "127.0.0.1"
    server.port = 8000
    server.master_port = 9000
    server.serve_args = []

    cmd = OmniServerStageCli._build_stage_cmd(
        server,
        1,
        headless=True,
        replica_id=0,
        omni_dp_size_local=2,
    )

    assert "--headless" in cmd
    assert cmd[cmd.index("--omni-dp-size-local") + 1] == "2"


def test_stage_cli_unknown_stage_type_accepts_diffusion_ready_marker(tmp_path) -> None:
    log_path = tmp_path / "stage1.log"
    log_path.write_text(
        "\n".join(
            [
                "[Headless] Diffusion replica id=0 for stage 1 is up",
                "[Headless] Diffusion replica id=1 for stage 1 is up",
            ]
        )
    )

    server = object.__new__(OmniServerStageCli)
    server.stage_ids = [0, 1]
    server.stage_replica_counts = {1: 2}
    server.stage_types = {1: "unknown"}
    server._stage_log_paths = {(1, 0): log_path}

    assert OmniServerStageCli._headless_stage_replicas_ready(server)


def test_multi_stage_diffusion_uses_diffusion_only_serving_path() -> None:
    stages = [
        SimpleNamespace(stage_type="diffusion"),
        SimpleNamespace(stage_type="diffusion"),
        SimpleNamespace(stage_type="diffusion"),
    ]

    assert _is_diffusion_only_stage_configs(stages)
    assert not _is_diffusion_only_stage_configs([stages[0], SimpleNamespace(stage_type="llm")])


def test_qwen_image_stage_deploy_clears_ar_kv_cache_default() -> None:
    repo = Path(__file__).resolve().parents[4]
    config_path = repo / "vllm_omni/deploy/qwen_image_stage_pipeline.yaml"

    _, stages = load_and_resolve_stage_configs(
        "Qwen/Qwen-Image",
        str(config_path),
        {
            "kv_cache_dtype": "auto",
            "quantization_config": "int8",
            "stage_configs_path": str(config_path),
        },
    )
    denoise_stage = next(stage for stage in stages if stage.stage_id == 1)
    od_config = build_diffusion_config(
        "Qwen/Qwen-Image",
        denoise_stage,
        extract_stage_metadata(denoise_stage),
    )

    assert od_config.diffusion_kv_cache_dtype is None


def test_qwen_image_denoise_sampling_key_uses_stage_payload_cfg_rule() -> None:
    request = OmniDiffusionRequest(
        prompts=[
            {
                QWEN_IMAGE_STAGE_KIND_KEY: "encode",
                QWEN_IMAGE_STAGE_PAYLOAD_KEY: {"do_true_cfg": True},
            }
        ],
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=2,
            true_cfg_scale=4.0,
        ),
        request_id="req-qwen-image-denoise-key",
    )
    od_config = SimpleNamespace(
        step_execution=True,
        model_class_name="QwenImageDenoisePipeline",
    )

    key = get_sampling_params_key(request, od_config)

    assert key is not None
    assert key.do_classifier_free_guidance is True


def test_qwen_image_denoise_dummy_step_does_not_call_transformer() -> None:
    pipeline = object.__new__(QwenImageDenoisePipeline)
    batch = SimpleNamespace(
        request_ids=[DUMMY_DIFFUSION_REQUEST_ID],
        is_dynamic=False,
        latents=torch.ones(1, 2, 4),
    )

    noise_pred = QwenImageDenoisePipeline.denoise_step(pipeline, batch)

    assert torch.equal(noise_pred, torch.zeros_like(batch.latents))


def test_step_scheduler_reads_total_steps_from_stage_payload() -> None:
    request = OmniDiffusionRequest(
        prompts=[
            {
                QWEN_IMAGE_STAGE_KIND_KEY: "encode",
                QWEN_IMAGE_STAGE_PAYLOAD_KEY: {
                    "num_inference_steps": 7,
                    "do_true_cfg": False,
                },
            }
        ],
        sampling_params=OmniDiffusionSamplingParams(),
        request_id="req-qwen-image-denoise-steps",
    )

    scheduler = StepScheduler()

    assert scheduler._get_total_steps(request) == 7
