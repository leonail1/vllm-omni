# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Diffusion Model Runner for vLLM-Omni.

Handles model loading, compilation, caching, and execution of diffusion model
forward passes. This follows the AR pattern where the Runner handles all
model-related operations.
"""

from __future__ import annotations

import copy
import gc
import time
from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any

import torch
from torch.profiler import record_function
from vllm.config import LoadConfig
from vllm.logger import init_logger
from vllm.utils.mem_utils import DeviceMemoryProfiler, GiB_bytes

from vllm_omni.diffusion.cache.cache_dit_backend import cache_summary
from vllm_omni.diffusion.cache.prompt_embed_cache import (
    install_prompt_embed_cache,
    resolve_prompt_embed_cache_config,
)
from vllm_omni.diffusion.cache.selector import get_cache_backend
from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import supports_diffusion_atoms
from vllm_omni.diffusion.offloader import get_offload_backend
from vllm_omni.diffusion.registry import _NO_CACHE_ACCELERATION
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput
from vllm_omni.diffusion.worker.input_batch import InputBatch, scatter_latents
from vllm_omni.diffusion.worker.utils import (
    BatchRunnerOutput,
    DiffusionRequestState,
    DiffusionStateTransport,
    RunnerOutput,
    TransportBoundary,
)
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import OmniKVTransferManager
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

logger = init_logger(__name__)


class DiffusionModelRunner(OmniConnectorModelRunnerMixin):
    """
    Model runner that handles model loading and execution for diffusion models.

    This class follows the AR pattern where the Runner handles all model-related
    operations including loading, compilation, offloading, caching, and execution.
    The Worker only handles infrastructure (device, distributed env).
    """

    def __init__(
        self,
        vllm_config,
        od_config: OmniDiffusionConfig,
        device: torch.device,
    ):
        """
        Initialize the diffusion model runner.

        Args:
            vllm_config: vLLM configuration.
            od_config: OmniDiffusion configuration.
            device: The device to run on.
        """
        self.vllm_config = vllm_config
        self.od_config = od_config
        self.device = device
        self.pipeline = None
        self.cache_backend = None
        self.offload_backend = None
        self.prompt_embed_cache = None

        # Cache for per-request stepwise state.
        self.state_cache: dict[str, DiffusionRequestState] = {}

        # Initialize KV cache manager for connector management
        self.kv_transfer_manager = OmniKVTransferManager.from_od_config(od_config)

    def _normalize_step_execution_mode(self) -> None:
        from vllm_omni.diffusion.stage_kind import DiffusionStageRole, normalize_diffusion_stage_role

        role = normalize_diffusion_stage_role(getattr(self.od_config, "diffusion_stage_role", "monolithic"))
        self.od_config.diffusion_stage_role = role.value
        self.od_config.step_execution = role != DiffusionStageRole.MONOLITHIC

    def _compile_transformer(self, attr_name: str) -> None:
        """Compile a transformer attribute on the pipeline with torch.compile."""
        model = getattr(self.pipeline, attr_name, None)
        if model is None:
            return
        try:
            setattr(self.pipeline, attr_name, regionally_compile(model, dynamic=True))
            logger.info("Model runner: %s compiled with torch.compile.", attr_name)
        except Exception as e:
            logger.warning(
                "Model runner: torch.compile for %s failed: %s. Using eager mode.",
                attr_name,
                e,
            )

    def load_model(
        self,
        memory_pool_context_fn: callable | None = None,
        load_format: str = "default",
        custom_pipeline_name: str | None = None,
    ) -> None:
        """
        Load the diffusion model, apply compilation and offloading.

        Args:
            memory_pool_context_fn: Optional function that returns a context manager
                for memory pool allocation (used for sleep mode).
            load_format: Format for loading model weights. Supported formats:
                - "default" (default): Automatically detect and use the default format based on configuration
                - "custom_pipeline": Init model from a custom pipeline class specified by `custom_pipeline_name`
                - "dummy": Skip actual weight loading, useful for testing and custom pipelines that
                    don't require default weights.
            custom_pipeline_name: Optional custom pipeline class name to use.
        """

        self._normalize_step_execution_mode()
        if load_format == "dummy":
            return

        load_device = (
            "cpu" if self.od_config.enable_cpu_offload or self.od_config.enable_layerwise_offload else str(self.device)
        )

        def get_memory_context():
            if memory_pool_context_fn is not None:
                return memory_pool_context_fn(tag="weights")
            return nullcontext()

        # Load model within forward context
        load_config = LoadConfig()
        model_loader = DiffusersPipelineLoader(load_config, od_config=self.od_config)
        time_before_load = time.perf_counter()

        with get_memory_context():
            with DeviceMemoryProfiler() as m:
                self.pipeline = model_loader.load_model(
                    load_device=load_device,
                    load_format=load_format,
                    custom_pipeline_name=custom_pipeline_name,
                    device=self.device,
                )
        time_after_load = time.perf_counter()

        logger.info(
            "Model loading took %.4f GiB and %.6f seconds",
            m.consumed_memory / GiB_bytes,
            time_after_load - time_before_load,
        )
        logger.info("Model runner: Model loaded successfully.")

        if getattr(self.od_config, "step_execution", False) and not self.supports_step_mode():
            raise ValueError(
                "step_execution=True requires a pipeline implementing "
                "the diffusion atom contract "
                "(init_state, validation, encoding, preparation, predict_noise, "
                "advance_scheduler, decoding, postprocess); "
                f"{self.od_config.model_class_name} does not support that contract."
            )

        self._configure_pipeline_stage_role()

        # Apply CPU offloading
        self.offload_backend = get_offload_backend(self.od_config, device=self.device)
        if self.offload_backend is not None:
            logger.info(f" Enabling offloader backend: {self.offload_backend.__class__.__name__}")
            self.offload_backend.enable(self.pipeline)

        # Apply torch.compile if not in eager mode
        if not self.od_config.enforce_eager:
            if current_omni_platform.supports_torch_inductor():
                self._compile_transformer("transformer")
                self._compile_transformer("transformer_2")
            else:
                logger.warning(
                    "Model runner: Platform %s does not support torch inductor, skipping torch.compile.",
                    current_omni_platform.get_torch_device(),
                )

        # Setup cache backend
        self.cache_backend = get_cache_backend(self.od_config.cache_backend, self.od_config.cache_config)

        if self.cache_backend is not None:
            if self.od_config.model_class_name in _NO_CACHE_ACCELERATION:
                logger.warning(
                    "Cache backend '%s' is not supported for %s; disabling cache acceleration.",
                    self.od_config.cache_backend,
                    self.od_config.model_class_name,
                )
                self.cache_backend = None
                self.od_config.cache_backend = None
            else:
                self.cache_backend.enable(self.pipeline)

        # Install prompt-embedding cache (transparent wrapper around
        # ``pipeline.encode_prompt``). Enabled via config or env var; a no-op
        # when the pipeline does not expose ``encode_prompt``.
        enable_pec, pec_size = resolve_prompt_embed_cache_config(
            enable=getattr(self.od_config, "enable_prompt_embed_cache", False),
            max_size=getattr(self.od_config, "prompt_embed_cache_size", 32),
        )
        if enable_pec:
            self.prompt_embed_cache = install_prompt_embed_cache(
                self.pipeline,
                max_size=pec_size,
                enabled=True,
                model_tag=self.od_config.model_class_name,
            )

        logger.info("Model runner: Initialization complete.")

    def _configure_pipeline_stage_role(self) -> None:
        role = getattr(self.od_config, "diffusion_stage_role", "monolithic")
        configure = getattr(self.pipeline, "configure_diffusion_stage_role", None)
        if not callable(configure):
            return

        configure(role)
        gc.collect()
        current_omni_platform.empty_cache()
        logger.info("Model runner: configured diffusion stage role %s.", role)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights into the pipeline."""
        return self.pipeline.load_weights(weights)

    def clear_prompt_embed_cache(self) -> None:
        """Evict all cached text-encoder outputs (e.g. between training epochs)."""
        if self.prompt_embed_cache is not None:
            self.prompt_embed_cache.clear()

    def get_prompt_embed_cache_stats(self) -> dict | None:
        """Return hit/miss statistics for the prompt-embedding cache, if enabled."""
        if self.prompt_embed_cache is None:
            return None
        return self.prompt_embed_cache.stats()

    def _record_peak_memory(self, output: DiffusionOutput) -> None:
        """Record peak GPU memory for the current forward pass into output.

        Must be called immediately after pipeline.forward(), with
        reset_peak_memory_stats() called just before it, so the measurement
        reflects this request only and not the global historical maximum.

        Uses max_memory_reserved (CUDA memory pool high-water mark) rather than
        max_memory_allocated so that allocator fragmentation is also visible.
        See: https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_reserved.html
        """
        peak_reserved_bytes = current_omni_platform.max_memory_reserved()
        peak_allocated_bytes = current_omni_platform.max_memory_allocated()

        output.peak_memory_mb = peak_reserved_bytes / (1024**2)
        peak_reserved_gb = peak_reserved_bytes / (1024**3)
        peak_allocated_gb = peak_allocated_bytes / (1024**3)
        pool_overhead_gb = peak_reserved_gb - peak_allocated_gb

        logger.debug(
            "Peak GPU memory (this request): %.2f GB reserved, %.2f GB allocated, %.2f GB pool overhead (%.1f%%)",
            peak_reserved_gb,
            peak_allocated_gb,
            pool_overhead_gb,
            pool_overhead_gb / peak_reserved_gb * 100 if peak_reserved_gb > 0 else 0.0,
        )

    def execute_model(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """
        Execute a forward pass for the given requests.

        Args:
            req: A diffusion request containing a list of prompts to process.

        Returns:
            DiffusionOutput with generated results.

        Note:
            We use torch.no_grad() for HSDP because HSDP2's fully_shard requires access
            to tensor version counters in pre_forward hooks, which inference tensors do
            not track. For non-HSDP inference, we use torch.inference_mode() for better
            performance.
        """
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if len(req.prompts) == 0:
            raise ValueError("Cannot execute model with empty request list")

        # Use no_grad() for HSDP compatibility, inference_mode() otherwise for better perf
        use_hsdp = self.od_config.parallel_config.use_hsdp
        grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
        with grad_context:
            # The manager handles the check for need_recv_cache internally
            self.kv_transfer_manager.receive_multi_kv_cache_distributed(
                req,
                cfg_kv_collect_func=getattr(self.od_config, "cfg_kv_collect_func", None),
                target_device=getattr(self.pipeline, "device", None),
            )

            if req.sampling_params.generator is None and req.sampling_params.seed is not None:
                if req.sampling_params.generator_device is not None:
                    gen_device = req.sampling_params.generator_device
                elif self.device.type == "cpu":
                    gen_device = "cpu"
                else:
                    gen_device = self.device
                req.sampling_params.generator = torch.Generator(device=gen_device).manual_seed(req.sampling_params.seed)

            # Refresh cache context if needed
            if (
                not getattr(req, "skip_cache_refresh", False)
                and self.cache_backend is not None
                and self.cache_backend.is_enabled()
            ):
                # FIXME (Alex): When num_inference_steps is None, we defer to
                # pipelines for default, but don't refresh the cache; the right
                # way to do this is to merge the sampling params first.
                #
                # For now, if num_inference_steps is not set, we pass 0 to allow
                # TeaCache to refresh to align with the param signature. This is
                # okay to force refresh TeaCache because the refresh does not use
                # num_inference_steps at all (i.e., just resets state and clears
                # stale residuals).
                num_inference_steps = req.sampling_params.num_inference_steps
                if self.od_config.cache_backend == "tea_cache" and num_inference_steps is None:
                    num_inference_steps = 0

                if num_inference_steps is not None:
                    self.cache_backend.refresh(self.pipeline, num_inference_steps)
                else:
                    logger.warning(
                        "Failed to refresh the diffusion transformer cache; backend %s "
                        "currently requires num_inference_steps to be passed explicitly",
                        self.od_config.cache_backend,
                    )

            is_primary = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            if is_primary:
                current_omni_platform.reset_peak_memory_stats()

            with set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config):
                with record_function("pipeline_forward"):
                    output = self.pipeline.forward(req)

            if is_primary:
                self._record_peak_memory(output)

            # Log prompt-embed cache activity (hits/misses accumulate across requests).
            if is_primary and self.prompt_embed_cache is not None:
                logger.debug("prompt-embed cache: %s", self.prompt_embed_cache.stats())

            # NOTE:
            if (
                self.cache_backend is not None
                and self.cache_backend.is_enabled()
                and self.od_config.cache_backend == "cache_dit"
                and self.od_config.enable_cache_dit_summary
            ):
                cache_summary(self.pipeline, details=True)

            return output

    # ------------------------------------------------------------------
    # Step-wise execution
    # ------------------------------------------------------------------

    def supports_step_mode(self) -> bool:
        """Return whether current pipeline supports step execution."""
        return self.pipeline is not None and supports_diffusion_atoms(self.pipeline)

    def _supports_diffusion_atoms(self) -> bool:
        return self.pipeline is not None and supports_diffusion_atoms(self.pipeline)

    @staticmethod
    def _coerce_transport_payload(payload: DiffusionStateTransport | dict[str, Any]) -> DiffusionStateTransport:
        if isinstance(payload, DiffusionStateTransport):
            return payload
        if isinstance(payload, dict):
            return DiffusionStateTransport(**payload)
        raise TypeError(f"Expected DiffusionStateTransport payload, got {type(payload)!r}.")

    def _pack_conditioning(self, state: DiffusionRequestState) -> Any:
        pack = getattr(self.pipeline, "pack_conditioning", None)
        if callable(pack):
            return pack(state)
        return state.conditioning

    def _unpack_conditioning(
        self,
        payload: DiffusionStateTransport,
        state: DiffusionRequestState,
    ) -> DiffusionRequestState:
        if payload.conditioning is None:
            return state
        # from_transport owns device normalization for payload.conditioning.
        conditioning = state.conditioning
        unpack = getattr(self.pipeline, "unpack_conditioning", None)
        if callable(unpack):
            return unpack(conditioning, state)
        state.conditioning = conditioning
        return state

    def _rehydrate_transport_state(self, state: DiffusionRequestState) -> DiffusionRequestState:
        rehydrate = getattr(self.pipeline, "rehydrate_stage_state", None)
        if callable(rehydrate):
            return rehydrate(state)
        # Remote payloads intentionally avoid serializing scheduler objects.
        # Recreate a per-request scheduler from the local role pipeline.
        if state.scheduler is None and getattr(self.pipeline, "scheduler", None) is not None:
            state.scheduler = copy.deepcopy(self.pipeline.scheduler)
            set_begin_index = getattr(state.scheduler, "set_begin_index", None)
            if callable(set_begin_index):
                set_begin_index(state.step_index)
        return state

    def _state_from_transport(
        self,
        payload: DiffusionStateTransport | dict[str, Any],
        *,
        expected_boundary: TransportBoundary,
    ) -> DiffusionRequestState:
        payload = self._coerce_transport_payload(payload)
        if payload.boundary != expected_boundary:
            raise ValueError(f"Expected {expected_boundary} payload, got {payload.boundary!r}.")
        state = DiffusionRequestState.from_transport(payload, device=self.device)
        state = self._unpack_conditioning(payload, state)
        return self._rehydrate_transport_state(state)

    def _state_to_transport(
        self,
        state: DiffusionRequestState,
        boundary: TransportBoundary,
    ) -> DiffusionStateTransport:
        # Only the encode boundary needs model-private conditioning; DiT->decode
        # remains model-agnostic and transports the final public latents.
        conditioning = self._pack_conditioning(state) if boundary == "encode_to_dit" else None
        return state.to_transport(boundary, conditioning=conditioning)

    def execute_encode(
        self,
        req: OmniDiffusionRequest,
        *,
        boundary: TransportBoundary = "encode_to_dit",
    ) -> DiffusionStateTransport:
        if boundary != "encode_to_dit":
            raise ValueError(f"execute_encode only emits encode_to_dit payloads, got {boundary!r}.")
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if not self.supports_step_mode():
            raise ValueError("Current pipeline does not support stage encode execution.")
        use_hsdp = self.od_config.parallel_config.use_hsdp
        grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
        with grad_context:
            state = self.pipeline.init_state(req)
            state.request_id = req.request_id
            self._ensure_state_generator(state)
            state = self.run_encode_stage(state)
            return self._state_to_transport(state, boundary)

    def execute_decode(self, payload: DiffusionStateTransport | dict[str, Any]) -> DiffusionOutput:
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        use_hsdp = self.od_config.parallel_config.use_hsdp
        grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
        with grad_context:
            state = self._state_from_transport(payload, expected_boundary="dit_to_decode")
            return self.run_decode_stage(state)

    def _update_states(
        self, scheduler_output: DiffusionSchedulerOutput
    ) -> tuple[list[DiffusionRequestState], list[str]]:
        """Step-before update: cleanup finished requests and get/create one running state."""
        for request_id in scheduler_output.finished_req_ids:
            self.state_cache.pop(request_id, None)

        resolved: list[DiffusionRequestState] = []
        new_request_ids: list[str] = []
        created_request_ids: list[str] = []
        try:
            # process new requests
            for sched_new_req in scheduler_output.scheduled_new_reqs:
                request_id = sched_new_req.request_id
                req = sched_new_req.req
                if request_id in self.state_cache:
                    raise ValueError(f"Received duplicate new-request payload for cached request {request_id}.")
                created_request_ids.append(request_id)
                transport_payload = getattr(req, "stage_transport_payload", None)
                if transport_payload is not None:
                    # A denoiser role receives already-encoded conditioning from
                    # an encoder worker and should not run encode atoms again.
                    new_state = self._state_from_transport(
                        transport_payload,
                        expected_boundary="encode_to_dit",
                    )
                    new_state.request_id = request_id
                else:
                    new_request_ids.append(request_id)
                    new_state = self.pipeline.init_state(req)
                    new_state.request_id = request_id
                self.state_cache[request_id] = new_state
                resolved.append(new_state)

            # process cached requests
            for request_id in scheduler_output.scheduled_cached_reqs.request_ids:
                state = self.state_cache.get(request_id)
                if state is None:
                    raise ValueError(f"Missing cached state for request {request_id}.")
                resolved.append(state)
        except Exception:
            for request_id in created_request_ids:
                self.state_cache.pop(request_id, None)
            raise

        return resolved, new_request_ids

    def _ensure_state_generator(self, state: DiffusionRequestState) -> None:
        if state.sampling.generator is not None or state.sampling.seed is None:
            return
        if state.sampling.generator_device is not None:
            gen_device = state.sampling.generator_device
        elif self.device.type == "cpu":
            gen_device = "cpu"
        else:
            gen_device = self.device
        state.sampling.generator = torch.Generator(device=gen_device).manual_seed(state.sampling.seed)

    def run_encode_stage(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Run validation/encoding/preparation atoms."""
        state = self.pipeline.validation(state)
        state = self.pipeline.encoding(state)
        state = self.pipeline.preparation(state)
        return state

    def run_denoise_stage(self, input_batch: InputBatch) -> torch.Tensor | None:
        return self.pipeline.predict_noise(input_batch)

    def run_scheduler_stage(
        self,
        state: DiffusionRequestState,
        noise_pred: torch.Tensor,
    ) -> DiffusionRequestState:
        return self.pipeline.advance_scheduler(state, noise_pred)

    def run_decode_stage(self, state: DiffusionRequestState) -> DiffusionOutput:
        state = self.pipeline.decoding(state)
        return self.pipeline.postprocess(state)

    def _attach_model_inputs(self, input_batch: InputBatch) -> None:
        input_batch.model_inputs.clear()
        build_model_inputs = getattr(self.pipeline, "build_model_inputs", None)
        if callable(build_model_inputs):
            input_batch.model_inputs.update(build_model_inputs(input_batch.states))

    def _prepare_batch_inputs(self, states: list[DiffusionRequestState], new_request_ids: list[str]) -> InputBatch:
        for index, state in enumerate(states):
            if state.request_id in new_request_ids:
                self._ensure_state_generator(state)
                new_state = self.run_encode_stage(state)
                # Treat the returned state as authoritative even when current
                # Qwen atoms mutate in place; other pipelines may return a new object.
                states[index] = new_state
                self.state_cache[new_state.request_id] = new_state

        input_batch = InputBatch.make_batch(
            states,
            cached_batch=getattr(self, "input_batch", None),
        )
        self._attach_model_inputs(input_batch)
        self.input_batch = input_batch
        return input_batch

    def _update_states_after(
        self,
        states: list[DiffusionRequestState],
        input_batch: InputBatch,
        interrupted: bool = False,
    ):
        """Step-after update: clear cached state for completed request."""
        gathered_latents = torch.cat([state.latents for state in states], dim=0)
        if (
            input_batch.latents.size() == gathered_latents.size()
            and input_batch.latents.dtype == gathered_latents.dtype
            and input_batch.latents.device == gathered_latents.device
        ):
            input_batch.latents.copy_(gathered_latents)
        else:
            input_batch.latents = gathered_latents.clone()

        self.input_batch = input_batch
        scatter_latents(states, input_batch)

        for state in states:
            if interrupted or state.denoise_completed:
                self.state_cache.pop(state.request_id, None)

    def _prepare_attn_metadata(self, input_batch: InputBatch) -> Any:
        if self._supports_diffusion_atoms():
            build_attn = getattr(self.pipeline, "build_step_attention_metadata", None)
            if callable(build_attn):
                return build_attn(input_batch)
        model_state = getattr(self, "model_state", None)
        if model_state is None:
            return {}
        prepare_attn = getattr(model_state, "prepare_attn", None)
        if not callable(prepare_attn):
            return {}
        return prepare_attn(input_batch)

    def execute_stepwise(self, scheduler_output: DiffusionSchedulerOutput) -> BatchRunnerOutput:
        """Execute one step for one scheduled request and return runner output."""
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if not self.supports_step_mode():
            raise ValueError("Current pipeline does not support step execution.")
        # Stepwise mode only supports the basic state-driven denoise path for now.
        # Request-mode extras such as cache backends, KV transfer, editing inputs,
        # and similar features are not supported here yet.
        if self.od_config.cache_backend not in (None, "none"):
            raise ValueError("Step mode does not support cache_backend yet.")

        use_hsdp = self.od_config.parallel_config.use_hsdp
        grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
        with grad_context:
            states, new_request_ids = self._update_states(scheduler_output)
            input_batch = self._prepare_batch_inputs(states, new_request_ids)
            attn_metadata = self._prepare_attn_metadata(input_batch)

            with set_forward_context(
                vllm_config=self.vllm_config,
                omni_diffusion_config=self.od_config,
                attn_metadata=attn_metadata,
            ):
                noise_pred = self.run_denoise_stage(input_batch)

                runner_output_list = []
                pipeline_interrupted = getattr(self.pipeline, "interrupt", False)
                if noise_pred is None and pipeline_interrupted:
                    for state in states:
                        runner_output_list.append(
                            RunnerOutput(
                                request_id=state.request_id,
                                step_index=state.step_index,
                                finished=True,
                                result=DiffusionOutput(error="stepwise denoise interrupted"),
                            )
                        )

                else:
                    if noise_pred is None:
                        raise ValueError("Denoise stage returned None without setting pipeline.interrupt.")
                    offset = 0
                    for index, (req, row_num) in enumerate(zip(states, input_batch.row_counts, strict=True)):
                        # Each request may expand to multiple CFG/output rows,
                        # so slice the merged noise tensor using the batch row layout.
                        new_req = self.run_scheduler_stage(
                            req,
                            noise_pred[offset : offset + row_num],
                        )
                        if new_req is not req:
                            states[index] = new_req
                            self.state_cache[new_req.request_id] = new_req
                            req = new_req
                        offset += row_num
                        if req.denoise_completed:
                            # Split deployments hand final latents to a decode
                            # role instead of decoding inside the denoiser.
                            result = DiffusionOutput(
                                custom_output={
                                    "stage_transport": self._state_to_transport(req, "dit_to_decode")
                                }
                            )
                        else:
                            result = None
                        runner_output_list.append(
                            RunnerOutput(
                                request_id=req.request_id,
                                step_index=req.step_index,
                                finished=req.denoise_completed,
                                result=result,
                            )
                        )

                    if noise_pred is not None and offset != noise_pred.shape[0]:
                        raise ValueError(
                            f"Stepwise noise_pred consumed {offset} rows, "
                            f"but batched noise_pred has {noise_pred.shape[0]} rows."
                        )

                self._update_states_after(states, input_batch, pipeline_interrupted)

                return BatchRunnerOutput.from_list(runner_output_list)
