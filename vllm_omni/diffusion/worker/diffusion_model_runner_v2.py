# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stepwise diffusion runner with explicit encode / denoise / decode stages."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch.profiler import record_function
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.interface import (
    supports_stage_atoms,
    supports_step_atoms,
    supports_step_payload_codec,
)
from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.input_batch import InputBatch, scatter_latents
from vllm_omni.diffusion.worker.utils import (
    BatchRunnerOutput,
    DiffusionRequestState,
    RunnerOutput,
    attach_stage_durations,
    clear_pipeline_stage_durations,
    consume_pipeline_stage_durations,
    merge_stage_durations,
)
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_ENCODE_TO_DIT = "encode_to_dit"
_DIT_TO_DECODE = "dit_to_decode"


class DiffusionModelRunnerV2(DiffusionModelRunner):
    """Runner v2 that exposes stepwise stages as runner methods.

    The class intentionally inherits v1 loading, state-cache, and batching
    helpers. Its stepwise entrypoint is owned here and dispatches through the
    explicit encode, DiT denoise, and decode stage methods.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._boundary_put_chunks: dict[tuple[str, str], int] = {}
        self._boundary_get_chunks: dict[tuple[str, str], int] = {}
        self._local_stage_payload_cache: dict[str, dict[str, Any]] = {}
        model_config = getattr(self.vllm_config, "model_config", None)
        if getattr(model_config, "stage_connector_config", None) is None:
            raise RuntimeError(
                "DiffusionModelRunnerV2 requires stage_connector_config because "
                "stage-boundary payloads are transferred through OmniConnector."
            )
        self.init_omni_connectors(
            vllm_config=self.vllm_config,
            model_config=model_config,
            kv_transfer_manager=self.kv_transfer_manager,
        )
        if self._omni_connector is None:
            raise RuntimeError("stage_connector_config did not create an OmniConnector.")

    def supports_step_mode(self) -> bool:
        return (
            super().supports_step_mode()
            and supports_stage_atoms(self.pipeline)
            and supports_step_atoms(self.pipeline)
            and supports_step_payload_codec(self.pipeline)
        )

    def recv_stage_payload(self, payload_key: str) -> dict[str, Any]:
        payload = self.pop_local_stage_payload(payload_key)
        if payload is None:
            raise KeyError(f"Missing stage payload for key {payload_key}.")
        return dict(payload)

    def _put_boundary_payload(self, request_id: str, boundary: str, payload: dict[str, Any]) -> None:
        if payload is None:
            raise ValueError(f"Cannot send empty diffusion boundary payload for {request_id}:{boundary}.")
        counter_key = (request_id, boundary)
        chunk_id = self._boundary_put_chunks.get(counter_key, 0)
        # Temporary stage-boundary split: include boundary in the connector key
        # to distinguish encode->DiT from DiT->decode. Use an SHM-safe
        # separator because SharedMemoryConnector treats the key as a POSIX SHM
        # name. Real split should use distinct connector stage ids instead.
        connector_key = f"{request_id}__{boundary}__{chunk_id}"
        stage_id = str(getattr(self, "_stage_id", getattr(self.od_config, "stage_id", 0)))
        if self.is_data_transfer_rank():
            success, _size, _metadata = self._omni_connector.put(
                from_stage=stage_id,
                to_stage=stage_id,
                put_key=connector_key,
                data=payload,
            )
            if not success:
                raise RuntimeError(f"Failed to put diffusion boundary payload {connector_key}.")
        self._boundary_put_chunks[counter_key] = chunk_id + 1

    def _recv_boundary_payload(self, request_id: str, boundary: str) -> None:
        counter_key = (request_id, boundary)
        chunk_id = self._boundary_get_chunks.get(counter_key, 0)
        # Temporary stage-boundary split: include boundary in the connector key
        # to distinguish encode->DiT from DiT->decode. Use an SHM-safe
        # separator because SharedMemoryConnector treats the key as a POSIX SHM
        # name. Real split should use distinct connector stage ids instead.
        # Local cache key: "{request_id}:{boundary}".
        connector_key = f"{request_id}__{boundary}__{chunk_id}"
        stage_id = str(getattr(self, "_stage_id", getattr(self.od_config, "stage_id", 0)))
        result = self._recv_ordinary_stage_result(
            self._omni_connector,
            stage_id,
            stage_id,
            connector_key,
        )
        result = self._broadcast_tp_payload_packet(result)
        if result is None:
            raise KeyError(f"Missing diffusion boundary payload {connector_key}.")
        payload, _size = result
        if payload is None:
            raise ValueError(f"Received empty diffusion boundary payload {connector_key}.")

        self.put_local_stage_payload(f"{request_id}:{boundary}", payload)
        self._boundary_get_chunks[counter_key] = chunk_id + 1

    def _cleanup_boundary_transfer_state(self, request_id: str) -> None:
        for key in list(self._boundary_put_chunks):
            if key[0] == request_id:
                self._boundary_put_chunks.pop(key, None)
        for key in list(self._boundary_get_chunks):
            if key[0] == request_id:
                self._boundary_get_chunks.pop(key, None)
        for boundary in (_ENCODE_TO_DIT, _DIT_TO_DECODE):
            self._local_stage_payload_cache.pop(f"{request_id}:{boundary}", None)

    def run_encode_stage(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Run encoder-side setup and publish its payload for the DiT stage."""
        if not supports_step_atoms(self.pipeline):
            raise ValueError(
                f"{type(self.pipeline).__name__} must implement step atoms "
                "for DiffusionModelRunnerV2."
            )
        if not supports_stage_atoms(self.pipeline):
            raise ValueError(
                f"{type(self.pipeline).__name__} must implement stage atoms "
                "for DiffusionModelRunnerV2."
            )
        if not supports_step_payload_codec(self.pipeline):
            raise ValueError(
                f"{type(self.pipeline).__name__} must implement step payload codec "
                "for DiffusionModelRunnerV2."
        )

        clear_pipeline_stage_durations(self.pipeline)
        self.pipeline.init_state(state)
        self.pipeline.check_inputs(state)
        self.pipeline.encode(state)
        self.pipeline.prepare(state)
        merge_stage_durations(
            state,
            consume_pipeline_stage_durations(self.pipeline),
        )

        payload = self.pipeline.pack_encode_payload(state)
        self._put_boundary_payload(state.request_id, _ENCODE_TO_DIT, payload)
        return state

    def _update_states(
        self, scheduler_output: DiffusionSchedulerOutput
    ) -> tuple[list[DiffusionRequestState], list[str]]:
        """Step-before update: cleanup finished requests and get/create one running state."""
        for request_id in scheduler_output.finished_req_ids:
            self.state_cache.pop(request_id, None)
            self._cleanup_boundary_transfer_state(request_id)

        resolved: list[DiffusionRequestState] = []
        new_request_ids: list[str] = []
        try:
            for sched_new_req in scheduler_output.scheduled_new_reqs:
                request_id = sched_new_req.request_id
                new_request_ids.append(request_id)
                if request_id in self.state_cache:
                    raise ValueError(f"Received duplicate new-request payload for cached request {request_id}.")
                new_state = DiffusionRequestState(
                    request_id=request_id,
                    sampling=copy.deepcopy(sched_new_req.req.sampling_params),
                    prompt=sched_new_req.req.prompt,
                    kv_sender_info=sched_new_req.req.kv_sender_info,
                )
                state_req = copy.copy(sched_new_req.req)
                state_req.sampling_params = new_state.sampling
                self.kv_transfer_manager.receive_multi_kv_cache_distributed(
                    state_req,
                    cfg_kv_collect_func=getattr(self.od_config, "cfg_kv_collect_func", None),
                    target_device=self.target_device,
                )
                self.state_cache[request_id] = new_state
                resolved.append(new_state)

            for request_id in scheduler_output.scheduled_cached_reqs.request_ids:
                state = self.state_cache.get(request_id)
                if state is None:
                    raise ValueError(f"Missing cached state for request {request_id}.")
                resolved.append(state)
        except Exception:
            for request_id in new_request_ids:
                self.state_cache.pop(request_id, None)
            raise

        return resolved, new_request_ids

    def _prepare_batch_inputs(self, states: list[DiffusionRequestState], new_request_ids: list[str]) -> InputBatch:
        for state in states:
            if state.request_id not in new_request_ids:
                continue
            if state.sampling.generator is None and state.sampling.seed is not None:
                if state.sampling.generator_device is not None:
                    gen_device = state.sampling.generator_device
                elif self.device.type == "cpu":
                    gen_device = "cpu"
                else:
                    gen_device = self.device
                state.sampling.generator = torch.Generator(device=gen_device).manual_seed(state.sampling.seed)
            self.run_encode_stage(state)
            self._recv_boundary_payload(state.request_id, _ENCODE_TO_DIT)
            self.pipeline.unpack_encode_payload(
                self.recv_stage_payload(f"{state.request_id}:{_ENCODE_TO_DIT}"),
                state,
            )

        input_batch = InputBatch.make_batch(
            states,
            cached_batch=getattr(self, "input_batch", None),
        )
        input_batch.model_inputs = self.pipeline.build_model_inputs(states)
        self.input_batch = input_batch
        return input_batch

    def _update_states_after(
        self,
        states: list[DiffusionRequestState],
        input_batch: InputBatch,
        interrupted: bool = False,
    ) -> None:
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
            if interrupted or state.request_denoise_completed:
                self.state_cache.pop(state.request_id, None)
                self._cleanup_boundary_transfer_state(state.request_id)

    def _prepare_attn_metadata(self, input_batch: InputBatch) -> Any:
        return self.pipeline.build_step_attention_metadata(input_batch)

    def run_denoise_stage(
        self,
        input_batch: InputBatch,
        states: list[DiffusionRequestState],
    ) -> tuple[dict[str, RunnerOutput], bool]:
        """Run one DiT denoise step and request-local scheduler updates."""
        clear_pipeline_stage_durations(self.pipeline)
        with record_function("pipeline_denoise_stage"):
            noise_pred = self.pipeline.denoise_step(input_batch, states=states)

        stage_results: dict[str, DiffusionOutput] = {}
        if noise_pred is None and getattr(self.pipeline, "interrupt", False):
            for state in states:
                stage_results[state.request_id] = DiffusionOutput(error="stepwise denoise interrupted")
            pipeline_interrupted = True
        elif noise_pred is None:
            raise RuntimeError("denoise_step returned None without pipeline interrupt.")
        else:
            pipeline_interrupted = False
            offset = 0
            for state in states:
                row_num = state.latents.shape[0]
                try:
                    step_noise_pred = noise_pred[offset : offset + row_num]
                    self.pipeline.step_scheduler(state, step_noise_pred)
                except Exception as per_req_exc:
                    state.step_index = max(state.step_index, state.total_steps)
                    state.chunk_index = state.total_chunks
                    logger.error(
                        "Stepwise denoise/scheduler error for %s: %s",
                        state.request_id,
                        per_req_exc,
                        exc_info=True,
                    )
                    stage_results[state.request_id] = DiffusionOutput(error=str(per_req_exc))
                finally:
                    offset += row_num

            if noise_pred is not None and offset != noise_pred.shape[0]:
                raise ValueError(
                    f"Stepwise noise_pred consumed {offset} rows, "
                    f"but batched noise_pred has {noise_pred.shape[0]} rows."
                )

        denoise_stage_durations = consume_pipeline_stage_durations(self.pipeline)
        for state in states:
            merge_stage_durations(state, denoise_stage_durations)

        states_by_id = {state.request_id: state for state in states}
        stage_outputs: dict[str, RunnerOutput] = {}
        for request_id, result in stage_results.items():
            state = states_by_id.get(request_id)
            if state is not None:
                attach_stage_durations(state, result)
            stage_outputs[request_id] = RunnerOutput(
                request_id=request_id,
                step_index=state.step_index if state is not None else 0,
                finished=True,
                result=result,
            )
        for state in states:
            if state.request_id in stage_results:
                continue
            if self.od_config.streaming_output:
                should_decode = state.chunk_denoise_completed
            else:
                should_decode = state.denoise_completed
            if not should_decode:
                continue
            payload = self.pipeline.pack_decode_payload(state)
            self._put_boundary_payload(state.request_id, _DIT_TO_DECODE, payload)
        return stage_outputs, pipeline_interrupted

    def run_decode_stage(self, state: DiffusionRequestState) -> DiffusionOutput:
        """Receive/unpack denoise output, then decode final/chunk latents."""
        if not supports_step_payload_codec(self.pipeline):
            raise ValueError(
                f"{type(self.pipeline).__name__} must implement step payload codec "
                "to consume decode-stage payloads."
        )
        self._recv_boundary_payload(state.request_id, _DIT_TO_DECODE)
        self.pipeline.unpack_decode_payload(self.recv_stage_payload(f"{state.request_id}:{_DIT_TO_DECODE}"), state)

        clear_pipeline_stage_durations(self.pipeline)
        with record_function("pipeline_decode_stage"):
            self.pipeline.decode(state)
            result = self.pipeline.postprocess(state)
        merge_stage_durations(
            state,
            consume_pipeline_stage_durations(self.pipeline),
        )
        attach_stage_durations(state, result)

        return result

    def execute_stepwise(self, scheduler_output: DiffusionSchedulerOutput) -> BatchRunnerOutput:
        """Execute one step via explicit encode, DiT denoise, and decode stages."""
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if not self.supports_step_mode():
            raise ValueError("Current pipeline does not support step execution.")
        if self.od_config.cache_backend not in (None, "none"):
            raise ValueError("Step mode does not support cache_backend yet.")

        use_hsdp = self.od_config.parallel_config.use_hsdp
        grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
        with grad_context:
            had_active_states = bool(self.state_cache)
            states, new_request_ids = self._update_states(scheduler_output)
            is_primary = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            if new_request_ids and not had_active_states and is_primary and current_omni_platform.is_available():
                current_omni_platform.reset_peak_memory_stats()
            input_batch = self._prepare_batch_inputs(states, new_request_ids)
            attn_metadata = self._prepare_attn_metadata(input_batch)

            with set_forward_context(
                vllm_config=self.vllm_config,
                omni_diffusion_config=self.od_config,
                attn_metadata=attn_metadata,
            ):
                denoise_outputs, pipeline_interrupted = self.run_denoise_stage(input_batch, states)
                runner_output_list: list[RunnerOutput] = []
                for state in states:
                    denoise_output = denoise_outputs.get(state.request_id)
                    if denoise_output is not None:
                        runner_output_list.append(denoise_output)
                        continue

                    if self.od_config.streaming_output:
                        should_decode = state.chunk_denoise_completed
                    else:
                        should_decode = state.denoise_completed

                    try:
                        if should_decode:
                            result = self.run_decode_stage(state)
                        else:
                            result = None

                        finished = (
                            state.request_denoise_completed
                            if self.od_config.streaming_output
                            else state.denoise_completed
                        )
                        runner_output_list.append(
                            RunnerOutput(
                                request_id=state.request_id,
                                step_index=state.step_index,
                                finished=finished,
                                result=result,
                            )
                        )
                    except Exception as per_req_exc:
                        state.step_index = max(state.step_index, state.total_steps)
                        state.chunk_index = state.total_chunks
                        logger.error(
                            "Stepwise decode error for %s: %s",
                            state.request_id,
                            per_req_exc,
                            exc_info=True,
                        )
                        runner_output_list.append(
                            RunnerOutput(
                                request_id=state.request_id,
                                step_index=state.step_index,
                                finished=True,
                                result=DiffusionOutput(error=str(per_req_exc)),
                            )
                        )

                if is_primary:
                    batch_peak_memory_mb = self._sample_peak_memory_mb()
                    states_by_id = {state.request_id: state for state in states}
                    for state in states:
                        state.peak_memory_mb = max(state.peak_memory_mb, batch_peak_memory_mb)
                    for runner_output in runner_output_list:
                        if runner_output.result is None:
                            continue
                        state = states_by_id.get(runner_output.request_id)
                        if state is None:
                            continue
                        runner_output.result.peak_memory_mb = max(
                            runner_output.result.peak_memory_mb,
                            state.peak_memory_mb,
                        )

                self._update_states_after(states, input_batch, pipeline_interrupted)
                return BatchRunnerOutput.from_list(runner_output_list)
