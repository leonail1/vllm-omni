# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RunnerV2 for stage-composed diffusion pipelines."""

from __future__ import annotations

import json
import os
import time

import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.interface import supports_step_execution
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import (
    BatchRunnerOutput,
    DiffusionRequestState,
    DiffusionStateTransport,
    RunnerOutput,
    TransportBoundary,
)


class DiffusionModelRunnerV2(DiffusionModelRunner):
    """Stage-aware runner.

    The model owns pure compute stages.  This runner owns execution mode,
    transport boundaries, role routing, and all serialization handoff points.
    """

    def _assert_stage_pipeline(self) -> None:
        if self.pipeline is None or not supports_step_execution(self.pipeline):
            raise ValueError(
                "stage execution requires a pipeline implementing "
                "encode_stage(), denoise_stage(), scheduler_stage(), and decode_stage()."
            )

    @staticmethod
    def _coerce_transport_payload(
        payload: DiffusionStateTransport | dict,
    ) -> DiffusionStateTransport:
        if isinstance(payload, DiffusionStateTransport):
            return payload
        if isinstance(payload, dict):
            return DiffusionStateTransport(**payload)
        raise TypeError(f"Expected DiffusionStateTransport payload, got {type(payload)!r}.")

    def load_model(self, *args, **kwargs) -> None:
        # Validate the stage contract immediately after role-specific modules load.
        super().load_model(*args, **kwargs)
        if self.pipeline is not None:
            self._assert_stage_pipeline()

    def _rehydrate_state(self, state: DiffusionRequestState) -> DiffusionRequestState:
        # Restore model-owned runtime objects after a transport boundary.
        rehydrate = getattr(self.pipeline, "rehydrate_stage_state", None)
        if callable(rehydrate):
            # Models rebuild non-serializable runtime objects after a role boundary.
            return rehydrate(state)
        return state

    def execute_encode(
        self,
        req: OmniDiffusionRequest,
        *,
        boundary: TransportBoundary = "encode_to_dit",
    ) -> DiffusionStateTransport:
        """Run the model encode stage and return an encode->DiT payload."""
        if boundary != "encode_to_dit":
            raise ValueError(f"execute_encode only emits encode_to_dit payloads, got {boundary!r}.")
        state = self.pipeline._state_from_request(req)
        self.pipeline.encode_stage(state)
        return state.to_transport(boundary)

    def execute_denoise(self, input_batch: InputBatch) -> torch.Tensor | None:
        """Run one batched DiT denoise stage inside the forward context."""
        attn_metadata = self._prepare_attn_metadata(input_batch)
        with set_forward_context(
            vllm_config=self.vllm_config,
            omni_diffusion_config=self.od_config,
            attn_metadata=attn_metadata,
        ):
            return self.pipeline.denoise_stage(input_batch)

    def execute_scheduler(
        self,
        state: DiffusionRequestState,
        noise: torch.Tensor | None,
    ) -> None:
        """Run one scheduler stage for one request state."""
        self.pipeline.scheduler_stage(state, noise)

    def execute_decode(self, payload: DiffusionStateTransport | dict) -> DiffusionOutput:
        """Decode a DiT->decode payload into a final DiffusionOutput."""
        payload = self._coerce_transport_payload(payload)
        if payload.boundary != "dit_to_decode":
            raise ValueError(f"execute_decode requires dit_to_decode payload, got {payload.boundary!r}.")
        state = DiffusionRequestState.from_transport(payload, device=self.device)
        state = self._rehydrate_state(state)
        return self.pipeline.decode_stage(state)

    def _update_transport_states(
        self,
        scheduler_output: DiffusionSchedulerOutput,
    ) -> tuple[list[DiffusionRequestState], list[str]]:
        # Translate StepScheduler output into cached DiT request states.
        for request_id in scheduler_output.finished_req_ids:
            self.state_cache.pop(request_id, None)

        resolved: list[DiffusionRequestState] = []
        new_request_ids: list[str] = []
        try:
            for sched_new_req in scheduler_output.scheduled_new_reqs:
                request_id = sched_new_req.request_id
                new_request_ids.append(request_id)
                if request_id in self.state_cache:
                    raise ValueError(f"Received duplicate DiT transport request {request_id}.")
                payload = getattr(sched_new_req.req, "stage_transport_payload", None)
                if payload is None:
                    raise ValueError("DiT stage request is missing encode_to_dit stage_transport payload.")
                payload = self._coerce_transport_payload(payload)
                if payload.boundary != "encode_to_dit":
                    raise ValueError(f"DiT stage requires encode_to_dit payload, got {payload.boundary!r}.")
                # DiT role starts from encode output instead of running encode atoms locally.
                state = DiffusionRequestState.from_transport(payload, device=self.device)
                state.request_id = request_id
                state = self._rehydrate_state(state)
                self.state_cache[request_id] = state
                resolved.append(state)

            for request_id in scheduler_output.scheduled_cached_reqs.request_ids:
                state = self.state_cache.get(request_id)
                if state is None:
                    raise ValueError(f"Missing cached DiT transport state for request {request_id}.")
                resolved.append(state)
        except Exception:
            for request_id in new_request_ids:
                self.state_cache.pop(request_id, None)
            raise

        return resolved, new_request_ids

    def _prepare_transport_batch_inputs(self, states: list[DiffusionRequestState]) -> InputBatch:
        # Rebuild the current scheduled batch while reusing cached InputBatch storage.
        input_batch = InputBatch.make_batch(
            states,
            cached_batch=getattr(self, "input_batch", None),
        )
        self.input_batch = input_batch
        return input_batch

    def _trace_dit_batch(self, states: list[DiffusionRequestState], input_batch: InputBatch) -> None:
        # Optional JSONL tracing is diagnostic-only and must not affect generation.
        trace_path = os.environ.get("VLLM_OMNI_DIFFUSION_BATCH_TRACE")
        if not trace_path:
            return
        try:
            payload = {
                "time": time.time(),
                "stage_role": getattr(self.od_config, "stage_role", "all"),
                "num_states": len(states),
                "request_ids": [state.request_id for state in states],
                "step_indices": [state.step_index for state in states],
                "latent_rows": int(input_batch.latents.shape[0]) if input_batch.latents is not None else None,
            }
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            # Tracing must never affect generation.
            pass

    def _run_one_batched_dit_step(
        self,
        states: list[DiffusionRequestState],
        input_batch: InputBatch,
    ) -> tuple[bool, bool]:
        """Run one batched DiT step and advance scheduler state per request."""
        noise_pred = self.execute_denoise(input_batch)

        pipeline_interrupted = bool(getattr(self.pipeline, "interrupt", False))
        if noise_pred is None and pipeline_interrupted:
            return pipeline_interrupted, True

        offset = 0
        for state in states:
            row_num = state.latents.shape[0]
            self.execute_scheduler(
                state,
                noise_pred[offset : offset + row_num] if noise_pred is not None else None,
            )
            offset += row_num

        if noise_pred is not None and offset != noise_pred.shape[0]:
            raise ValueError(
                f"DiT step consumed {offset} rows, "
                f"but batched noise_pred has {noise_pred.shape[0]} rows."
            )
        return pipeline_interrupted, False

    def execute_stepwise(self, scheduler_output: DiffusionSchedulerOutput) -> BatchRunnerOutput:
        """Run one DiT step for the stage-split DiT role."""
        # StepScheduler calls this once per DiT tick; completion is reported through RunnerOutput.
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if self.od_config.cache_backend not in (None, "none"):
            raise ValueError("Step mode does not support cache_backend yet.")

        use_hsdp = self.od_config.parallel_config.use_hsdp
        grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
        with grad_context:
            states, _ = self._update_transport_states(scheduler_output)
            input_batch = self._prepare_transport_batch_inputs(states)
            self._trace_dit_batch(states, input_batch)
            pipeline_interrupted, step_interrupted = self._run_one_batched_dit_step(states, input_batch)

            runner_output_list: list[RunnerOutput] = []
            if step_interrupted:
                runner_output_list = [
                    RunnerOutput(
                        request_id=state.request_id,
                        step_index=state.step_index,
                        finished=True,
                        result=DiffusionOutput(error="DiT step interrupted"),
                    )
                    for state in states
                ]
            else:
                for state in states:
                    finished = state.denoise_completed
                    result = (
                        DiffusionOutput(
                            custom_output={
                                "stage_transport": state.to_transport("dit_to_decode")
                            },
                            finished=True,
                        )
                        if finished
                        else None
                    )
                    runner_output_list.append(
                        RunnerOutput(
                            request_id=state.request_id,
                            step_index=state.step_index,
                            finished=finished,
                            result=result,
                        )
                    )

            self._update_states_after(states, input_batch, pipeline_interrupted)
            return BatchRunnerOutput.from_list(runner_output_list)
