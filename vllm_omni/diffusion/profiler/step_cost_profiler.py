# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.worker.utils import DiffusionRequestState
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

DEFAULT_STEP_PROFILE_OUTPUT_PATH = "/tmp/vllm_omni_step_cost_raw.replica{replica_id}.rank{rank}.jsonl"


@dataclass(frozen=True)
class DiffusionStepProfileConfig:
    enabled: bool = False
    output_path: str = DEFAULT_STEP_PROFILE_OUTPUT_PATH
    sync_device: bool = True
    record_rank: int | str | None = 0


class DiffusionStepCostProfiler:
    """Write one JSONL record for each executed stepwise diffusion bucket."""

    def __init__(
        self,
        config: DiffusionStepProfileConfig,
        *,
        od_config: Any,
        device: Any,
    ) -> None:
        self.config = config
        self.od_config = od_config
        self.device = device
        self.rank = _optional_int(os.environ.get("RANK"))
        if self.rank is None:
            self.rank = 0
        self.local_rank = _optional_int(os.environ.get("LOCAL_RANK"))
        self.replica_id = _profile_replica_id(od_config)
        self._write_lock = Lock()
        self._enabled_for_rank = self._should_record_rank()
        self.output_path = self._resolve_output_path(config.output_path)
        if self.enabled:
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            logger.info("[DiffusionStepCostProfiler] Writing step profile records to %s", self.output_path)

    @classmethod
    def from_od_config(cls, od_config: Any, *, device: Any) -> DiffusionStepCostProfiler | None:
        config = _get_step_profile_config(od_config)
        if not config.enabled:
            return None
        return cls(config, od_config=od_config, device=device)

    @property
    def enabled(self) -> bool:
        return self.config.enabled and self._enabled_for_rank

    def sync(self) -> None:
        if not self.enabled or not self.config.sync_device:
            return
        if current_omni_platform.is_available():
            current_omni_platform.synchronize()

    def timer_start(self) -> float:
        self.sync()
        return time.perf_counter()

    def timer_end_ms(self, start_s: float) -> float:
        self.sync()
        return (time.perf_counter() - start_s) * 1000.0

    def write_step_record(
        self,
        *,
        scheduler_output: Any,
        states: list[DiffusionRequestState],
        timings_ms: dict[str, float],
        interrupted: bool,
        step_indices: list[int] | None = None,
    ) -> None:
        if not self.enabled or not states:
            return

        now_s = time.time()
        sampling_params = [_state_sampling(state) for state in states]
        step_indices_before = (
            step_indices if step_indices is not None else [_state_step_index(state) for state in states]
        )
        post_step_indices = [_state_step_index(state) for state in states]
        total_steps = [_state_total_steps(state) for state in states]
        arrival_time_s = [_state_arrival_time_s(state, now_s) for state in states]
        deadline_time_s = [_state_deadline_time_s(state) for state in states]
        reference_cost_ms = [_state_reference_cost_ms(state) for state in states]
        slo_ms = [_state_slo_ms(state) for state in states]
        record: dict[str, Any] = {
            "timestamp_s": now_s,
            "model": getattr(self.od_config, "model", None),
            "stage_id": getattr(self.od_config, "stage_id", None),
            "replica_id": self.replica_id,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "device": str(self.device),
            "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tp_size": getattr(getattr(self.od_config, "parallel_config", None), "tensor_parallel_size", None),
            "max_num_seqs": getattr(self.od_config, "max_num_seqs", None),
            "global_scheduler_step_id": getattr(scheduler_output, "step_id", None),
            "scheduler_debug": getattr(scheduler_output, "debug_info", None),
            "num_running_reqs": getattr(scheduler_output, "num_running_reqs", None),
            "num_waiting_reqs": getattr(scheduler_output, "num_waiting_reqs", None),
            "request_ids": [state.req_id for state in states],
            "scheduled_req_ids": list(getattr(scheduler_output, "scheduled_req_ids", [])),
            "batch_size": len(states),
            "effective_batch_size": _effective_batch_size(states),
            "step_indices": step_indices_before,
            "post_step_indices": post_step_indices,
            "remaining_steps": [max(total - step, 0) for total, step in zip(total_steps, post_step_indices)],
            "total_steps": total_steps,
            "arrival_time_s": arrival_time_s,
            "deadline_time_s": deadline_time_s,
            "reference_cost_ms": reference_cost_ms,
            "slo_ms": slo_ms,
            "age_ms": [(now_s - arrival) * 1000.0 for arrival in arrival_time_s],
            "time_to_deadline_ms": [
                (deadline - now_s) * 1000.0 if deadline is not None else None for deadline in deadline_time_s
            ],
            "shape_key": _shape_key(sampling_params[0]),
            "shapes": [_shape_dict(params) for params in sampling_params],
            "profile_tags": _profile_tags(states),
            "interrupted": interrupted,
            **timings_ms,
        }
        self._write_jsonl(record)

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._write_lock:
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")

    def _should_record_rank(self) -> bool:
        record_rank = self.config.record_rank
        if record_rank is None or record_rank == "all":
            return True
        try:
            return self.rank == int(record_rank)
        except (TypeError, ValueError):
            logger.warning(
                "[DiffusionStepCostProfiler] Invalid record_rank=%r; recording on all ranks.",
                record_rank,
            )
            return True

    def _resolve_output_path(self, output_path: str) -> str:
        values = {
            "rank": self.rank if self.rank is not None else "unknown",
            "local_rank": self.local_rank if self.local_rank is not None else "unknown",
            "replica_id": self.replica_id if self.replica_id is not None else "unknown",
        }
        try:
            resolved = output_path.format(**values)
        except Exception:
            resolved = output_path

        needs_replica_suffix = (
            "{replica_id" not in output_path
            and self.replica_id is not None
            and (self.config.record_rank in (None, "all") or self.replica_id != 0)
        )
        needs_rank_suffix = self.config.record_rank in (None, "all") and "{rank" not in output_path
        if needs_replica_suffix or needs_rank_suffix:
            path = Path(resolved)
            suffix_parts = []
            if needs_replica_suffix:
                suffix_parts.append(f"replica{values['replica_id']}")
            if needs_rank_suffix:
                suffix_parts.append(f"rank{values['rank']}")
            return str(path.with_name(f"{path.stem}.{'.'.join(suffix_parts)}{path.suffix}"))
        return resolved


def _get_step_profile_config(od_config: Any) -> DiffusionStepProfileConfig:
    additional_config = getattr(od_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        return DiffusionStepProfileConfig()
    raw = additional_config.get("diffusion_step_profile") or {}
    if not isinstance(raw, dict):
        return DiffusionStepProfileConfig(enabled=bool(raw))
    return DiffusionStepProfileConfig(
        enabled=bool(raw.get("enabled", False)),
        output_path=str(raw.get("output_path") or DEFAULT_STEP_PROFILE_OUTPUT_PATH),
        sync_device=bool(raw.get("sync_device", True)),
        record_rank=raw.get("record_rank", 0),
    )


def _shape_dict(sampling: Any) -> dict[str, Any]:
    height = getattr(sampling, "height", None)
    width = getattr(sampling, "width", None)
    resolution = getattr(sampling, "resolution", None)
    if height is None:
        height = resolution
    if width is None:
        width = resolution
    return {
        "height": height,
        "width": width,
        "num_frames": getattr(sampling, "num_frames", 1),
        "resolution": resolution,
    }


def _shape_key(sampling: Any) -> str:
    shape = _shape_dict(sampling)
    return f"{shape['width']}x{shape['height']}x{shape['num_frames']}"


def _effective_batch_size(states: list[DiffusionRequestState]) -> float:
    return sum(_request_effective_size(state) for state in states)


def _request_effective_size(state: DiffusionRequestState) -> float:
    sampling = _state_sampling(state)
    prompts = getattr(state, "prompts", None)
    if prompts is None:
        prompts = getattr(getattr(state, "req", None), "prompts", None)
    prompt_count = len(prompts) if prompts else 1
    outputs = _optional_float(getattr(sampling, "num_outputs_per_prompt", None)) or 1.0
    guidance = 2.0 if bool(getattr(state, "do_true_cfg", False)) else 1.0
    return max(prompt_count, 1) * max(outputs, 1.0) * guidance


def _profile_tags(states: list[DiffusionRequestState]) -> dict[str, Any]:
    tags: dict[str, Any] = {}
    keys = (
        "profile_combo_id",
        "profile_phase",
        "profile_repeat",
        "profile_batch_size",
        "profile_shape",
        "profile_mode",
        "profile_policy",
        "profile_scale",
        "profile_trace_id",
    )
    for key in keys:
        values = []
        for state in states:
            extra_args = getattr(_state_sampling(state), "extra_args", None)
            if isinstance(extra_args, dict) and key in extra_args:
                values.append(extra_args[key])
        if not values:
            continue
        tags[key] = values[0] if all(value == values[0] for value in values) else values
    return tags


def _profile_replica_id(od_config: Any) -> int | None:
    replica_id = _optional_int(getattr(od_config, "omni_replica_id", None))
    if replica_id is not None:
        return replica_id
    additional_config = getattr(od_config, "additional_config", None)
    if isinstance(additional_config, dict):
        return _optional_int(additional_config.get("_omni_replica_id"))
    return None


def _state_sampling(state: DiffusionRequestState) -> Any:
    sampling = getattr(state, "sampling", None)
    if sampling is not None:
        return sampling
    req = getattr(state, "req", None)
    return getattr(req, "sampling_params", None)


def _state_step_index(state: DiffusionRequestState) -> int:
    value = _optional_int(getattr(state, "step_index", None))
    if value is not None:
        return value
    return _optional_int(getattr(_state_sampling(state), "step_index", None)) or 0


def _state_total_steps(state: DiffusionRequestState) -> int:
    value = _optional_int(getattr(state, "total_steps", None))
    if value is not None:
        return max(value, 1)
    value = _optional_int(getattr(_state_sampling(state), "num_inference_steps", None))
    return max(value or 1, 1)


def _state_arrival_time_s(state: DiffusionRequestState, default: float) -> float:
    value = _optional_float(getattr(state, "arrival_time_s", None))
    if value is not None:
        return value
    value = _optional_float(getattr(_state_sampling(state), "arrival_time_s", None))
    return default if value is None else value


def _state_deadline_time_s(state: DiffusionRequestState) -> float | None:
    value = _optional_float(getattr(state, "deadline_time_s", None))
    if value is not None:
        return value
    return _optional_float(getattr(_state_sampling(state), "deadline_time_s", None))


def _state_reference_cost_ms(state: DiffusionRequestState) -> float | None:
    value = _optional_float(getattr(state, "reference_cost_ms", None))
    if value is not None:
        return value
    return _optional_float(getattr(_state_sampling(state), "reference_cost_ms", None))


def _state_slo_ms(state: DiffusionRequestState) -> float | None:
    value = _optional_float(getattr(state, "slo_ms", None))
    if value is not None:
        return value
    return _optional_float(getattr(_state_sampling(state), "slo_ms", None))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "DEFAULT_STEP_PROFILE_OUTPUT_PATH",
    "DiffusionStepCostProfiler",
    "DiffusionStepProfileConfig",
]
