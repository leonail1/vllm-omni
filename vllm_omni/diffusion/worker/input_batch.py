# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-local diffusion batch structures.

``InputBatch`` is intentionally model-agnostic. It exposes only the fields the
runner must understand to drive continuous batching: request identity, row
layout, latents, timesteps, and generic CFG branch metadata. Pipeline-private
tensors such as prompt embeddings, image latents, and shape metadata are packed
by ``pipeline.build_model_inputs(...)`` into ``model_inputs``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from vllm_omni.diffusion.worker.utils import DiffusionRequestState

RUNNER_STEP_EXTRA_KEY = "_runner_step"


def set_runner_step_config(
    state: DiffusionRequestState,
    *,
    do_true_cfg: bool = False,
    true_cfg_scale: float = 4.0,
    cfg_normalize: bool = False,
) -> None:
    """Store runner-visible CFG metadata without adding model fields to state."""

    state.extra[RUNNER_STEP_EXTRA_KEY] = {
        "do_true_cfg": bool(do_true_cfg),
        "true_cfg_scale": float(true_cfg_scale),
        "cfg_normalize": bool(cfg_normalize),
    }


def get_runner_step_config(state: DiffusionRequestState) -> tuple[bool, float, bool]:
    cfg = state.extra.get(RUNNER_STEP_EXTRA_KEY, {})
    sampling_true_cfg = getattr(state.sampling, "true_cfg_scale", None)
    return (
        bool(cfg.get("do_true_cfg", False)),
        float(cfg.get("true_cfg_scale", sampling_true_cfg if sampling_true_cfg is not None else 4.0)),
        bool(cfg.get("cfg_normalize", getattr(state.sampling, "cfg_normalize", False))),
    )


def _select_states(
    states: Sequence[DiffusionRequestState],
    idx_mapping: torch.Tensor | None,
) -> tuple[list[DiffusionRequestState], torch.Tensor, np.ndarray]:
    if not states:
        raise ValueError("Cannot build InputBatch from empty states.")

    if idx_mapping is None:
        device = states[0].latents.device if states[0].latents is not None else None
        idx_mapping = torch.arange(len(states), dtype=torch.int32, device=device)
    else:
        if idx_mapping.ndim != 1:
            raise ValueError("idx_mapping must be a 1D tensor.")
        idx_mapping = idx_mapping.to(dtype=torch.int32)

    selected_states: list[DiffusionRequestState] = []
    for batch_idx, state_idx in enumerate(idx_mapping.tolist()):
        if state_idx < 0 or state_idx >= len(states):
            raise ValueError(f"idx_mapping[{batch_idx}]={state_idx} is out of range for states.")
        selected_states.append(states[state_idx])
    return selected_states, idx_mapping, idx_mapping.detach().cpu().numpy()


def _prepare_request_ids(states: Sequence[DiffusionRequestState]) -> list[str]:
    return [state.request_id for state in states]


def _prepare_reused_buffer(
    current: torch.Tensor | None,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if current is not None and tuple(current.shape) == shape and current.dtype == dtype and current.device == device:
        return current
    return torch.empty(shape, dtype=dtype, device=device)


def _validate_gather_tensors(
    values: Sequence[torch.Tensor],
    *,
    field_name: str,
) -> tuple[torch.dtype, torch.device, tuple[int, ...], int]:
    if not values:
        raise ValueError(f"Cannot gather empty tensor list for {field_name}.")

    first = values[0]
    dtype = first.dtype
    device = first.device
    suffix_shape = tuple(first.shape[1:])
    total_rows = 0

    for value in values:
        if value.dtype != dtype:
            raise ValueError(f"Mixed dtypes in {field_name} batch.")
        if value.device != device:
            raise ValueError(f"Mixed devices in {field_name} batch.")
        if tuple(value.shape[1:]) != suffix_shape:
            raise ValueError(
                f"Mixed trailing shapes in {field_name} batch: expected {suffix_shape}, got {tuple(value.shape[1:])}."
            )
        total_rows += int(value.shape[0])

    return dtype, device, suffix_shape, total_rows


def _gather_tensor_rows(
    values: Sequence[torch.Tensor],
    *,
    field_name: str,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    dtype, device, suffix_shape, total_rows = _validate_gather_tensors(
        values,
        field_name=field_name,
    )
    gathered = _prepare_reused_buffer(
        out,
        shape=(total_rows, *suffix_shape),
        dtype=dtype,
        device=device,
    )

    row_offset = 0
    for value in values:
        next_row_offset = row_offset + int(value.shape[0])
        gathered[row_offset:next_row_offset].copy_(value)
        row_offset = next_row_offset
    return gathered


def _prepare_latents(
    states: Sequence[DiffusionRequestState],
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    latents_values = [state.latents for state in states]
    if any(latents is None for latents in latents_values):
        raise ValueError("All requests must have `latents` initialized.")
    return _gather_tensor_rows(
        [latents for latents in latents_values if latents is not None],
        field_name="latents",
        out=out,
    )


def _row_counts(states: Sequence[DiffusionRequestState]) -> list[int]:
    counts: list[int] = []
    for state in states:
        if state.latents is None:
            raise ValueError(f"Request {state.request_id} has no latents.")
        counts.append(int(state.latents.shape[0]))
    return counts


def _expand_scalar_or_vector(
    value: torch.Tensor,
    *,
    num_rows: int,
    field_name: str,
) -> torch.Tensor:
    if value.ndim == 0:
        return value.reshape(1).expand(num_rows)
    if value.ndim != 1:
        raise ValueError(f"{field_name} must be scalar or 1D, got ndim={value.ndim}.")
    if value.shape[0] == num_rows:
        return value
    if value.shape[0] == 1:
        return value.expand(num_rows)
    raise ValueError(
        f"Per-request {field_name} must have either 1 element or {num_rows} elements; got {value.shape[0]}."
    )


def _prepare_timesteps(
    states: Sequence[DiffusionRequestState],
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    timestep_values: list[torch.Tensor] = []
    for state in states:
        timestep = state.current_timestep
        if timestep is None:
            raise ValueError("All requests must have a current timestep initialized.")
        if not torch.is_tensor(timestep):
            raise ValueError("InputBatch expects tensor timesteps; normalize them before batching.")
        if state.latents is None:
            raise ValueError(f"Request {state.request_id} has no latents while preparing timesteps.")
        timestep_values.append(
            _expand_scalar_or_vector(
                timestep,
                num_rows=int(state.latents.shape[0]),
                field_name="timestep tensor",
            )
        )

    return _gather_tensor_rows(
        timestep_values,
        field_name="timesteps",
        out=out,
    )


def _prepare_cfg_scalars(states: Sequence[DiffusionRequestState]) -> tuple[bool, float, bool]:
    scalars = get_runner_step_config(states[0])
    for state in states[1:]:
        if get_runner_step_config(state) != scalars:
            raise ValueError("Mixed CFG settings in one diffusion batch are not supported.")
    return scalars


def _same_composition(
    cached_batch: InputBatch | None,
    request_ids: list[str],
    idx_mapping_np: np.ndarray,
) -> bool:
    if cached_batch is None:
        return False
    if cached_batch.request_ids != request_ids:
        return False
    return np.array_equal(cached_batch.idx_mapping_np, idx_mapping_np)


def _scatter_batch_tensor_by_mapping(
    states: Sequence[DiffusionRequestState],
    idx_mapping_np: np.ndarray,
    *,
    attr_name: str,
    value: torch.Tensor,
) -> None:
    row_offset = 0
    for batch_idx, state_idx in enumerate(idx_mapping_np.tolist()):
        if state_idx < 0 or state_idx >= len(states):
            raise ValueError(f"idx_mapping[{batch_idx}]={state_idx} is out of range for states.")
        state = states[state_idx]
        state_value = getattr(state, attr_name)
        num_rows = 1 if state_value is None else int(state_value.shape[0])
        next_row_offset = row_offset + num_rows
        value_slice = value[row_offset:next_row_offset]
        if state_value is None:
            setattr(state, attr_name, value_slice.clone())
        elif (
            tuple(state_value.shape) != tuple(value_slice.shape)
            or state_value.dtype != value_slice.dtype
            or state_value.device != value_slice.device
        ):
            setattr(state, attr_name, value_slice.clone())
        else:
            state_value.copy_(value_slice)
        row_offset = next_row_offset

    if row_offset != int(value.shape[0]):
        raise ValueError(
            f"Scatter for {attr_name} consumed {row_offset} rows, but batch has {int(value.shape[0])} rows."
        )


@dataclass
class InputBatch:
    """Ephemeral, model-agnostic step-level batch view."""

    states: list[DiffusionRequestState]
    request_ids: list[str]
    num_reqs: int
    num_reqs_after_padding: int
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray
    row_counts: list[int]
    output_splits: list[int]

    latents: torch.Tensor
    timesteps: torch.Tensor
    do_true_cfg: bool = False
    true_cfg_scale: float = 4.0
    cfg_normalize: bool = False
    model_inputs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.request_ids) != int(self.idx_mapping.numel()):
            raise ValueError("`request_ids` and `idx_mapping` must have the same length.")
        if self.num_reqs != len(self.request_ids):
            raise ValueError("`num_reqs` must match the number of request ids.")
        if self.num_reqs_after_padding < self.num_reqs:
            raise ValueError("`num_reqs_after_padding` must be >= `num_reqs`.")
        if len(self.row_counts) != self.num_reqs:
            raise ValueError("`row_counts` must have one entry per request.")
        if len(self.output_splits) != self.num_reqs:
            raise ValueError("`output_splits` must have one entry per request.")

    def _refresh_dynamic_fields(
        self,
        selected_states: Sequence[DiffusionRequestState],
    ) -> None:
        self.states = list(selected_states)
        self.row_counts = _row_counts(selected_states)
        self.output_splits = list(self.row_counts)
        self.latents = _prepare_latents(selected_states, out=self.latents)
        self.timesteps = _prepare_timesteps(selected_states, out=self.timesteps)

    def _refresh_static_fields(
        self,
        states: Sequence[DiffusionRequestState],
    ) -> None:
        self.do_true_cfg, self.true_cfg_scale, self.cfg_normalize = _prepare_cfg_scalars(states)

    def _repack_dynamic_fields(
        self,
        selected_states: Sequence[DiffusionRequestState],
    ) -> None:
        self._refresh_dynamic_fields(selected_states)
        self._refresh_static_fields(selected_states)
        self.model_inputs = {}

    def _rebuild(
        self,
        selected_states: Sequence[DiffusionRequestState],
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        request_ids: list[str],
    ) -> InputBatch:
        self.states = list(selected_states)
        self.request_ids = request_ids
        self.num_reqs = len(request_ids)
        self.num_reqs_after_padding = len(request_ids)
        self.idx_mapping = idx_mapping
        self.idx_mapping_np = idx_mapping_np
        self.row_counts = _row_counts(selected_states)
        self.output_splits = list(self.row_counts)
        self.latents = _prepare_latents(selected_states, out=self.latents)
        self.timesteps = _prepare_timesteps(selected_states, out=self.timesteps)
        self._refresh_static_fields(selected_states)
        self.model_inputs = {}
        self.__post_init__()
        return self

    @classmethod
    def make_batch(
        cls,
        states: Sequence[DiffusionRequestState],
        idx_mapping: torch.Tensor | None = None,
        cached_batch: InputBatch | None = None,
    ) -> InputBatch:
        """Build a temporary step-local batch view from request states."""

        selected_states, idx_mapping, idx_mapping_np = _select_states(states, idx_mapping)
        request_ids = _prepare_request_ids(selected_states)

        if _same_composition(cached_batch, request_ids, idx_mapping_np):
            assert cached_batch is not None
            cached_batch._repack_dynamic_fields(selected_states)
            return cached_batch

        if cached_batch is not None:
            return cached_batch._rebuild(
                selected_states,
                idx_mapping,
                idx_mapping_np,
                request_ids,
            )

        do_true_cfg, true_cfg_scale, cfg_normalize = _prepare_cfg_scalars(selected_states)
        row_counts = _row_counts(selected_states)
        return cls(
            states=list(selected_states),
            request_ids=request_ids,
            num_reqs=len(selected_states),
            num_reqs_after_padding=len(selected_states),
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            row_counts=row_counts,
            output_splits=list(row_counts),
            latents=_prepare_latents(selected_states),
            timesteps=_prepare_timesteps(selected_states),
            do_true_cfg=do_true_cfg,
            true_cfg_scale=true_cfg_scale,
            cfg_normalize=cfg_normalize,
        )


def scatter_latents(
    states: Sequence[DiffusionRequestState],
    input_batch: InputBatch,
) -> None:
    """Scatter the step-updated latents back into persistent request states."""

    # Scatter uses the same idx_mapping that built the batch, so padded or
    # reordered batch rows are written back to their original request states.
    _scatter_batch_tensor_by_mapping(
        states,
        input_batch.idx_mapping_np,
        attr_name="latents",
        value=input_batch.latents,
    )


DiffusionInputBatch = InputBatch
