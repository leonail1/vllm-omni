# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.outputs import OmniRequestOutput


def is_latent_output_request(request: OmniDiffusionRequest, output_type_config: Any = None) -> bool:
    """Return whether this request asks the engine to return raw latents."""
    output_type = getattr(request.sampling_params, "output_type", None)
    if output_type is None:
        output_type = output_type_config
    return str(output_type).lower() in {"latent", "latents"}


def _slice_latent_output(value: Any, start: int, end: int) -> Any:
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] >= end:
        return value[start:end]
    if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] >= end:
        return value[start:end]
    if isinstance(value, list):
        return value[start:end] if len(value) >= end else value
    if isinstance(value, tuple):
        return value[start:end] if len(value) >= end else value
    if isinstance(value, dict):
        return {key: _slice_latent_output(item, start, end) for key, item in value.items()}
    return value


def make_latent_request_outputs(
    request: OmniDiffusionRequest,
    output: DiffusionOutput,
    latents: Any,
    metrics: dict[str, Any],
) -> list[OmniRequestOutput]:
    """Wrap latent tensors in OmniRequestOutput without image postprocessing."""
    custom_output = output.custom_output or {}
    if len(request.prompts) == 1:
        return [
            OmniRequestOutput.from_diffusion(
                request_id=request.request_id,
                images=[],
                prompt=request.prompts[0],
                metrics=metrics,
                latents=latents,
                trajectory_latents=output.trajectory_latents,
                trajectory_timesteps=output.trajectory_timesteps,
                trajectory_log_probs=output.trajectory_log_probs,
                trajectory_decoded=output.trajectory_decoded,
                custom_output=custom_output,
                final_output_type="latents",
                stage_durations=output.stage_durations,
                peak_memory_mb=output.peak_memory_mb,
            ),
        ]

    results = []
    num_outputs = int(request.sampling_params.num_outputs_per_prompt)
    for idx, prompt in enumerate(request.prompts):
        start = idx * num_outputs
        end = start + num_outputs
        results.append(
            OmniRequestOutput.from_diffusion(
                request_id=request.request_id,
                images=[],
                prompt=prompt,
                metrics=metrics,
                latents=_slice_latent_output(latents, start, end),
                trajectory_latents=output.trajectory_latents,
                trajectory_timesteps=output.trajectory_timesteps,
                trajectory_log_probs=output.trajectory_log_probs,
                trajectory_decoded=output.trajectory_decoded,
                custom_output=custom_output,
                final_output_type="latents",
                stage_durations=output.stage_durations,
                peak_memory_mb=output.peak_memory_mb,
            )
        )
    return results
