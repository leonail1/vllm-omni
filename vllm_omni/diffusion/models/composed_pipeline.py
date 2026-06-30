# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Composable diffusion pipeline stages.

This mixin is the model-side half of the Atoms -> Stages design.  Concrete
pipelines own the small atom methods, while this class assembles them into the
shared stage contract used by request-mode forward and step execution.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import DiffusionRequestState

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.request import OmniDiffusionRequest


class ComposedDiffusionPipeline:
    """Build common stages and forward() from model-provided atoms.

    Models implement the atom methods and optionally override
    :attr:`ENCODE_ATOMS` to express their request setup order.

    There is intentionally no independent ``supports_stage_execution`` flag:
    the stage methods are the single state-driven pipeline contract.
    """

    supports_step_execution: ClassVar[bool] = True
    ENCODE_ATOMS: ClassVar[tuple[str, ...]] = (
        "check_inputs",
        "encode_prompt",
        "prepare_timesteps",
        "prepare_latents",
    )
    def _state_from_request(self, req: "OmniDiffusionRequest") -> DiffusionRequestState:
        # Keep request-to-state construction generic; model atoms own model-specific fields.
        sampling = copy.deepcopy(req.sampling_params)
        state = DiffusionRequestState(
            request_id=req.request_id,
            sampling=sampling,
            prompts=req.prompts,
        )
        if sampling.generator is None and sampling.seed is not None:
            device = sampling.generator_device or getattr(self, "device", "cpu")
            sampling.generator = torch.Generator(device=device).manual_seed(sampling.seed)
        return state

    def encode_stage(self, state: DiffusionRequestState) -> DiffusionRequestState:
        # ENCODE_ATOMS is the only per-model ordering knob in the shared encode stage.
        for name in self.ENCODE_ATOMS:
            getattr(self, name)(state)
        return state

    def denoise_stage(self, batch: InputBatch) -> torch.Tensor | None:
        # Delegate the shared DiT stage to the model-specific noise atom.
        return self.predict_noise(batch)

    def scheduler_stage(
        self,
        state: DiffusionRequestState,
        noise: torch.Tensor | None,
    ) -> None:
        # Keep scheduler mutation in the model atom so each pipeline can own its scheduler semantics.
        self.advance_scheduler(state, noise)

    def decode_stage(self, state: DiffusionRequestState) -> "DiffusionOutput":
        # Delegate decode/postprocess to the model-specific decode atom.
        return self.decode(state)

    def forward(self, req: "OmniDiffusionRequest") -> "DiffusionOutput":
        # Request-mode execution uses the same stages with a local single-request loop.
        state = self._state_from_request(req)
        self.encode_stage(state)
        while not state.denoise_completed:
            batch = InputBatch.make_batch([state])
            noise = self.denoise_stage(batch)
            self.scheduler_stage(state, noise)
        return self.decode_stage(state)

    def encode_conditioning(self, state: DiffusionRequestState) -> None:
        """Optional atom for edit/i2v pipelines."""
        # Text-to-image pipelines have no extra conditioning work.
        return None
