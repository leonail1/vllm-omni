# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import (
    QWEN_IMAGE_STAGE_KIND_KEY,
    QWEN_IMAGE_STAGE_PAYLOAD_KEY,
)

logger = init_logger(__name__)


def _get_single_stage_payload(
    source_outputs: list[Any],
    expected_kind: str,
) -> dict[str, Any]:
    if not source_outputs:
        raise RuntimeError(f"Qwen-Image {expected_kind} stage bridge received no upstream output.")

    custom_output = getattr(source_outputs[0], "custom_output", {}) or {}
    kind = custom_output.get(QWEN_IMAGE_STAGE_KIND_KEY)
    payload = custom_output.get(QWEN_IMAGE_STAGE_PAYLOAD_KEY)
    if kind != expected_kind or not isinstance(payload, dict):
        raise RuntimeError(
            f"Qwen-Image stage bridge expected {expected_kind!r} payload, "
            f"got kind={kind!r}, keys={list(custom_output.keys())}."
        )
    return payload


def _wrap_stage_payload(payload: dict[str, Any], kind: str) -> dict[str, Any]:
    return {
        QWEN_IMAGE_STAGE_KIND_KEY: kind,
        QWEN_IMAGE_STAGE_PAYLOAD_KEY: payload,
    }


def encode_to_denoise(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    sampling_params: Any | None = None,
) -> dict[str, Any]:
    del prompt, requires_multimodal_data, sampling_params
    return _wrap_stage_payload(_get_single_stage_payload(source_outputs, "encode"), "encode")


def denoise_to_decode(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    sampling_params: Any | None = None,
) -> dict[str, Any]:
    del prompt, requires_multimodal_data, sampling_params
    return _wrap_stage_payload(_get_single_stage_payload(source_outputs, "denoise"), "denoise")
