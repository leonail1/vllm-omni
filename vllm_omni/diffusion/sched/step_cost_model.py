# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class StepCostEstimate:
    step_ms: float
    source: str
    latent_tokens: int | None = None


@dataclass(frozen=True)
class _LatentQuadraticFormula:
    coefficients: tuple[float, float, float, float, float]
    token_scale: float = 4096.0
    supported_models: tuple[str, ...] = ()
    min_latent_tokens: int | None = None
    max_latent_tokens: int | None = None

    def estimate(self, latent_tokens: int, batch_eff: float) -> float:
        c0, c1, c2, c3, c4 = self.coefficients
        tokens = max(float(latent_tokens), 1.0) / max(float(self.token_scale), 1.0)
        batch = max(float(batch_eff), 1.0)
        return c0 + c1 * tokens + c2 * batch + c3 * tokens * batch + c4 * tokens * tokens * batch

    def supports(self, model: str, latent_tokens: int) -> bool:
        if self.supported_models and not any(
            _model_matches(profiled_model, model) for profiled_model in self.supported_models
        ):
            return False
        if self.min_latent_tokens is not None and latent_tokens < self.min_latent_tokens:
            return False
        if self.max_latent_tokens is not None and latent_tokens > self.max_latent_tokens:
            return False
        return True


class DiffusionStepCostModel:
    """Offline-profiled denoise step cost lookup with formula fallback."""

    def __init__(
        self,
        *,
        model_path: str | None = None,
        metric: str = "p90_ms",
        safety_factor: float = 1.0,
        default_step_ms: float = 1.0,
        token_patch_size: int = 16,
        fallback_formula: _LatentQuadraticFormula | None = None,
    ) -> None:
        self.metric = metric
        self.safety_factor = max(float(safety_factor), 0.001)
        self.default_step_ms = max(float(default_step_ms), 0.001)
        self.token_patch_size = max(int(token_patch_size), 1)
        self._exact_table: dict[tuple[str, str, int, float], float] = {}
        self._latent_table: dict[tuple[str, int, int, float], float] = {}
        self._known_models: set[str] = set()
        self._json_area_fallback_models: set[str] = set()
        self._json_area_fallback: tuple[float, float, float, float] | None = None
        self._fallback_formula = fallback_formula
        if model_path:
            self._load(Path(model_path))

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        default_step_ms: float = 1.0,
    ) -> DiffusionStepCostModel | None:
        model_path = config.get("step_cost_model_path") or config.get("cost_model_path")
        formula = _formula_from_config(config.get("step_cost_formula") or config.get("fallback_formula"))
        if not model_path and formula is None:
            return None
        return cls(
            model_path=str(model_path) if model_path else None,
            metric=str(config.get("step_cost_metric") or config.get("cost_metric") or "p90_ms"),
            safety_factor=_coerce_float(config.get("step_cost_safety_factor"), 1.0),
            default_step_ms=default_step_ms,
            token_patch_size=int(_coerce_float(config.get("latent_patch_size"), 16.0)),
            fallback_formula=formula,
        )

    def estimate(
        self,
        *,
        model: str | None,
        width: float | int | None,
        height: float | int | None,
        num_frames: float | int | None = 1,
        batch_size: int = 1,
        effective_batch_size: float | None = None,
    ) -> StepCostEstimate:
        model_key = model or ""
        eff = max(float(effective_batch_size if effective_batch_size is not None else batch_size), 1.0)
        batch_int = max(int(batch_size), 1)
        shape_key = _shape_key(width, height, num_frames)
        if shape_key is not None:
            value = self._lookup_exact(model_key, shape_key, batch_int, eff)
            if value is not None:
                return self._finish(value, "exact_table", width, height, num_frames)

        latent_tokens = self.latent_tokens(width, height, num_frames)
        if latent_tokens is not None:
            value = self._lookup_latent(model_key, latent_tokens, batch_int, eff)
            if value is not None:
                return self._finish(value, "latent_table", width, height, num_frames)
            if self._fallback_formula is not None and self._fallback_formula.supports(model_key, latent_tokens):
                value = self._fallback_formula.estimate(latent_tokens, eff)
                return self._finish(value, "latent_formula", width, height, num_frames)

        if (
            self._json_area_fallback is not None
            and width is not None
            and height is not None
            and self._is_json_fallback_model(model_key)
        ):
            c0, c1, c2, gamma = self._json_area_fallback
            area = max(float(width) * float(height) / float(1024 * 1024), 0.001)
            value = c0 + c1 * area + c2 * area * (eff**gamma)
            return self._finish(value, "area_formula", width, height, num_frames)

        return StepCostEstimate(self.default_step_ms, "default", latent_tokens)

    def latent_tokens(
        self,
        width: float | int | None,
        height: float | int | None,
        num_frames: float | int | None = 1,
    ) -> int | None:
        if width is None or height is None:
            return None
        try:
            w = float(width)
            h = float(height)
            frames = max(float(num_frames or 1), 1.0)
        except (TypeError, ValueError):
            return None
        tokens = (w / self.token_patch_size) * (h / self.token_patch_size) * frames
        return max(int(round(tokens)), 1)

    def _finish(
        self,
        value: float,
        source: str,
        width: float | int | None,
        height: float | int | None,
        num_frames: float | int | None,
    ) -> StepCostEstimate:
        latent_tokens = self.latent_tokens(width, height, num_frames)
        return StepCostEstimate(max(value * self.safety_factor, 0.001), source, latent_tokens)

    def _load(self, path: Path) -> None:
        try:
            with path.open(encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.warning("[DiffusionStepCostModel] failed to load %s: %s", path, exc)
            return

        latent_values: dict[tuple[str, int, int, float], list[float]] = defaultdict(list)
        table_lookup = payload.get("table_lookup", {})
        if isinstance(table_lookup, dict):
            for model_key, shape_table in table_lookup.items():
                if not isinstance(shape_table, dict):
                    continue
                model_key = str(model_key)
                if model_key:
                    self._known_models.add(model_key)
                for shape_key, batch_table in shape_table.items():
                    parsed_shape = _parse_shape_key(shape_key)
                    for batch_key, eff_table in _iter_dict(batch_table):
                        batch_int = int(_coerce_float(batch_key, 1.0))
                        for eff_key, row in _iter_dict(eff_table):
                            if not isinstance(row, dict):
                                continue
                            value = _coerce_float(row.get(self.metric), None)
                            if value is None:
                                value = _coerce_float(row.get("denoise_step_ms"), None)
                            eff = _coerce_float(row.get("effective_batch_size"), _coerce_float(eff_key, batch_int))
                            if value is None or eff is None:
                                continue
                            self._exact_table[(model_key, str(shape_key), batch_int, float(eff))] = float(value)
                            if parsed_shape is not None:
                                width, height, frames = parsed_shape
                                latent_tokens = self.latent_tokens(width, height, frames)
                                if latent_tokens is not None:
                                    latent_values[(model_key, latent_tokens, batch_int, float(eff))].append(
                                        float(value)
                                    )

        for key, values in latent_values.items():
            self._latent_table[key] = statistics.median(values)

        fallback = payload.get("fallback", {})
        if isinstance(fallback, dict):
            coeffs = fallback.get("coefficients", {})
            if isinstance(coeffs, dict):
                c0 = _coerce_float(coeffs.get("c0"), None)
                c1 = _coerce_float(coeffs.get("c1"), None)
                c2 = _coerce_float(coeffs.get("c2"), None)
                gamma = _coerce_float(fallback.get("gamma"), 1.0)
                if c0 is not None and c1 is not None and c2 is not None:
                    self._json_area_fallback = (c0, c1, c2, gamma)
                    self._json_area_fallback_models = set(self._known_models)

    def _lookup_exact(self, model: str, shape_key: str, batch_size: int, eff: float) -> float | None:
        candidates = [
            ((candidate_model, candidate_shape, candidate_batch, candidate_eff), value)
            for (candidate_model, candidate_shape, candidate_batch, candidate_eff), value in self._exact_table.items()
            if candidate_shape == shape_key
            and candidate_batch == batch_size
            and _model_matches(candidate_model, model)
        ]
        return _nearest_eff_value(candidates, eff)

    def _lookup_latent(self, model: str, latent_tokens: int, batch_size: int, eff: float) -> float | None:
        candidates = [
            ((candidate_model, candidate_tokens, candidate_eff), value)
            for (candidate_model, candidate_tokens, candidate_batch, candidate_eff), value in self._latent_table.items()
            if candidate_tokens == latent_tokens
            and candidate_batch == batch_size
            and _model_matches(candidate_model, model)
        ]
        return _nearest_eff_value(candidates, eff)

    def _is_json_fallback_model(self, model: str) -> bool:
        if not self._json_area_fallback_models:
            return True
        return any(_model_matches(profiled_model, model) for profiled_model in self._json_area_fallback_models)


def estimate_request_effective_size(
    sampling_params: Any,
    *,
    prompts: Any = None,
    do_true_cfg: bool | None = None,
    has_negative_prompt: bool | None = None,
) -> float:
    outputs = _coerce_float(getattr(sampling_params, "num_outputs_per_prompt", None), 1.0)
    guidance = 1.0
    true_cfg_scale = _coerce_float(getattr(sampling_params, "true_cfg_scale", None), None)
    if do_true_cfg is None:
        if has_negative_prompt is None:
            has_negative_prompt = _has_negative_prompt(prompts)
        do_true_cfg = true_cfg_scale is not None and true_cfg_scale > 1.0 and bool(has_negative_prompt)
    if bool(do_true_cfg):
        guidance = 2.0
    return max(outputs, 1.0) * guidance


def _formula_from_config(raw: Any) -> _LatentQuadraticFormula | None:
    if isinstance(raw, str):
        if raw in {"qwen_image_910b_tp2_v1", "qwen_image_910b_tp2_tokens_ge_1024"}:
            return _LatentQuadraticFormula(
                (
                    288.68,
                    -230.01,
                    -38.12,
                    335.52,
                    82.63,
                ),
                token_scale=4096.0,
                supported_models=("Qwen/Qwen-Image", "Qwen-Image"),
                min_latent_tokens=1024,
                max_latent_tokens=4096,
            )
        if raw == "qwen_image_910b_tp2_all":
            return _LatentQuadraticFormula(
                (
                    319.25,
                    -273.16,
                    -17.38,
                    246.55,
                    158.92,
                ),
                token_scale=4096.0,
                supported_models=("Qwen/Qwen-Image", "Qwen-Image"),
                min_latent_tokens=256,
                max_latent_tokens=4096,
            )
        return None
    if not isinstance(raw, dict):
        return None
    formula_type = str(raw.get("type") or "latent_quadratic_batch")
    if formula_type != "latent_quadratic_batch":
        return None
    coefficients = raw.get("coefficients")
    if isinstance(coefficients, dict):
        ordered = [coefficients.get(name) for name in ("c0", "c1", "c2", "c3", "c4")]
    elif isinstance(coefficients, (list, tuple)):
        ordered = list(coefficients)
    else:
        return None
    parsed = [_coerce_float(value, None) for value in ordered[:5]]
    if len(parsed) != 5 or any(value is None for value in parsed):
        return None
    token_scale = _coerce_float(raw.get("token_scale"), 4096.0)
    raw_models = raw.get("models") or raw.get("supported_models") or ()
    if isinstance(raw_models, str):
        supported_models = (raw_models,)
    elif isinstance(raw_models, (list, tuple)):
        supported_models = tuple(str(model) for model in raw_models)
    else:
        supported_models = ()
    min_latent_tokens = _coerce_float(raw.get("min_latent_tokens"), None)
    max_latent_tokens = _coerce_float(raw.get("max_latent_tokens"), None)
    return _LatentQuadraticFormula(
        tuple(float(value) for value in parsed),
        token_scale=token_scale,
        supported_models=supported_models,
        min_latent_tokens=None if min_latent_tokens is None else int(min_latent_tokens),
        max_latent_tokens=None if max_latent_tokens is None else int(max_latent_tokens),
    )


def _has_negative_prompt(prompts: Any) -> bool:
    if prompts is None:
        return False
    if isinstance(prompts, dict):
        return prompts.get("negative_prompt") is not None
    if isinstance(prompts, (list, tuple)):
        return any(_has_negative_prompt(prompt) for prompt in prompts)
    return False


def _shape_key(
    width: float | int | None,
    height: float | int | None,
    num_frames: float | int | None,
) -> str | None:
    if width is None or height is None:
        return None
    try:
        return f"{int(float(width))}x{int(float(height))}x{int(float(num_frames or 1))}"
    except (TypeError, ValueError):
        return None


def _parse_shape_key(shape_key: str) -> tuple[int, int, int] | None:
    try:
        width_s, height_s, frames_s = shape_key.split("x")
        return int(width_s), int(height_s), int(frames_s)
    except (AttributeError, ValueError):
        return None


def _iter_dict(value: Any):
    if isinstance(value, dict):
        return value.items()
    return ()


def _nearest_eff_value(candidates: list[tuple[tuple[Any, ...], float]], eff: float) -> float | None:
    if not candidates:
        return None
    _, value = min(candidates, key=lambda item: abs(float(item[0][-1]) - eff))
    return value


def _model_matches(profiled_model: str, requested_model: str) -> bool:
    if not profiled_model:
        return True
    if not requested_model:
        return False
    profiled_aliases = _model_aliases(profiled_model)
    requested_aliases = _model_aliases(requested_model)
    return not profiled_aliases.isdisjoint(requested_aliases)


def _model_aliases(model: str) -> set[str]:
    normalized = str(model).strip()
    if not normalized:
        return set()
    trimmed = normalized.rstrip("/")
    basename = trimmed.rsplit("/", 1)[-1]
    aliases = {normalized, trimmed}
    if basename:
        aliases.add(basename)
    return {alias.lower() for alias in aliases if alias}


def _coerce_float(value: Any, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "DiffusionStepCostModel",
    "StepCostEstimate",
    "estimate_request_effective_size",
]
