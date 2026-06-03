"""Unified stage-local runtime abstraction for vLLM-Omni."""

from __future__ import annotations

import asyncio
import json
import time as _time
from collections.abc import Sequence
from dataclasses import MISSING, dataclass, fields
from typing import TYPE_CHECKING, Any, cast

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreOutputs

from vllm_omni.distributed.omni_coordinator import (
    LoadBalancer,
    OmniCoordClientForHub,
    ReplicaInfo,
    ReplicaStatus,
)
from vllm_omni.distributed.omni_coordinator.load_balancer import Task
from vllm_omni.diffusion.sched.base_scheduler import (
    qwen_image_dynamic_step_batching_enabled,
)
from vllm_omni.diffusion.sched.interface import SamplingParamsKey
from vllm_omni.diffusion.sched.step_cost_model import (
    DiffusionStepCostModel,
    estimate_request_effective_size,
)
from vllm_omni.engine.stage_client import (
    StagePoolClient,
    StagePoolDiffusionClient,
    StagePoolLLMClient,
)
from vllm_omni.metrics.stats import StageRequestStats as StageRequestMetrics
from vllm_omni.metrics.stats import StageStats
from vllm_omni.metrics.utils import count_tokens_from_outputs

if TYPE_CHECKING:
    from vllm_omni.engine.orchestrator import OrchestratorRequestState

logger = init_logger(__name__)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _stage_slo_scheduler_config(stage_vllm_config: Any) -> dict[str, Any]:
    additional_config = getattr(stage_vllm_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        return {}
    raw = additional_config.get("diffusion_slo_scheduler") or additional_config.get("slo_scheduler") or {}
    return raw if isinstance(raw, dict) else {}


def _stage_model_name(stage_vllm_config: Any) -> str | None:
    model = getattr(stage_vllm_config, "model", None)
    if isinstance(model, str):
        return model
    model_config = getattr(stage_vllm_config, "model_config", None)
    for attr in ("model", "served_model_name"):
        value = getattr(model_config, attr, None)
        if isinstance(value, str):
            return value
    return None


def _sampling_get(sampling_params: Any, name: str, default: Any = None) -> Any:
    if sampling_params is None:
        return default
    if isinstance(sampling_params, dict):
        return sampling_params.get(name, default)
    return getattr(sampling_params, name, default)


def _sampling_dimensions(sampling_params: Any) -> tuple[float | None, float | None]:
    height = _optional_float(_sampling_get(sampling_params, "height"))
    width = _optional_float(_sampling_get(sampling_params, "width"))
    return width, height


def _dimensions_from_key(key: dict[str, Any] | None) -> tuple[float | None, float | None]:
    if key is None:
        return None, None
    height = _optional_float(key.get("height"))
    width = _optional_float(key.get("width"))
    return width, height


def _prompt_count_from_request(request: Any) -> float:
    if isinstance(request, (list, tuple)):
        return float(max(len(request), 1))
    prompts = getattr(request, "prompts", None)
    if isinstance(prompts, (list, tuple)):
        return float(max(len(prompts), 1))
    return 1.0


def _has_negative_prompt(value: Any) -> bool:
    if value is None:
        return False
    prompts = getattr(value, "prompts", None)
    if prompts is not None:
        return _has_negative_prompt(prompts)
    if isinstance(value, dict):
        return value.get("negative_prompt") is not None
    if isinstance(value, (list, tuple)):
        return any(_has_negative_prompt(item) for item in value)
    return False


def _task_prompt_count(task: Task | None) -> float:
    if task is None:
        return 1.0
    return max(_optional_float(task.get("prompt_count")) or 1.0, 1.0)


def _task_effective_size(task: Task | None) -> float:
    sampling_params = None if task is None else task.get("sampling_params")
    return _task_prompt_count(task) * estimate_request_effective_size(
        sampling_params,
        has_negative_prompt=bool(task.get("has_negative_prompt")) if task is not None else False,
    )


def _snapshot_incremental_step_ms(
    bucket: dict[str, Any],
    incoming_eff: float,
    current_step_ms: float,
) -> float | None:
    incremental = _optional_float(bucket.get("incremental_step_ms_if_add_one"))
    if incremental is None:
        step_if_add_one = _optional_float(bucket.get("estimated_step_ms_if_add_one"))
        if step_if_add_one is not None:
            incremental = max(step_if_add_one - current_step_ms, 0.0)
    if incremental is None:
        return None
    return max(incremental, 0.0) * max(float(incoming_eff), 1.0)


def _snapshot_token_incremental_step_ms(
    bucket: dict[str, Any],
    incoming_tokens: float,
    incoming_eff: float,
    current_step_ms: float,
    batch_growth_alpha: float,
) -> float | None:
    bucket_tokens = _optional_float(bucket.get("max_latent_tokens")) or _optional_float(bucket.get("latent_tokens"))
    existing_work = _optional_float(bucket.get("total_token_work"))
    if bucket_tokens is None or bucket_tokens <= 0 or existing_work is None or current_step_ms <= 0:
        return None

    current_max_tokens = max(float(bucket_tokens), 1.0)
    incoming_tokens = max(float(incoming_tokens), 1.0)
    incoming_eff = max(float(incoming_eff), 1.0)
    after_max_tokens = max(current_max_tokens, incoming_tokens)
    current_eff = max(existing_work / current_max_tokens, 1.0)
    after_eff = max((existing_work + incoming_tokens * incoming_eff) / after_max_tokens, 1.0)
    current_scale = 1.0 + batch_growth_alpha * max(current_eff - 1.0, 0.0)
    after_scale = 1.0 + batch_growth_alpha * max(after_eff - 1.0, 0.0)
    current_single_step_ms = current_step_ms / max(current_scale, 0.001)
    token_size_scale = max(after_max_tokens / current_max_tokens, 1.0)
    after_step_ms = current_single_step_ms * token_size_scale * after_scale
    return max(after_step_ms - current_step_ms, 0.0)


@dataclass
class _ReplicaMetrics:
    """Per-replica metrics accumulators owned by a stage pool."""

    batch_seq: int = 0
    agg_total_tokens: int = 0
    agg_total_gen_time_ms: float = 0.0


class StagePool:
    """Replicas of one logical stage + per-stage routing (LB + affinity).

    The pool owns the head-side stage clients for one logical stage. It also
    absorbs the per-stage dispatch responsibility (load balancing, affinity
    tracking, bounded-wait pick) that used to live in a separate
    ``StageDispatcher`` class — see the design doc for the rationale.

    In distributed mode (when an :class:`OmniCoordClientForHub` and a
    :class:`LoadBalancer` are injected via :meth:`attach_hub` /
    :meth:`attach_load_balancer`), :meth:`pick` consults the hub's cached
    replica list and routes via the load balancer, sticking subsequent calls
    for the same ``request_id`` to the same replica.

    In non-distributed mode (no hub attached), :meth:`pick` falls back to the
    legacy ``select_replica_id`` round-robin path so the multi-stage
    in-process invocation is unchanged.

    Dynamic replica membership: when a remote replica is added or removed
    (driven by :class:`Orchestrator` via :meth:`add_client` /
    :meth:`remove_client`), the pool keeps stable integer ``replica_id``s by
    storing clients in a list whose entries can be ``None`` after a removal.
    Iteration callers should use :meth:`live_replica_ids` rather than
    ``range(pool.num_replicas)`` to skip the gaps.
    """

    DISPATCH_WAIT_TIMEOUT_S: float = 10.0
    DISPATCH_RETRY_INTERVAL_S: float = 0.1
    SCHEDULER_SNAPSHOT_TTL_S: float = 0.5
    SCHEDULER_SNAPSHOT_RPC_TIMEOUT_S: float = 0.05
    _STAGEPOOL_SELECTION_OBJECTIVES = {
        "legacy",
        "token_slo",
        "token_slo_objective",
        "slo_objective",
        "objective",
    }
    _STAGEPOOL_OBJECTIVE_POLICY_ALIASES = {
        "slo_no_preemption_token_objective",
        "slo_token_stagepool_objective",
    }

    def __init__(
        self,
        stage_id: int,
        clients: StagePoolClient | list[StagePoolClient],
        *,
        output_processor: Any = None,
        stage_vllm_config: Any = None,
        stage_slo_config: Any = None,
        replica_device_ids: Sequence[Sequence[str]] | None = None,
        replica_card_counts: Sequence[int] | None = None,
    ) -> None:
        if isinstance(clients, list):
            normalized_clients: list[StagePoolClient] = list(clients)
        else:
            normalized_clients = [clients]

        # Allow empty pools when running in distributed head mode for a
        # non-self stage; clients will arrive via add_client(...).
        self.stage_id = stage_id
        # Slots can become None after a dynamic remove_client (distributed mode);
        # iterate via live_replica_ids() to skip holes.
        self.clients: list[StagePoolClient | None] = list(normalized_clients)
        self._output_processor = output_processor
        self._stage_vllm_config = stage_vllm_config
        self._stage_slo_config = stage_slo_config if stage_slo_config is not None else stage_vllm_config
        self._dag_replica_device_ids = self._normalize_replica_device_ids(
            replica_device_ids,
            len(self.clients),
        )
        self._dag_replica_card_counts = self._normalize_replica_card_counts(
            replica_card_counts,
            self._dag_replica_device_ids,
            len(self.clients),
        )
        self._next_replica_id = 0
        self._request_bindings: dict[str, int] = {}
        self._replica_metrics: list[_ReplicaMetrics] = [_ReplicaMetrics() for _ in self.clients]

        # Distributed-mode state. Populated by add_client / remove_client.
        self._addr_to_replica_id: dict[str, int] = {}
        for replica_id, client in enumerate(self.clients):
            if client is not None:
                addr = self._client_input_addr(client)
                if addr is not None:
                    self._addr_to_replica_id[addr] = replica_id

        # Distributed-mode dispatch hooks (injected by Orchestrator on bring-up).
        self._hub: OmniCoordClientForHub | None = None
        self._lb: LoadBalancer | None = None
        # ``request_id`` → ``input_addr`` affinity (distributed mode only).
        # Kept separate from the legacy ``_request_bindings`` so the two
        # binding shapes do not collide.
        self._affinity: dict[str, str] = {}
        self._scheduler_snapshot_cache: dict[int, tuple[float, dict[str, Any]]] = {}
        self._slo_tie_break_cursor = 0
        slo_config = _stage_slo_scheduler_config(self._stage_slo_config)
        self._enable_stagepool_slo = _optional_bool(
            slo_config.get("enable_stagepool_slo"),
            self._default_stagepool_slo_enabled(self._stage_slo_config),
        )
        self._slo_default_step_ms = _optional_float(slo_config.get("default_step_ms")) or 1.0
        batch_growth_alpha = _optional_float(slo_config.get("batch_growth_alpha"))
        self._slo_batch_growth_alpha = max(0.60 if batch_growth_alpha is None else batch_growth_alpha, 0.0)
        self._slo_cost_model = DiffusionStepCostModel.from_config(slo_config, default_step_ms=1.0)
        self._slo_default_num_inference_steps = int(_optional_float(slo_config.get("default_num_inference_steps")) or 50)
        self._slo_decode_ms = _optional_float(slo_config.get("decode_ms")) or 0.0
        self._slo_ignore_reference_cost = _optional_bool(slo_config.get("ignore_request_reference_cost"), False)

        laxity_window_ms = _optional_float(slo_config.get("stagepool_laxity_window_ms"))
        self._slo_stagepool_laxity_window_ms = max(laxity_window_ms or 0.0, 0.0)
        pack_min_laxity_ms = _optional_float(slo_config.get("stagepool_pack_min_laxity_ms"))
        self._slo_stagepool_pack_min_laxity_ms = 0.0 if pack_min_laxity_ms is None else pack_min_laxity_ms
        pack_max_queue_length = _optional_float(slo_config.get("stagepool_pack_max_queue_length"))
        self._slo_stagepool_pack_max_queue_length = int(max(pack_max_queue_length or 0.0, 0.0))
        pack_max_matching_bucket_size = _optional_float(slo_config.get("stagepool_pack_max_matching_bucket_size"))
        self._slo_stagepool_pack_max_matching_bucket_size = int(max(pack_max_matching_bucket_size or 0.0, 0.0))
        profile_reject_laxity_ms = _optional_float(slo_config.get("stagepool_profile_reject_laxity_ms"))
        self._slo_stagepool_profile_reject_laxity_ms = (
            0.0 if profile_reject_laxity_ms is None else profile_reject_laxity_ms
        )
        selection_objective = slo_config.get("stagepool_selection_objective")
        if selection_objective is None:
            policy = self._stage_scheduler_policy(self._stage_slo_config)
            selection_objective = (
                "token_slo_objective"
                if isinstance(policy, str) and policy.lower() in self._STAGEPOOL_OBJECTIVE_POLICY_ALIASES
                else "legacy"
            )
        self._slo_stagepool_selection_objective = str(selection_objective).strip().lower()
        if self._slo_stagepool_selection_objective not in self._STAGEPOOL_SELECTION_OBJECTIVES:
            raise ValueError(
                "Unsupported stagepool_selection_objective="
                f"{selection_objective!r}. Expected one of {sorted(self._STAGEPOOL_SELECTION_OBJECTIVES)}."
            )
        self._slo_stagepool_objective_miss_risk_weight = _optional_float(
            slo_config.get("stagepool_objective_miss_risk_weight")
        )
        if self._slo_stagepool_objective_miss_risk_weight is None:
            self._slo_stagepool_objective_miss_risk_weight = 1.0
        self._slo_stagepool_objective_token_pressure_weight = _optional_float(
            slo_config.get("stagepool_objective_token_pressure_weight")
        )
        if self._slo_stagepool_objective_token_pressure_weight is None:
            self._slo_stagepool_objective_token_pressure_weight = 0.01
        self._slo_stagepool_objective_queue_weight = _optional_float(
            slo_config.get("stagepool_objective_queue_weight")
        )
        if self._slo_stagepool_objective_queue_weight is None:
            self._slo_stagepool_objective_queue_weight = 1.0
        self._slo_stagepool_objective_pack_bonus = _optional_float(
            slo_config.get("stagepool_objective_pack_bonus")
        )
        if self._slo_stagepool_objective_pack_bonus is None:
            self._slo_stagepool_objective_pack_bonus = 1.0
        self._slo_stagepool_objective_safe_capacity_bonus = _optional_float(
            slo_config.get("stagepool_objective_safe_capacity_bonus")
        )
        if self._slo_stagepool_objective_safe_capacity_bonus is None:
            self._slo_stagepool_objective_safe_capacity_bonus = 1.0
        self._scheduler_snapshot_ttl_s = (
            0.0
            if self._slo_stagepool_laxity_window_ms > 0
            else self.SCHEDULER_SNAPSHOT_TTL_S
        )
        profile_config = self._stage_profile_config(self._stage_slo_config)
        self._stagepool_profile_enabled = _optional_bool(profile_config.get("enabled"), False)
        self._stagepool_profile_path = profile_config.get("output_path")

    @staticmethod
    def _normalize_replica_device_ids(
        replica_device_ids: Sequence[Sequence[str]] | None,
        num_slots: int,
    ) -> tuple[tuple[str, ...], ...]:
        normalized: list[tuple[str, ...]] = []
        raw = list(replica_device_ids or [])
        for replica_id in range(num_slots):
            values = raw[replica_id] if replica_id < len(raw) else ()
            normalized.append(tuple(str(v) for v in values if str(v) != ""))
        return tuple(normalized)

    @staticmethod
    def _normalize_replica_card_counts(
        replica_card_counts: Sequence[int] | None,
        replica_device_ids: tuple[tuple[str, ...], ...],
        num_slots: int,
    ) -> tuple[int, ...]:
        raw = list(replica_card_counts or [])
        normalized: list[int] = []
        for replica_id in range(num_slots):
            if replica_id < len(raw):
                normalized.append(max(1, int(raw[replica_id])))
                continue
            normalized.append(
                max(1, len(replica_device_ids[replica_id]) if replica_id < len(replica_device_ids) else 0)
            )
        return tuple(normalized)

    def dag_replica_device_ids(self, replica_id: int) -> tuple[str, ...]:
        if 0 <= replica_id < len(self._dag_replica_device_ids):
            return self._dag_replica_device_ids[replica_id]
        return ()

    def dag_replica_card_count(self, replica_id: int) -> int:
        if 0 <= replica_id < len(self._dag_replica_card_counts):
            return self._dag_replica_card_counts[replica_id]
        return 1

    # ---- Stage-level properties ----

    @property
    def num_replicas(self) -> int:
        """Total slot count, including ``None`` holes from removed replicas.

        Use :meth:`live_replica_ids` to iterate only live entries.
        """
        return len(self.clients)

    @property
    def live_num_replicas(self) -> int:
        """Number of currently live (non-None) replicas in this pool."""
        return sum(1 for c in self.clients if c is not None)

    def live_replica_ids(self) -> list[int]:
        """Return the indices of currently live replicas in this pool."""
        return [i for i, c in enumerate(self.clients) if c is not None]

    @property
    def stage_type(self) -> str | None:
        client = self.stage_client
        return None if client is None else client.stage_type

    @property
    def final_output(self) -> bool:
        client = self.stage_client
        return False if client is None else bool(client.final_output)

    @property
    def stage_client(self) -> StagePoolClient | None:
        for client in self.clients:
            if client is not None:
                return client
        return None

    @property
    def llm_stage_client(self) -> StagePoolLLMClient:
        return cast(StagePoolLLMClient, self.stage_client)

    @property
    def stage_vllm_config(self) -> Any:
        return self._stage_vllm_config

    @property
    def output_processor(self) -> Any:
        return self._output_processor

    @property
    def is_distributed(self) -> bool:
        """True iff a hub has been attached (i.e. running in head-distributed mode)."""
        return self._hub is not None

    # ---- Distributed-mode dispatch hooks ----

    def attach_hub(self, hub: OmniCoordClientForHub | None) -> None:
        """Inject the shared :class:`OmniCoordClientForHub`.

        Called once by :class:`Orchestrator` after the hub is constructed.
        ``hub=None`` keeps the pool in legacy mode (no behavior change).
        """
        self._hub = hub

    def attach_load_balancer(self, lb: LoadBalancer | None) -> None:
        """Inject the per-pool :class:`LoadBalancer` for distributed-mode pick."""
        self._lb = lb

    # ---- Dynamic membership (distributed mode) ----

    @staticmethod
    def _client_input_addr(client: Any) -> str | None:
        """Return the input ZMQ address advertised by ``client`` if any.

        LLM clients expose ``client_addresses["input_address"]``; diffusion
        clients expose ``request_address``. Both are stable strings used by
        :class:`OmniCoordinator` to key replicas.
        """
        request_address = getattr(client, "request_address", None)
        if isinstance(request_address, str) and request_address:
            return request_address
        addrs = getattr(client, "client_addresses", None)
        if isinstance(addrs, dict):
            addr = addrs.get("input_address")
            if isinstance(addr, str) and addr:
                return addr
        return None

    def add_client(self, input_addr: str, client: Any) -> int:
        """Register a head-side client for ``input_addr``.

        Returns the assigned ``replica_id`` (index into :attr:`clients`).
        If the address is already known, replaces the existing client and
        returns its existing id (this should not happen in practice — the
        master server assigns unique slots — but the contract is idempotent
        to keep the dispatch layer robust).
        """
        if not input_addr:
            raise ValueError("input_addr must be a non-empty string")

        existing = self._addr_to_replica_id.get(input_addr)
        if existing is not None:
            self.clients[existing] = client
            return existing

        replica_id = len(self.clients)
        self.clients.append(client)
        self._addr_to_replica_id[input_addr] = replica_id
        self._replica_metrics.append(_ReplicaMetrics())
        return replica_id

    def remove_client(self, input_addr: str) -> Any | None:
        """Remove the client at ``input_addr``. Returns the removed client or ``None``.

        Slot is marked ``None`` to preserve indices for outstanding bindings.
        """
        replica_id = self._addr_to_replica_id.pop(input_addr, None)
        if replica_id is None:
            return None
        client = self.clients[replica_id]
        self.clients[replica_id] = None
        return client

    def get_client_by_addr(self, input_addr: str) -> Any | None:
        """Return the live client for ``input_addr`` if present."""
        replica_id = self._addr_to_replica_id.get(input_addr)
        if replica_id is None:
            return None
        return self.clients[replica_id]

    def get_replica_id_by_addr(self, input_addr: str) -> int | None:
        """Return the stable replica_id for ``input_addr`` if registered."""
        return self._addr_to_replica_id.get(input_addr)

    # ---- Per-request distributed dispatch ----

    async def pick(
        self,
        request_id: str,
        task: Task | None = None,
        *,
        affinity_request_id: str | None = None,
    ) -> int:
        """Return a replica id for ``request_id``.

        In distributed mode: consults the hub for UP replicas, runs the load
        balancer, and records affinity so future picks for the same
        ``request_id`` return the same replica. Bounded wait up to
        ``DISPATCH_WAIT_TIMEOUT_S`` when no UP replica is currently usable.

        In non-distributed (legacy) mode: delegates to
        :meth:`select_replica_id`.
        """
        if self._hub is None or self._lb is None:
            return self.select_replica_id(request_id, affinity_request_id=affinity_request_id)

        # 1. Sticky: previously bound and still serviceable?
        bound_addr = self._affinity.get(request_id)
        if bound_addr is not None:
            replica_id = self._serviceable_replica_id_for_addr(bound_addr)
            if replica_id is not None:
                return replica_id
            # Bound replica is gone or DOWN — fall through to re-select.
            self._affinity.pop(request_id, None)

        # 2. Inherited affinity (CFG companion sharing a parent request_id).
        if affinity_request_id is not None:
            parent_addr = self._affinity.get(affinity_request_id)
            if parent_addr is not None:
                replica_id = self._serviceable_replica_id_for_addr(parent_addr)
                if replica_id is not None:
                    self._affinity[request_id] = parent_addr
                    return replica_id

        # 3. Fresh pick: poll hub + LB with bounded wait.
        task = task or Task(request_id=request_id)
        deadline = _time.monotonic() + self.DISPATCH_WAIT_TIMEOUT_S
        while True:
            candidates = self._collect_serviceable_replicas()
            if candidates:
                slo_idx = await self._select_slo_aware_candidate_index(candidates, task)
                # LB chose an index *into our candidates list*.
                lb_idx = slo_idx if slo_idx is not None else self._lb.select(task, [rep for rep, _ in candidates])
                replica_info, replica_id = candidates[lb_idx]
                self._affinity[request_id] = replica_info.input_addr
                if slo_idx is None and self.stage_type == "diffusion":
                    self._write_stagepool_profile(
                        "distributed_lb_select",
                        task,
                        replica_id,
                        self._distributed_candidate_profile(candidates),
                        _time.time(),
                    )
                return replica_id

            now = _time.monotonic()
            if now >= deadline:
                raise RuntimeError(f"no UP replica for stage {self.stage_id} after {self.DISPATCH_WAIT_TIMEOUT_S:.1f}s")
            await asyncio.sleep(min(self.DISPATCH_RETRY_INTERVAL_S, deadline - now))

    def preselect_replica_id(
        self,
        request_id: str,
        task: Task | None = None,
        *,
        affinity_request_id: str | None = None,
    ) -> int | None:
        """Synchronously pick and bind a replica before request preprocessing.

        The main-thread input preprocessing path cannot await :meth:`pick`, but
        multimodal cache UUID scoping needs to know the same replica that
        :meth:`submit_initial` will later use. In distributed mode this checks
        the hub's cached replica snapshot once and records the selected input
        address in ``_affinity`` so the async submit path reuses the route. If
        no replica is currently serviceable, return ``None`` and let the async
        submit-time router wait without blocking the caller.
        """
        if self._hub is None or self._lb is None:
            return self.select_replica_id(request_id, affinity_request_id=affinity_request_id)

        bound_addr = self._affinity.get(request_id)
        if bound_addr is not None:
            replica_id = self._serviceable_replica_id_for_addr(bound_addr)
            if replica_id is not None:
                return replica_id
            self._affinity.pop(request_id, None)

        if affinity_request_id is not None:
            parent_addr = self._affinity.get(affinity_request_id)
            if parent_addr is not None:
                replica_id = self._serviceable_replica_id_for_addr(parent_addr)
                if replica_id is not None:
                    self._affinity[request_id] = parent_addr
                    return replica_id

        task = task or Task(request_id=request_id)
        candidates = self._collect_serviceable_replicas()
        if not candidates:
            return None

        lb_idx = self._lb.select(task, [rep for rep, _ in candidates])
        replica_info, replica_id = candidates[lb_idx]
        self._affinity[request_id] = replica_info.input_addr
        return replica_id

    def _collect_serviceable_replicas(self) -> list[tuple[ReplicaInfo, int]]:
        """Return list of ``(ReplicaInfo, replica_id)`` for UP, attached replicas."""
        if self._hub is None:
            return []
        snap = self._hub.get_replicas_for_stage(self.stage_id)
        out: list[tuple[ReplicaInfo, int]] = []
        for rep in snap.replicas:
            if rep.status != ReplicaStatus.UP:
                continue
            replica_id = self._addr_to_replica_id.get(rep.input_addr)
            if replica_id is None:
                continue  # Hub knows about it but head-side client not attached yet.
            if self.clients[replica_id] is None:
                continue
            out.append((rep, replica_id))
        return out

    def _serviceable_replica_id_for_addr(self, input_addr: str) -> int | None:
        """Return ``replica_id`` for ``input_addr`` iff currently UP + attached."""
        if self._hub is None:
            return None
        replica_id = self._addr_to_replica_id.get(input_addr)
        if replica_id is None or self.clients[replica_id] is None:
            return None
        snap = self._hub.get_replicas_for_stage(self.stage_id)
        for rep in snap.replicas:
            if rep.input_addr == input_addr and rep.status == ReplicaStatus.UP:
                return replica_id
        return None

    def bind(self, request_id: str, input_addr: str) -> None:
        """Explicitly record affinity (distributed mode)."""
        self._affinity[request_id] = input_addr

    def release(self, request_id: str) -> None:
        """Drop affinity (distributed mode) and legacy binding for ``request_id``."""
        self._affinity.pop(request_id, None)
        self.release_binding(request_id)

    def invalidate_addr(self, input_addr: str) -> list[str]:
        """Drop affinity rows pointing at ``input_addr``; return affected request ids."""
        affected: list[str] = [rid for rid, addr in self._affinity.items() if addr == input_addr]
        for rid in affected:
            self._affinity.pop(rid, None)
        return affected

    # ---- Legacy (non-distributed) route binding ----

    def get_bound_replica_id(self, request_id: str) -> int | None:
        """Return the currently bound replica id for *request_id* if present.

        In distributed mode the binding may have been recorded via
        :meth:`pick`; we honor it transparently here.
        """
        legacy = self._request_bindings.get(request_id)
        if legacy is not None:
            return legacy
        addr = self._affinity.get(request_id)
        if addr is None:
            return None
        return self._addr_to_replica_id.get(addr)

    def get_bound_client(self, request_id: str) -> StagePoolClient | None:
        """Return the currently bound client for *request_id* if present."""
        replica_id = self.get_bound_replica_id(request_id)
        if replica_id is None:
            return None
        return self.clients[replica_id]

    def get_bound_llm_client(self, request_id: str) -> StagePoolLLMClient | None:
        """Return the currently bound LLM client for *request_id* if present."""
        client = self.get_bound_client(request_id)
        if client is None:
            return None
        return cast(StagePoolLLMClient, client)

    def release_binding(self, request_id: str) -> None:
        """Drop the route binding for *request_id* in this stage."""
        self._request_bindings.pop(request_id, None)
        self._affinity.pop(request_id, None)

    def release_bindings(self, request_ids: list[str]) -> None:
        """Drop route bindings for the given request ids in this stage."""
        for request_id in request_ids:
            self.release_binding(request_id)

    def select_replica_id(
        self,
        request_id: str,
        *,
        affinity_request_id: str | None = None,
    ) -> int:
        """Pick a replica id for *request_id* and cache the choice (legacy path)."""
        cached = self.get_bound_replica_id(request_id)
        if cached is not None and self.clients[cached] is not None:
            return cached

        chosen: int | None = None
        if affinity_request_id is not None:
            parent = self.get_bound_replica_id(affinity_request_id)
            if parent is not None and self.clients[parent] is not None:
                chosen = parent

        if chosen is None:
            live = self.live_replica_ids()
            if not live:
                raise RuntimeError(f"stage {self.stage_id} has no live replicas")
            if len(live) == 1:
                chosen = live[0]
            else:
                # Round-robin over live replicas only.
                start = self._next_replica_id % len(live)
                chosen = live[start]
                self._next_replica_id = (self._next_replica_id + 1) % len(live)

        self._request_bindings[request_id] = chosen
        return chosen

    def _llm_client(self, replica_id: int) -> StagePoolLLMClient:
        client = self.clients[replica_id]
        if client is None:
            raise RuntimeError(f"stage {self.stage_id} replica {replica_id} is not attached")
        return cast(StagePoolLLMClient, client)

    def _diffusion_client(self, replica_id: int) -> StagePoolDiffusionClient:
        client = self.clients[replica_id]
        if client is None:
            raise RuntimeError(f"stage {self.stage_id} replica {replica_id} is not attached")
        return cast(StagePoolDiffusionClient, client)

    # ---- Metrics ----

    def build_stage_metrics(
        self,
        request_outputs: list[Any],
        *,
        submit_ts: float,
        replica_id: int,
    ) -> StageRequestMetrics:
        """Build stage metrics for outputs produced on one replica."""
        now = _time.time()
        stage_gen_time_ms = (now - submit_ts) * 1000.0

        num_tokens_out = count_tokens_from_outputs(request_outputs)
        num_tokens_in = 0
        if self.stage_id == 0:
            for ro in request_outputs:
                ptids = getattr(ro, "prompt_token_ids", None)
                if ptids is not None:
                    num_tokens_in += len(ptids)

        metrics = self._replica_metrics[replica_id]
        metrics.batch_seq += 1
        batch_id = metrics.batch_seq
        metrics.agg_total_tokens += num_tokens_out
        metrics.agg_total_gen_time_ms += stage_gen_time_ms

        return StageRequestMetrics(
            num_tokens_in=num_tokens_in,
            num_tokens_out=num_tokens_out,
            stage_gen_time_ms=stage_gen_time_ms,
            batch_id=batch_id,
            batch_size=1,
            rx_decode_time_ms=0.0,
            rx_transfer_bytes=0,
            rx_in_flight_time_ms=0.0,
            stage_stats=StageStats(
                total_token=metrics.agg_total_tokens,
                total_gen_time_ms=metrics.agg_total_gen_time_ms,
            ),
        )

    # ---- Stage-local admission ----

    async def submit_initial(
        self,
        request_id: str,
        req_state: OrchestratorRequestState,
        request: Any,
        *,
        prompt_text: Any = None,
        affinity_request_id: str | None = None,
        submit_kwargs: dict[str, Any] | None = None,
        params_override: Any = None,
    ) -> int:
        """Submit a stage-entry request into this pool."""
        params = params_override if params_override is not None else req_state.sampling_params_list[self.stage_id]
        submit_kwargs = dict(submit_kwargs or {})
        if self.stage_type == "diffusion":
            task: Task = {
                "request_id": request_id,
                "sampling_params": params,
                "prompt_count": _prompt_count_from_request(request),
                "has_negative_prompt": _has_negative_prompt(request),
            }
            replica_id = await self._pick_or_select(
                request_id,
                task=task,
                affinity_request_id=affinity_request_id,
            )
            client = self._diffusion_client(replica_id)
            if isinstance(request, list):
                await client.add_batch_request_async(request_id, request, params, **submit_kwargs)
            else:
                await client.add_request_async(request_id, request, params, **submit_kwargs)
            return replica_id

        replica_id = await self._pick_or_select(
            request_id,
            affinity_request_id=affinity_request_id,
        )
        client = self.clients[replica_id]
        if client is None:
            raise RuntimeError(f"stage {self.stage_id} replica {replica_id} is not attached")
        try:
            self.output_processor.add_request(
                request=request,
                prompt=prompt_text,
                parent_req=None,
                request_index=0,
                queue=None,
            )
        except Exception:
            self.release_binding(request_id)
            raise

        try:
            await self._llm_client(replica_id).add_request_async(request, **submit_kwargs)
        except Exception:
            self.release_binding(request_id)
            rollback = getattr(self.output_processor, "remove_request", None)
            if callable(rollback):
                try:
                    rollback(request_id)
                except Exception as rollback_error:
                    logger.warning(
                        "[StagePool] Failed to rollback output processor state for req=%s stage-%s: %s",
                        request_id,
                        self.stage_id,
                        rollback_error,
                    )
            raise
        return replica_id

    async def submit_update(
        self,
        request_id: str,
        req_state: OrchestratorRequestState,
        request: Any,
        *,
        prompt_text: Any = None,
    ) -> int:
        """Submit a streaming update to an already admitted request."""
        params = req_state.sampling_params_list[self.stage_id]
        replica_id = self.get_bound_replica_id(request_id)
        if replica_id is None or self.clients[replica_id] is None:
            replica_id = await self._pick_or_select(request_id)

        client = self.clients[replica_id]
        if client is None:
            raise RuntimeError(f"stage {self.stage_id} replica {replica_id} is not attached")

        if self.stage_type == "diffusion":
            await self._diffusion_client(replica_id).add_request_async(request_id, request, params)
        else:
            # Refresh the shared output-processor state before yielding to the
            # stage client so streaming segments are merged against the latest
            # prompt/token metadata.
            self.output_processor.add_request(
                request=request,
                prompt=prompt_text,
                parent_req=None,
                request_index=0,
                queue=None,
            )
            await self._llm_client(replica_id).add_request_async(request)
        return replica_id

    async def _pick_or_select(
        self,
        request_id: str,
        *,
        task: Task | None = None,
        affinity_request_id: str | None = None,
    ) -> int:
        """Bridge to ``pick`` in distributed mode or ``select_replica_id`` legacy."""
        if self.is_distributed:
            return await self.pick(request_id, task=task, affinity_request_id=affinity_request_id)
        if self.stage_type == "diffusion" and affinity_request_id is None and task is not None:
            replica_id = await self._select_slo_aware_local_replica_id(task)
            if replica_id is not None:
                self._request_bindings[request_id] = replica_id
                return replica_id
            replica_id = self.select_replica_id(request_id, affinity_request_id=affinity_request_id)
            self._write_stagepool_profile(
                "local_lb_select",
                task,
                replica_id,
                self._local_candidate_profile(),
                _time.time(),
            )
            return replica_id
        return self.select_replica_id(request_id, affinity_request_id=affinity_request_id)

    async def _select_slo_aware_candidate_index(
        self,
        candidates: list[tuple[ReplicaInfo, int]],
        task: Task | None,
    ) -> int | None:
        if not self._enable_stagepool_slo or not self._task_has_deadline(task) or len(candidates) <= 1:
            return None

        snapshots = await asyncio.gather(
            *(self._get_scheduler_snapshot(replica_id) for _, replica_id in candidates),
            return_exceptions=True,
        )
        scored: list[dict[str, Any]] = []
        profile_candidates: list[dict[str, Any]] = []
        cursor = self._slo_tie_break_cursor % len(candidates)
        now_s = _time.time()
        sampling_params = None if task is None else task.get("sampling_params")
        incoming_key = self._task_sampling_key_dict(task)
        for idx, ((replica_info, _), snapshot) in enumerate(zip(candidates, snapshots, strict=False)):
            if isinstance(snapshot, BaseException) or snapshot is None:
                continue
            tie_rank = (idx - cursor) % len(candidates)
            candidate = self._stagepool_slo_candidate_profile(
                task,
                snapshot,
                incoming_key=incoming_key,
                now_s=now_s,
                candidate_index=idx,
                replica_id=candidates[idx][1],
                tie_rank=tie_rank,
                fallback_queue_length=replica_info.queue_length,
                input_addr=replica_info.input_addr,
            )
            scored.append(candidate)
            profile_candidates.append(candidate)

        if not scored:
            self._write_stagepool_profile("distributed_slo_select", task, None, profile_candidates, now_s)
            return None
        selected = self._choose_stagepool_slo_candidate(scored)
        selected_idx = int(selected["candidate_index"])
        self._slo_tie_break_cursor = (selected_idx + 1) % len(candidates)
        self._scheduler_snapshot_cache.pop(candidates[selected_idx][1], None)
        self._write_stagepool_profile(
            "distributed_slo_select",
            task,
            candidates[selected_idx][1],
            profile_candidates,
            now_s,
        )
        return selected_idx

    async def _select_slo_aware_local_replica_id(self, task: Task) -> int | None:
        live = self.live_replica_ids()
        if not self._enable_stagepool_slo or not self._task_has_deadline(task) or len(live) <= 1:
            return None

        snapshots = await asyncio.gather(
            *(self._get_scheduler_snapshot(replica_id) for replica_id in live),
            return_exceptions=True,
        )
        scored: list[dict[str, Any]] = []
        profile_candidates: list[dict[str, Any]] = []
        cursor = self._slo_tie_break_cursor % len(live)
        now_s = _time.time()
        sampling_params = None if task is None else task.get("sampling_params")
        incoming_key = self._task_sampling_key_dict(task)
        for idx, (replica_id, snapshot) in enumerate(zip(live, snapshots, strict=False)):
            if isinstance(snapshot, BaseException) or snapshot is None:
                continue
            tie_rank = (idx - cursor) % len(live)
            candidate = self._stagepool_slo_candidate_profile(
                task,
                snapshot,
                incoming_key=incoming_key,
                now_s=now_s,
                candidate_index=idx,
                replica_id=replica_id,
                tie_rank=tie_rank,
                fallback_queue_length=0,
            )
            scored.append(candidate)
            profile_candidates.append(candidate)

        if not scored:
            self._write_stagepool_profile("local_slo_select", task, None, profile_candidates, now_s)
            return None
        selected = self._choose_stagepool_slo_candidate(scored)
        selected_replica_id = int(selected["replica_id"])
        selected_live_idx = live.index(selected_replica_id)
        self._slo_tie_break_cursor = (selected_live_idx + 1) % len(live)
        self._scheduler_snapshot_cache.pop(selected_replica_id, None)
        self._write_stagepool_profile("local_slo_select", task, selected_replica_id, profile_candidates, now_s)
        return selected_replica_id

    def _stagepool_slo_candidate_profile(
        self,
        task: Task | None,
        snapshot: dict[str, Any],
        *,
        incoming_key: dict[str, Any] | None,
        now_s: float,
        candidate_index: int,
        replica_id: int,
        tie_rank: int,
        fallback_queue_length: int,
        input_addr: str | None = None,
    ) -> dict[str, Any]:
        predicted_laxity_ms = self._predict_task_laxity_ms(task, snapshot, now_s=now_s)
        safe_capacity = int(snapshot.get("safe_admit_capacity", 0) or 0)
        queue_length = int(snapshot.get("num_waiting", fallback_queue_length) or 0) + int(
            snapshot.get("num_running", 0) or 0
        )
        sampling_params = None if task is None else task.get("sampling_params")
        matching_bucket = self._matching_bucket(snapshot, incoming_key)
        matching_bucket_size = (
            int(matching_bucket.get("candidate_batch_size", 0) or 0) if matching_bucket is not None else 0
        )
        matching_bucket_laxity_ms = (
            _optional_float(matching_bucket.get("min_laxity_ms")) if matching_bucket is not None else None
        )
        pack_queue_limit_exceeded = (
            self._slo_stagepool_pack_max_queue_length > 0
            and queue_length >= self._slo_stagepool_pack_max_queue_length
        )
        pack_bucket_limit_exceeded = (
            self._slo_stagepool_pack_max_matching_bucket_size > 0
            and matching_bucket_size > 0
            and matching_bucket_size >= self._slo_stagepool_pack_max_matching_bucket_size
        )
        can_pack_same_key = (
            safe_capacity > 0
            and matching_bucket_size > 0
            and not pack_queue_limit_exceeded
            and not pack_bucket_limit_exceeded
            and predicted_laxity_ms >= self._slo_stagepool_pack_min_laxity_ms
            and (
                matching_bucket_laxity_ms is None
                or matching_bucket_laxity_ms >= self._slo_stagepool_pack_min_laxity_ms
            )
        )
        token_pressure = self._snapshot_token_pressure(snapshot)
        matching_token_utilization = self._matching_bucket_token_utilization(matching_bucket)
        incoming_latent_tokens = self._incoming_latent_tokens(sampling_params) if sampling_params is not None else 0.0
        incoming_effective_size = _task_effective_size(task)
        incoming_remaining_steps = (
            self._task_remaining_steps(sampling_params) if sampling_params is not None else 0
        )
        incoming_token_work = incoming_latent_tokens * incoming_effective_size * max(float(incoming_remaining_steps), 1.0)
        resident_token_work, waiting_token_work = self._snapshot_token_work_breakdown(snapshot)
        matching_bucket_token_work = self._bucket_token_work(matching_bucket)
        token_pressure_after = token_pressure + incoming_token_work
        candidate_min_laxity_ratio = (
            _optional_float(matching_bucket.get("min_laxity_ratio")) if matching_bucket is not None else None
        )
        incoming_laxity_ms = self._predict_incoming_laxity_ms(task, snapshot, now_s=now_s)
        existing_min_laxity_after_ms = self._estimate_existing_laxity_after_ms(task, snapshot)
        admission_reject_reason = None
        if predicted_laxity_ms < self._slo_stagepool_profile_reject_laxity_ms:
            admission_reject_reason = "predicted_laxity_below_profile_threshold"
        candidate = {
            "candidate_index": candidate_index,
            "replica_id": replica_id,
            "predicted_laxity_ms": predicted_laxity_ms,
            "safe_capacity": safe_capacity,
            "queue_length": queue_length,
            "tie_rank": tie_rank,
            "snapshot_policy": snapshot.get("policy"),
            "matching_bucket_size": matching_bucket_size,
            "matching_bucket_min_laxity_ms": matching_bucket_laxity_ms,
            "can_pack_same_key": can_pack_same_key,
            "pack_queue_limit_exceeded": pack_queue_limit_exceeded,
            "pack_bucket_limit_exceeded": pack_bucket_limit_exceeded,
            "token_pressure": token_pressure,
            "matching_bucket_token_utilization": matching_token_utilization,
            "incoming_latent_tokens": incoming_latent_tokens,
            "incoming_token_work": incoming_token_work,
            "incoming_remaining_steps": incoming_remaining_steps,
            "incoming_effective_size": incoming_effective_size,
            "token_pressure_before": token_pressure,
            "token_pressure_after": token_pressure_after,
            "resident_token_work": resident_token_work,
            "waiting_token_work": waiting_token_work,
            "matching_bucket_token_work": matching_bucket_token_work,
            "candidate_min_laxity_ratio": candidate_min_laxity_ratio,
            "existing_min_laxity_after_ms": existing_min_laxity_after_ms,
            "incoming_laxity_ms": incoming_laxity_ms,
            "admission_decision": "profile_only",
            "admission_reject_reason": admission_reject_reason,
        }
        if self._uses_stagepool_objective():
            self._annotate_stagepool_objective(candidate)
        if input_addr is not None:
            candidate["input_addr"] = input_addr
        return candidate

    def _choose_stagepool_slo_candidate(self, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if self._uses_stagepool_objective():
            for candidate in candidates:
                self._annotate_stagepool_objective(candidate)

            def objective_first(candidate: dict[str, Any]) -> tuple[float, float, int, int]:
                return (
                    float(candidate["stagepool_objective_score"]),
                    -float(candidate["predicted_laxity_ms"]),
                    int(candidate["tie_rank"]),
                    int(candidate["candidate_index"]),
                )

            return min(candidates, key=objective_first)

        def laxity_first(candidate: dict[str, Any]) -> tuple[float, int, int, int, int]:
            return (
                -float(candidate["predicted_laxity_ms"]),
                -int(candidate["safe_capacity"]),
                int(candidate["queue_length"]),
                int(candidate["tie_rank"]),
                int(candidate["candidate_index"]),
            )

        best = min(candidates, key=laxity_first)
        if self._slo_stagepool_laxity_window_ms <= 0:
            return best

        best_laxity_ms = float(best["predicted_laxity_ms"])
        if best_laxity_ms < self._slo_stagepool_pack_min_laxity_ms:
            return best

        laxity_floor_ms = max(
            self._slo_stagepool_pack_min_laxity_ms,
            best_laxity_ms - self._slo_stagepool_laxity_window_ms,
        )
        eligible = [
            candidate
            for candidate in candidates
            if float(candidate["predicted_laxity_ms"]) >= laxity_floor_ms
        ]
        if not any(bool(candidate["can_pack_same_key"]) for candidate in eligible):
            return best

        def packing_first(candidate: dict[str, Any]) -> tuple[int, int, float, int, int, int, int]:
            return (
                -int(bool(candidate["can_pack_same_key"])),
                -int(candidate["matching_bucket_size"]),
                -float(candidate["predicted_laxity_ms"]),
                -int(candidate["safe_capacity"]),
                int(candidate["queue_length"]),
                int(candidate["tie_rank"]),
                int(candidate["candidate_index"]),
            )

        return min(eligible, key=packing_first)

    def _uses_stagepool_objective(self) -> bool:
        return self._slo_stagepool_selection_objective != "legacy"

    def _annotate_stagepool_objective(self, candidate: dict[str, Any]) -> None:
        predicted_laxity_ms = _optional_float(candidate.get("predicted_laxity_ms")) or 0.0
        incoming_laxity_ms = _optional_float(candidate.get("incoming_laxity_ms"))
        existing_laxity_after_ms = _optional_float(candidate.get("existing_min_laxity_after_ms"))
        miss_risk_ms = max(-predicted_laxity_ms, 0.0)
        if incoming_laxity_ms is not None:
            miss_risk_ms = max(miss_risk_ms, -incoming_laxity_ms)
        if existing_laxity_after_ms is not None:
            miss_risk_ms = max(miss_risk_ms, -existing_laxity_after_ms)
        token_pressure_units = max(
            float(candidate.get("token_pressure_after", candidate.get("token_pressure", 0.0)) or 0.0) / 1024.0,
            0.0,
        )
        queue_cost = max(float(candidate.get("queue_length", 0) or 0), 0.0)
        pack_bonus = (
            max(float(candidate.get("matching_bucket_size", 0) or 0), 0.0)
            if bool(candidate.get("can_pack_same_key"))
            else 0.0
        )
        safe_capacity_bonus = max(float(candidate.get("safe_capacity", 0) or 0), 0.0)
        score = (
            self._slo_stagepool_objective_miss_risk_weight * miss_risk_ms
            + self._slo_stagepool_objective_token_pressure_weight * token_pressure_units
            + self._slo_stagepool_objective_queue_weight * queue_cost
            - self._slo_stagepool_objective_pack_bonus * pack_bonus
            - self._slo_stagepool_objective_safe_capacity_bonus * safe_capacity_bonus
        )
        candidate["stagepool_selection_objective"] = self._slo_stagepool_selection_objective
        candidate["stagepool_objective_score"] = score
        candidate["stagepool_objective_miss_risk_ms"] = miss_risk_ms
        candidate["stagepool_objective_token_pressure_units"] = token_pressure_units
        candidate["stagepool_objective_queue_cost"] = queue_cost
        candidate["stagepool_objective_pack_bonus"] = pack_bonus
        candidate["stagepool_objective_safe_capacity_bonus"] = safe_capacity_bonus

    def _distributed_candidate_profile(self, candidates: list[tuple[ReplicaInfo, int]]) -> list[dict[str, Any]]:
        return [
            {
                "candidate_index": idx,
                "replica_id": replica_id,
                "input_addr": replica_info.input_addr,
                "queue_length": replica_info.queue_length,
                "status": getattr(replica_info.status, "name", str(replica_info.status)),
            }
            for idx, (replica_info, replica_id) in enumerate(candidates)
        ]

    def _local_candidate_profile(self) -> list[dict[str, Any]]:
        return [
            {
                "candidate_index": idx,
                "replica_id": replica_id,
            }
            for idx, replica_id in enumerate(self.live_replica_ids())
        ]

    async def _get_scheduler_snapshot(self, replica_id: int) -> dict[str, Any] | None:
        now_s = _time.time()
        cached = self._scheduler_snapshot_cache.get(replica_id)
        if cached is not None:
            cached_at_s, snapshot = cached
            if now_s - cached_at_s <= self._scheduler_snapshot_ttl_s:
                return snapshot

        if replica_id >= len(self.clients):
            return None
        client = self.clients[replica_id]
        if client is None:
            return None
        rpc = getattr(client, "collective_rpc_async", None)
        if not callable(rpc):
            return None

        try:
            snapshot = await rpc(
                "get_scheduler_load_snapshot",
                timeout=self.SCHEDULER_SNAPSHOT_RPC_TIMEOUT_S,
            )
        except Exception:
            return None

        if not self._is_valid_scheduler_snapshot(snapshot):
            return None
        if self._scheduler_snapshot_ttl_s > 0:
            self._scheduler_snapshot_cache[replica_id] = (now_s, snapshot)
        else:
            self._scheduler_snapshot_cache.pop(replica_id, None)
        return snapshot

    @staticmethod
    def _is_valid_scheduler_snapshot(snapshot: Any) -> bool:
        if not isinstance(snapshot, dict):
            return False
        if snapshot.get("supported") is False:
            return False
        if not isinstance(snapshot.get("timestamp_s"), (int, float)):
            return False
        if snapshot.get("policy") not in {
            "SloStepScheduler",
            "StepScheduler",
            "TokenSloStepScheduler",
            "AdaptiveTokenSloStepScheduler",
            "TokenStepPreemptiveSloStepScheduler",
        }:
            return False
        buckets = snapshot.get("buckets", [])
        if not isinstance(buckets, list):
            return False
        if not buckets:
            return int(snapshot.get("safe_admit_capacity", 0) or 0) > 0
        for bucket in buckets:
            if not isinstance(bucket, dict):
                return False
            if not isinstance(bucket.get("estimated_step_ms"), (int, float)):
                return False
            if not isinstance(bucket.get("min_remaining_steps"), (int, float)):
                return False
        return True

    def _predict_task_laxity_ms(
        self,
        task: Task | None,
        snapshot: dict[str, Any],
        *,
        now_s: float | None = None,
    ) -> float:
        deadline_s = self._task_deadline_time_s(task)
        if deadline_s is None:
            return 0.0

        now_s = _time.time() if now_s is None else now_s
        sampling_params = None if task is None else task.get("sampling_params")
        incoming_key = self._task_sampling_key_dict(task)
        incoming_cost_ms = self._estimate_task_remaining_cost_ms(task, snapshot)
        admit_delay_ms = self._estimate_admit_delay_ms(
            snapshot,
            incoming_key,
            incoming_effective_size=_task_effective_size(task),
        )
        incoming_laxity_ms = (deadline_s - now_s) * 1000.0 - admit_delay_ms - incoming_cost_ms
        existing_laxity_after_ms = self._estimate_existing_laxity_after_ms(task, snapshot)
        if existing_laxity_after_ms is None:
            return incoming_laxity_ms
        return min(incoming_laxity_ms, existing_laxity_after_ms)

    def _predict_incoming_laxity_ms(
        self,
        task: Task | None,
        snapshot: dict[str, Any],
        *,
        now_s: float | None = None,
    ) -> float:
        deadline_s = self._task_deadline_time_s(task)
        if deadline_s is None:
            return 0.0

        now_s = _time.time() if now_s is None else now_s
        incoming_key = self._task_sampling_key_dict(task)
        incoming_cost_ms = self._estimate_task_remaining_cost_ms(task, snapshot)
        admit_delay_ms = self._estimate_admit_delay_ms(
            snapshot,
            incoming_key,
            incoming_effective_size=_task_effective_size(task),
        )
        return (deadline_s - now_s) * 1000.0 - admit_delay_ms - incoming_cost_ms

    def _estimate_admit_delay_ms(
        self,
        snapshot: dict[str, Any],
        incoming_key: dict[str, Any] | None,
        *,
        incoming_effective_size: float = 1.0,
    ) -> float:
        safe_capacity = int(snapshot.get("safe_admit_capacity", 0) or 0)
        no_preemption_snapshot = self._snapshot_disables_step_preemption(snapshot)
        if safe_capacity > 0 and int(snapshot.get("num_waiting", 0) or 0) <= 0:
            if int(snapshot.get("num_running", 0) or 0) <= 0:
                return 0.0
            buckets = snapshot.get("buckets")
            if isinstance(buckets, list) and buckets:
                same_key_steps = [
                    float(bucket.get("estimated_step_ms", 0.0) or 0.0)
                    for bucket in buckets
                    if isinstance(bucket, dict)
                    and incoming_key is not None
                    and bucket.get("key") == incoming_key
                ]
                same_key_steps = [value for value in same_key_steps if value > 0]
                if same_key_steps:
                    return min(same_key_steps)
                tail_candidates = []
                for bucket in buckets:
                    if not isinstance(bucket, dict):
                        continue
                    estimated_step_ms = float(bucket.get("estimated_step_ms", 0.0) or 0.0)
                    max_remaining_steps = float(
                        bucket.get("max_remaining_steps", bucket.get("min_remaining_steps", 1.0)) or 1.0
                    )
                    if estimated_step_ms > 0:
                        tail_candidates.append(estimated_step_ms * max(max_remaining_steps, 1.0))
                if tail_candidates:
                    return min(tail_candidates)
            return float(snapshot.get("num_running", 0) or 0)

        buckets = snapshot.get("buckets")
        if not isinstance(buckets, list) or not buckets:
            return float(snapshot.get("num_running", 0) or 0)

        candidates: list[float] = []
        for bucket in buckets:
            if not isinstance(bucket, dict):
                continue
            estimated_step_ms = float(bucket.get("estimated_step_ms", 0.0) or 0.0)
            remaining_steps_key = "max_remaining_steps" if no_preemption_snapshot else "min_remaining_steps"
            remaining_steps = float(
                bucket.get(remaining_steps_key, bucket.get("min_remaining_steps", 1.0)) or 1.0
            )
            delay_ms = max(estimated_step_ms * max(remaining_steps, 1.0), 0.0)
            if bucket.get("key") == incoming_key:
                # Same-key requests can be admitted at a step boundary. If
                # capacity is available but older waiting requests exist, charge
                # one representative step instead of the whole bucket tail.
                if safe_capacity > 0 and not no_preemption_snapshot:
                    delay_ms = min(delay_ms, estimated_step_ms)
                elif safe_capacity <= 0 and not no_preemption_snapshot:
                    delay_ms *= 0.95
            candidates.append(delay_ms)
        return min(candidates) if candidates else 0.0

    @staticmethod
    def _snapshot_disables_step_preemption(snapshot: dict[str, Any]) -> bool:
        enable_step_preemption = snapshot.get("enable_step_preemption")
        if isinstance(enable_step_preemption, bool):
            return not enable_step_preemption
        return snapshot.get("policy") in {
            "TokenSloStepScheduler",
            "AdaptiveTokenSloStepScheduler",
        }

    def _estimate_task_remaining_cost_ms(self, task: Task | None, snapshot: dict[str, Any]) -> float:
        sampling_params = None if task is None else task.get("sampling_params")
        if sampling_params is None:
            return 0.0

        remaining_steps = self._task_remaining_steps(sampling_params)
        step_estimate = self._estimate_incoming_step(
            sampling_params,
            snapshot,
            prompt_count=_task_prompt_count(task),
            has_negative_prompt=bool(task.get("has_negative_prompt")),
        )
        if step_estimate is not None and step_estimate.source != "default":
            return remaining_steps * step_estimate.step_ms + self._slo_decode_ms

        reference_cost_ms = _optional_float(_sampling_get(sampling_params, "reference_cost_ms"))
        if reference_cost_ms is not None and not self._slo_ignore_reference_cost:
            return reference_cost_ms
        slo_ms = _optional_float(_sampling_get(sampling_params, "slo_ms"))
        if slo_ms is not None and not self._slo_ignore_reference_cost:
            return slo_ms / 3.0
        return remaining_steps * self._slo_default_step_ms + self._slo_decode_ms

    def _estimate_incoming_step(
        self,
        sampling_params: Any,
        snapshot: dict[str, Any],
        *,
        prompt_count: float = 1.0,
        has_negative_prompt: bool = False,
    ) -> Any | None:
        if self._slo_cost_model is None:
            return None
        width, height = _sampling_dimensions(sampling_params)
        frames = _sampling_get(sampling_params, "num_frames", 1)
        incoming_eff = max(prompt_count, 1.0) * estimate_request_effective_size(
            sampling_params,
            has_negative_prompt=has_negative_prompt,
        )
        effective_batch_size = incoming_eff
        incoming_tokens = self._incoming_latent_tokens(sampling_params)
        max_tokens = incoming_tokens
        batch_size = 1
        incoming_key = self._sampling_key_dict(
            sampling_params,
            dynamic_qwen=self._uses_qwen_image_dynamic_step_batching(),
            has_negative_prompt=has_negative_prompt,
        )
        bucket = self._matching_bucket(snapshot, incoming_key)
        if bucket is not None:
            bucket_tokens = _optional_float(bucket.get("max_latent_tokens"))
            if bucket_tokens is not None and bucket_tokens > 0:
                max_tokens = max(max_tokens, bucket_tokens)
                existing_work = _optional_float(bucket.get("total_token_work"))
                bucket_shape = bucket.get("shape")
                if bucket_tokens >= incoming_tokens and isinstance(bucket_shape, dict):
                    bucket_width = _optional_float(bucket_shape.get("width"))
                    bucket_height = _optional_float(bucket_shape.get("height"))
                    bucket_frames = _optional_float(bucket_shape.get("num_frames"))
                    if bucket_width is not None and bucket_height is not None:
                        width = bucket_width
                        height = bucket_height
                        frames = bucket_frames or frames
                if existing_work is not None and max_tokens > 0:
                    effective_batch_size = (existing_work + incoming_tokens * incoming_eff) / max_tokens
                else:
                    effective_batch_size += float(bucket.get("effective_batch_size", bucket.get("candidate_batch_size", 0)) or 0)
            else:
                effective_batch_size += float(bucket.get("effective_batch_size", bucket.get("candidate_batch_size", 0)) or 0)
            batch_size += int(bucket.get("candidate_batch_size", 0) or 0)
        return self._slo_cost_model.estimate(
            model=_stage_model_name(self._stage_slo_config),
            width=width,
            height=height,
            num_frames=frames,
            batch_size=batch_size,
            effective_batch_size=effective_batch_size,
        )

    def _estimate_existing_laxity_after_ms(
        self,
        task: Task | None,
        snapshot: dict[str, Any],
    ) -> float | None:
        sampling_params = None if task is None else task.get("sampling_params")
        if sampling_params is None:
            return None
        incoming_key = self._task_sampling_key_dict(task)
        bucket = self._matching_bucket(snapshot, incoming_key)
        if bucket is None:
            return None
        min_laxity_ms = _optional_float(bucket.get("min_laxity_ms"))
        if min_laxity_ms is None:
            return None
        current_step_ms = _optional_float(bucket.get("estimated_step_ms"))
        if current_step_ms is None:
            return None
        shape = bucket.get("shape")
        width = height = frames = None
        if isinstance(shape, dict):
            width = _optional_float(shape.get("width"))
            height = _optional_float(shape.get("height"))
            frames = _optional_float(shape.get("num_frames"))
        if width is None or height is None:
            width, height = _dimensions_from_key(incoming_key)
        incoming_eff = _task_effective_size(task)
        current_eff = float(bucket.get("effective_batch_size", bucket.get("candidate_batch_size", 0)) or 0)
        effective_batch_size_after = current_eff + incoming_eff
        incoming_tokens = self._incoming_latent_tokens(sampling_params)
        bucket_tokens = _optional_float(bucket.get("max_latent_tokens"))
        existing_work = _optional_float(bucket.get("total_token_work"))
        if bucket_tokens is not None and bucket_tokens > 0 and existing_work is not None:
            max_tokens = max(bucket_tokens, incoming_tokens)
            effective_batch_size_after = (existing_work + incoming_tokens * incoming_eff) / max_tokens
            if incoming_tokens > bucket_tokens:
                incoming_width, incoming_height = _sampling_dimensions(sampling_params)
                if incoming_width is not None and incoming_height is not None:
                    width = incoming_width
                    height = incoming_height
                    frames = _optional_float(_sampling_get(sampling_params, "num_frames")) or frames
        extra_per_step_ms = None
        if self._slo_cost_model is not None:
            estimate = self._slo_cost_model.estimate(
                model=_stage_model_name(self._stage_slo_config),
                width=width,
                height=height,
                num_frames=frames or _sampling_get(sampling_params, "num_frames", 1),
                batch_size=int(bucket.get("candidate_batch_size", 0) or 0) + 1,
                effective_batch_size=effective_batch_size_after,
            )
            if estimate.source != "default":
                extra_per_step_ms = max(estimate.step_ms - current_step_ms, 0.0)
        if extra_per_step_ms is None:
            extra_per_step_ms = _snapshot_token_incremental_step_ms(
                bucket,
                incoming_tokens,
                incoming_eff,
                current_step_ms,
                self._slo_batch_growth_alpha,
            )
        if extra_per_step_ms is None:
            extra_per_step_ms = _snapshot_incremental_step_ms(bucket, incoming_eff, current_step_ms)
        if extra_per_step_ms is None:
            return None
        remaining_steps_key = "max_remaining_steps" if self._snapshot_disables_step_preemption(snapshot) else "min_remaining_steps"
        remaining = float(bucket.get(remaining_steps_key, bucket.get("min_remaining_steps", 1.0)) or 1.0)
        return min_laxity_ms - extra_per_step_ms * max(remaining, 1.0)

    @staticmethod
    def _matching_bucket(snapshot: dict[str, Any], incoming_key: dict[str, Any] | None) -> dict[str, Any] | None:
        if incoming_key is None:
            return None
        buckets = snapshot.get("buckets")
        if not isinstance(buckets, list):
            return None
        for bucket in buckets:
            if isinstance(bucket, dict) and bucket.get("key") == incoming_key:
                return bucket
        return None

    def _task_remaining_steps(self, sampling_params: Any) -> int:
        total_steps = _optional_float(_sampling_get(sampling_params, "num_inference_steps"))
        if total_steps is None:
            total_steps = float(self._slo_default_num_inference_steps)
        step_index = _optional_float(_sampling_get(sampling_params, "step_index")) or 0.0
        return max(int(total_steps - step_index), 1)

    def _task_has_deadline(self, task: Task | None) -> bool:
        return self._task_deadline_time_s(task) is not None

    def _task_deadline_time_s(self, task: Task | None) -> float | None:
        if task is None:
            return None
        sampling_params = task.get("sampling_params")
        if sampling_params is None:
            return None
        deadline_time_s = _optional_float(_sampling_get(sampling_params, "deadline_time_s"))
        if deadline_time_s is not None:
            return deadline_time_s
        arrival_time_s = _optional_float(_sampling_get(sampling_params, "arrival_time_s")) or _time.time()
        slo_ms = _optional_float(_sampling_get(sampling_params, "slo_ms"))
        if slo_ms is not None:
            return arrival_time_s + slo_ms / 1000.0
        reference_cost_ms = _optional_float(_sampling_get(sampling_params, "reference_cost_ms"))
        if reference_cost_ms is not None:
            return arrival_time_s + (3.0 * reference_cost_ms) / 1000.0
        return None

    def _write_stagepool_profile(
        self,
        event: str,
        task: Task | None,
        selected_replica_id: int | None,
        candidates: list[dict[str, Any]],
        timestamp_s: float,
    ) -> None:
        if not self._stagepool_profile_enabled or not self._stagepool_profile_path:
            return
        sampling_params = None if task is None else task.get("sampling_params")
        arrival_time_s = None if sampling_params is None else _optional_float(_sampling_get(sampling_params, "arrival_time_s"))
        deadline_time_s = None if sampling_params is None else _optional_float(
            _sampling_get(sampling_params, "deadline_time_s")
        )
        reference_cost_ms = None if sampling_params is None else _optional_float(
            _sampling_get(sampling_params, "reference_cost_ms")
        )
        slo_ms = None if sampling_params is None else _optional_float(_sampling_get(sampling_params, "slo_ms"))
        raw_client_request_id = None if sampling_params is None else _sampling_get(sampling_params, "client_request_id")
        client_request_id = None if raw_client_request_id is None else str(raw_client_request_id)
        profile_only_reject_reason = self._profile_only_reject_reason(candidates)
        if profile_only_reject_reason is not None:
            for candidate in candidates:
                candidate.setdefault("profile_only_reject_reason", profile_only_reject_reason)
        record = {
            "timestamp_s": timestamp_s,
            "event": event,
            "stage_id": self.stage_id,
            "stage_type": self.stage_type,
            "request_id": None if task is None else task.get("request_id"),
            "client_request_id": client_request_id,
            "selected_replica_id": selected_replica_id,
            "arrival_time_s": arrival_time_s,
            "deadline_time_s": deadline_time_s,
            "reference_cost_ms": reference_cost_ms,
            "slo_ms": slo_ms,
            "enable_stagepool_slo": self._enable_stagepool_slo,
            "stagepool_selection_objective": self._slo_stagepool_selection_objective,
            "profile_only_reject_reason": profile_only_reject_reason,
            "candidates": candidates,
        }
        try:
            with open(str(self._stagepool_profile_path), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception as exc:
            logger.debug("[StagePool] failed to write stagepool profile: %s", exc)

    def _profile_only_reject_reason(self, candidates: list[dict[str, Any]]) -> str | None:
        if not candidates:
            return None
        threshold = self._slo_stagepool_profile_reject_laxity_ms
        laxities = [_optional_float(candidate.get("predicted_laxity_ms")) for candidate in candidates]
        finite_laxities = [value for value in laxities if value is not None]
        if finite_laxities and all(value < threshold for value in finite_laxities):
            return "all_replicas_predicted_below_profile_threshold"
        return None

    @staticmethod
    def _stage_profile_config(stage_vllm_config: Any) -> dict[str, Any]:
        additional_config = getattr(stage_vllm_config, "additional_config", None)
        if not isinstance(additional_config, dict):
            return {}
        raw = additional_config.get("diffusion_stagepool_profile") or additional_config.get("stagepool_profile") or {}
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _stage_scheduler_policy(stage_vllm_config: Any) -> str | None:
        additional_config = getattr(stage_vllm_config, "additional_config", None)
        if not isinstance(additional_config, dict):
            return None
        policy = (
            additional_config.get("diffusion_scheduler_policy")
            or additional_config.get("diffusion_step_scheduler_policy")
            or additional_config.get("scheduler_policy")
        )
        return policy if isinstance(policy, str) else None

    @classmethod
    def _default_stagepool_slo_enabled(cls, stage_vllm_config: Any) -> bool:
        policy = cls._stage_scheduler_policy(stage_vllm_config)
        if not isinstance(policy, str):
            return False
        return policy.lower() in {
            "slo",
            "slo_aware",
            "bucket_slo",
            "slo_no_preemption_guarded",
            "token_slo",
            "slo_token",
            "slo_token_dynamic",
            "slo_no_preemption_token_guarded",
            "slo_no_preemption_token_adaptive",
            "slo_no_preemption_token_objective",
            "slo_token_stagepool_objective",
            "slo_token_step_preemptive",
            "slo_step_preemptive_token",
            "slo_token_preemptive",
            "slo_token_adaptive",
        }

    def _task_sampling_key_dict(self, task: Task | None) -> dict[str, Any] | None:
        sampling_params = None if task is None else task.get("sampling_params")
        return self._sampling_key_dict(
            sampling_params,
            dynamic_qwen=self._uses_qwen_image_dynamic_step_batching(),
            has_negative_prompt=bool(task.get("has_negative_prompt")) if task is not None else False,
        )

    def _uses_qwen_image_dynamic_step_batching(self) -> bool:
        try:
            return (
                getattr(self._stage_slo_config, "model_class_name", None) == "QwenImagePipeline"
                and qwen_image_dynamic_step_batching_enabled(self._stage_slo_config)
            )
        except Exception:
            return False

    @staticmethod
    def _snapshot_token_pressure(snapshot: dict[str, Any]) -> float:
        value = _optional_float(snapshot.get("token_pressure"))
        if value is not None:
            return max(value, 0.0)
        pressure = 0.0
        buckets = snapshot.get("buckets")
        if isinstance(buckets, list):
            for bucket in buckets:
                if not isinstance(bucket, dict):
                    continue
                work = _optional_float(bucket.get("total_token_work"))
                remaining = (
                    _optional_float(bucket.get("max_remaining_steps"))
                    or _optional_float(bucket.get("min_remaining_steps"))
                    or 1.0
                )
                if work is not None:
                    pressure += max(work, 0.0) * max(remaining, 1.0)
        if pressure > 0:
            return pressure
        return float(snapshot.get("num_waiting", 0) or 0) + float(snapshot.get("num_running", 0) or 0)

    @classmethod
    def _snapshot_token_work_breakdown(cls, snapshot: dict[str, Any]) -> tuple[float, float]:
        resident_work = 0.0
        waiting_work = 0.0
        buckets = snapshot.get("buckets")
        if not isinstance(buckets, list):
            return resident_work, waiting_work
        for bucket in buckets:
            if not isinstance(bucket, dict):
                continue
            work = cls._bucket_token_work(bucket)
            if work <= 0:
                continue
            running = max(float(bucket.get("num_running", 0) or 0), 0.0)
            waiting = max(float(bucket.get("num_waiting", 0) or 0), 0.0)
            total = running + waiting
            if total <= 0:
                waiting_work += work
                continue
            resident_work += work * (running / total)
            waiting_work += work * (waiting / total)
        return resident_work, waiting_work

    @staticmethod
    def _bucket_token_work(bucket: dict[str, Any] | None) -> float:
        if bucket is None:
            return 0.0
        work = _optional_float(bucket.get("total_token_work"))
        if work is not None:
            return max(work, 0.0)
        max_tokens = _optional_float(bucket.get("max_latent_tokens")) or _optional_float(bucket.get("latent_tokens"))
        effective_batch_size = _optional_float(bucket.get("effective_batch_size"))
        if max_tokens is not None and effective_batch_size is not None:
            return max(max_tokens * effective_batch_size, 0.0)
        return max(float(bucket.get("candidate_batch_size", 0) or 0), 0.0)

    @staticmethod
    def _matching_bucket_token_utilization(bucket: dict[str, Any] | None) -> float | None:
        if bucket is None:
            return None
        return _optional_float(bucket.get("token_batch_utilization"))

    def _incoming_latent_tokens(self, sampling_params: Any) -> float:
        width, height = _sampling_dimensions(sampling_params)
        if self._slo_cost_model is not None:
            tokens = self._slo_cost_model.latent_tokens(width, height, _sampling_get(sampling_params, "num_frames", 1))
            if tokens is not None:
                return float(tokens)
        if width is None or height is None:
            resolution = _optional_float(_sampling_get(sampling_params, "resolution")) or 1024.0
            width = width or resolution
            height = height or resolution
        return max(
            (float(width) / 16.0)
            * (float(height) / 16.0)
            * max(float(_sampling_get(sampling_params, "num_frames", 1) or 1), 1.0),
            1.0,
        )

    @staticmethod
    def _sampling_key_dict(
        sampling_params: Any,
        *,
        dynamic_qwen: bool = False,
        has_negative_prompt: bool = False,
    ) -> dict[str, Any] | None:
        if sampling_params is None:
            return None
        lora_request = _sampling_get(sampling_params, "lora_request")
        if dynamic_qwen:
            true_cfg_scale = _sampling_get(sampling_params, "true_cfg_scale", None)
            true_cfg_scale = 4.0 if true_cfg_scale is None else true_cfg_scale
            do_classifier_free_guidance = float(true_cfg_scale) > 1.0 and has_negative_prompt
            out: dict[str, Any] = {}
            for field in fields(SamplingParamsKey):
                if field.name == "lora_int_id":
                    out[field.name] = None if lora_request is None else getattr(lora_request, "lora_int_id", None)
                elif field.name == "do_classifier_free_guidance":
                    out[field.name] = do_classifier_free_guidance
                elif field.name == "lora_scale":
                    out[field.name] = _sampling_get(sampling_params, "lora_scale", 1.0)
                elif field.default is not MISSING:
                    out[field.name] = field.default
                elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
                    out[field.name] = field.default_factory()  # type: ignore[misc]
                else:
                    out[field.name] = None
            return out
        out: dict[str, Any] = {}
        for field in fields(SamplingParamsKey):
            if field.name == "lora_int_id":
                out[field.name] = None if lora_request is None else getattr(lora_request, "lora_int_id", None)
            else:
                default = None
                if field.default is not MISSING:
                    default = field.default
                elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
                    default = field.default_factory()  # type: ignore[misc]
                out[field.name] = _sampling_get(sampling_params, field.name, default)
        return out

    # ---- Stage-local polling ----

    async def _poll_stage_raw(self, client: StagePoolLLMClient) -> EngineCoreOutputs | None:
        """Pull raw EngineCoreOutputs from a stage replica without processing."""
        outputs = await client.get_output_async()
        if not outputs.outputs:
            return None
        return outputs

    async def process_llm_raw_outputs(
        self,
        replica_id: int,
        raw_outputs: EngineCoreOutputs,
    ) -> list[Any]:
        """Run the shared LLM output processor on one raw poll result."""
        raw_client = self.clients[replica_id]
        if raw_client is None:
            return []
        client = cast(StagePoolLLMClient, raw_client)
        processor = self.output_processor
        processed = processor.process_outputs(
            raw_outputs.outputs,
            raw_outputs.timestamp,
            None,
        )

        if processed.reqs_to_abort:
            await client.abort_requests_async(processed.reqs_to_abort)

        if raw_outputs.scheduler_stats is not None:
            processor.update_scheduler_stats(raw_outputs.scheduler_stats)

        return processed.request_outputs

    async def poll_llm_raw_output(
        self,
        replica_id: int,
        *,
        timeout_s: float = 0.001,
    ) -> EngineCoreOutputs | None:
        """Poll raw EngineCore outputs from one LLM replica once."""
        raw_client = self.clients[replica_id]
        if raw_client is None:
            return None
        client = cast(StagePoolLLMClient, raw_client)
        try:
            return await asyncio.wait_for(
                self._poll_stage_raw(client),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            return None
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "[StagePool] _poll_stage_raw failed for stage-%s replica-%s",
                self.stage_id,
                replica_id,
            )
            raise

    def poll_diffusion_output(self, replica_id: int) -> Any | None:
        """Drain one ready diffusion output from the given replica if present."""
        raw_client = self.clients[replica_id]
        if raw_client is None:
            return None
        return cast(StagePoolDiffusionClient, raw_client).get_diffusion_output_nowait()

    # ---- Stage-local control plane ----

    async def abort_requests(self, request_ids: list[str]) -> None:
        """Abort the given requests in this stage pool.

        Request-bound abort routing stays inside the pool because route affinity
        (``request_id -> replica_id``) is pool-owned.
        """
        if not request_ids:
            return

        request_ids_by_replica: dict[int, list[str]] = {}
        for request_id in request_ids:
            replica_id = self.get_bound_replica_id(request_id)
            if replica_id is None or self.clients[replica_id] is None:
                logger.debug("[StagePool] abort: no live binding for req=%s in stage-%s", request_id, self.stage_id)
                continue
            request_ids_by_replica.setdefault(replica_id, []).append(request_id)

        for replica_id, replica_request_ids in request_ids_by_replica.items():
            client = self.clients[replica_id]
            if client is None:
                continue
            await client.abort_requests_async(replica_request_ids)

        # Clean up OutputProcessor state (e.g. mm_accumulated tensors) that
        # would otherwise leak — aborted requests never produce a final
        # EngineCoreOutput, so process_outputs() never fires its cleanup path.
        all_aborted = [rid for ids in request_ids_by_replica.values() for rid in ids]
        if all_aborted and self._output_processor is not None:
            self._output_processor.abort_requests(all_aborted, internal=True)

    async def collective_rpc(
        self,
        replica_id: int,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any] | Any:
        """Dispatch a stage-scoped control-plane RPC to one physical route."""
        kwargs = dict(kwargs or {})
        client = self.clients[replica_id]
        if client is None:
            return {
                "supported": False,
                "error": f"stage {self.stage_id} replica {replica_id} is not attached",
            }
        try:
            return await client.collective_rpc_async(
                method=method,
                timeout=timeout,
                args=args,
                kwargs=kwargs,
            )
        except Exception as exc:
            logger.exception(
                "[StagePool] collective_rpc failed: stage=%s replica=%s method=%s",
                self.stage_id,
                replica_id,
                method,
            )
            return {
                "supported": False,
                "error": str(exc),
            }

    def shutdown_replica(self, replica_id: int) -> None:
        """Shutdown one backend handle in this stage pool."""
        if replica_id >= len(self.clients):
            return
        client = self.clients[replica_id]
        if client is None:
            return
        try:
            client.shutdown()
            logger.info(
                "[StagePool] Stage %d replica %d shut down",
                self.stage_id,
                replica_id,
            )
        except Exception as e:
            logger.warning(
                "[StagePool] Failed to shutdown stage %d replica %d: %s",
                self.stage_id,
                replica_id,
                e,
            )
