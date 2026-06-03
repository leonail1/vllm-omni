"""Adapters that expose existing stage pools through the DAG runtime API."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from vllm_omni.engine.dag_types import (
    DagStageBatchSpec,
    DagStageInput,
    DagStageKind,
    DagStageOutput,
    DagStageReplicaSpec,
    DagStageResourceSpec,
    DagStageSpec,
)


@runtime_checkable
class DagStageAdapter(Protocol):
    stage_spec: DagStageSpec

    async def prepare_input(self, dag_input: DagStageInput) -> DagStageInput: ...

    async def submit(self, dag_input: DagStageInput, *, req_state: Any, **kwargs: Any) -> Any: ...

    async def poll(self) -> list[DagStageOutput]: ...

    async def cancel(self, request_id: str) -> None: ...

    async def cleanup(self, request_id: str) -> None: ...

    def snapshot(self) -> dict[str, Any]: ...


def _read_config_value(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    if hasattr(obj, "get"):
        try:
            return obj.get(key)
        except Exception:
            pass
    return getattr(obj, key, None)


def _normalize_kind(value: Any) -> DagStageKind | None:
    if value is None:
        return None
    try:
        return DagStageKind(str(value).strip().lower())
    except ValueError:
        return None


def infer_stage_kind(stage_type: str | None, model_stage: str | None, final_output_type: str | None) -> DagStageKind:
    """Infer a DAG role from existing vLLM-Omni stage metadata."""

    normalized_stage = (model_stage or "").lower()
    normalized_output = (final_output_type or "").lower()
    normalized_type = (stage_type or "").lower()
    if "text_encoder" in normalized_stage or normalized_stage in {"text_encoder", "text-encoder"}:
        return DagStageKind.TEXT_ENCODER
    if "image_encoder" in normalized_stage or normalized_stage in {"image_encoder", "vision_encoder"}:
        return DagStageKind.IMAGE_ENCODER
    if "vae_encoder" in normalized_stage:
        return DagStageKind.VAE_ENCODER
    if "vae_decoder" in normalized_stage:
        return DagStageKind.VAE_DECODER
    if "audio_decoder" in normalized_stage or "code2wav" in normalized_stage or normalized_output == "audio":
        return DagStageKind.AUDIO_DECODER
    if normalized_type == "diffusion" or "diffusion" in normalized_stage or "dit" in normalized_stage:
        return DagStageKind.DIT_DENOISE
    return DagStageKind.GENERIC_LLM


def _stage_kind_from_pool(pool: Any) -> DagStageKind:
    client = getattr(pool, "stage_client", None)
    stage_cfg = getattr(pool, "stage_vllm_config", None)
    additional_config = _read_config_value(stage_cfg, "additional_config")
    explicit = _normalize_kind(_read_config_value(additional_config, "dag_stage_kind"))
    if explicit is not None:
        return explicit
    return infer_stage_kind(
        getattr(pool, "stage_type", None),
        getattr(client, "model_stage", None),
        getattr(client, "final_output_type", None),
    )


def build_stage_spec_from_pool(pool: Any, *, pres: tuple[int, ...], subs: tuple[int, ...]) -> DagStageSpec:
    live_replica_ids = tuple(int(rid) for rid in pool.live_replica_ids())
    replicas: list[DagStageReplicaSpec] = []
    for rid in live_replica_ids:
        client = pool.clients[rid]
        if hasattr(pool, "dag_replica_device_ids"):
            device_ids = tuple(pool.dag_replica_device_ids(rid))
        else:
            device_attr = getattr(client, "devices", None) or getattr(client, "device", None)
            if isinstance(device_attr, str):
                device_ids = tuple(part.strip() for part in device_attr.split(",") if part.strip())
            elif isinstance(device_attr, (list, tuple)):
                device_ids = tuple(str(part) for part in device_attr)
            else:
                device_ids = ()
        replicas.append(DagStageReplicaSpec(replica_id=rid, device_ids=device_ids))
    if not replicas:
        replicas.append(DagStageReplicaSpec(replica_id=0, device_ids=()))

    card_counts: list[int] = []
    for replica in replicas:
        if hasattr(pool, "dag_replica_card_count"):
            card_counts.append(max(1, int(pool.dag_replica_card_count(replica.replica_id))))
        else:
            card_counts.append(max(1, replica.card_count))
    inferred_cards = max(card_counts, default=1)
    stage_vllm_config = getattr(pool, "stage_vllm_config", None)
    resource = DagStageResourceSpec(
        replica_count=len(replicas),
        cards_per_replica=inferred_cards,
        replicas=tuple(replicas),
        max_batch_size=getattr(stage_vllm_config, "max_num_seqs", None),
    )
    client = pool.stage_client
    scheduler_policy = None
    additional_config = _read_config_value(getattr(pool, "stage_slo_config", None), "additional_config")
    if isinstance(additional_config, dict):
        scheduler_policy = additional_config.get("diffusion_scheduler_policy")
    kind = _stage_kind_from_pool(pool)
    return DagStageSpec(
        stage_id=int(pool.stage_id),
        name=f"stage-{pool.stage_id}:{kind.value}",
        kind=kind,
        pres=pres,
        subs=subs,
        resource=resource,
        batch=DagStageBatchSpec(
            scheduler_policy=scheduler_policy,
            allow_step_boundary_preemption=kind == DagStageKind.DIT_DENOISE,
            max_batch_size=resource.max_batch_size,
        ),
        stage_type=getattr(pool, "stage_type", None),
        final_output=bool(getattr(pool, "final_output", False)),
        final_output_type=getattr(client, "final_output_type", None),
        metadata={"model_stage": getattr(client, "model_stage", None)},
    )


class StagePoolDagAdapter:
    """Real DAG adapter backed by an existing StagePool."""

    def __init__(self, pool: Any, stage_spec: DagStageSpec) -> None:
        self.pool = pool
        self.stage_spec = stage_spec

    async def prepare_input(self, dag_input: DagStageInput) -> DagStageInput:
        return dag_input

    async def submit(self, dag_input: DagStageInput, *, req_state: Any, **kwargs: Any) -> Any:
        submit_kwargs = dict(kwargs.pop("submit_kwargs", {}) or {})
        params_override = kwargs.pop("params_override", None)
        prompt_text = kwargs.pop("prompt_text", None)
        affinity_request_id = kwargs.pop("affinity_request_id", None)
        return await self.pool.submit_initial(
            dag_input.request_id,
            req_state,
            dag_input.payload,
            submit_kwargs=submit_kwargs,
            params_override=params_override,
            prompt_text=prompt_text,
            affinity_request_id=affinity_request_id,
        )

    async def poll(self) -> list[DagStageOutput]:
        outputs: list[DagStageOutput] = []
        for replica_id in self.pool.live_replica_ids():
            if self.pool.stage_type == "diffusion":
                output = self.pool.poll_diffusion_output(replica_id)
                if output is not None:
                    outputs.append(
                        DagStageOutput(
                            request_id=output.request_id,
                            stage_id=self.stage_spec.stage_id,
                            payload=output,
                            finished=bool(getattr(output, "finished", False)),
                            metadata={"replica_id": replica_id},
                        )
                    )
                continue
            raw_outputs = await self.pool.poll_llm_raw_output(replica_id, timeout_s=0.001)
            if raw_outputs is None:
                continue
            for output in await self.pool.process_llm_raw_outputs(replica_id, raw_outputs):
                outputs.append(
                    DagStageOutput(
                        request_id=output.request_id,
                        stage_id=self.stage_spec.stage_id,
                        payload=output,
                        finished=bool(getattr(output, "finished", False)),
                        metadata={"replica_id": replica_id},
                    )
                )
        return outputs

    async def cancel(self, request_id: str) -> None:
        await self.pool.abort_requests([request_id])

    async def cleanup(self, request_id: str) -> None:
        self.pool.release_bindings([request_id])

    def snapshot(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_spec.stage_id,
            "kind": self.stage_spec.kind.value,
            "stage_type": self.stage_spec.stage_type,
            "replicas": list(self.pool.live_replica_ids()),
            "final_output": self.stage_spec.final_output,
            "final_output_type": self.stage_spec.final_output_type,
        }
