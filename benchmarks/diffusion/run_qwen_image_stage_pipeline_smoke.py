# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import concurrent.futures
import copy
import json
import os
import time
from pathlib import Path

from openai import NOT_GIVEN as omit

from tests.helpers.runtime import (
    OmniServerStageCli,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import (
    QWEN_IMAGE_DENOISE_BATCH_TRACE_KEY,
    QWEN_IMAGE_STAGE_TRACE_KEY,
)


def _prepend_pythonpath(path: str, existing: str | None) -> str:
    entries = [path]
    seen = {path}
    for entry in (existing or "").split(os.pathsep):
        if not entry or entry in seen:
            continue
        entries.append(entry)
        seen.add(entry)
    return os.pathsep.join(entries)


def _chat_completion_metrics(chat_completion) -> dict:
    merged: dict = {}
    metrics = getattr(chat_completion, "metrics", None)
    if isinstance(metrics, dict):
        merged.update(metrics)
    model_extra = getattr(chat_completion, "model_extra", None)
    if isinstance(model_extra, dict) and isinstance(model_extra.get("metrics"), dict):
        merged.update(model_extra["metrics"])
    if hasattr(chat_completion, "model_dump"):
        dumped = chat_completion.model_dump()
        if isinstance(dumped, dict):
            if isinstance(dumped.get("metrics"), dict):
                merged.update(dumped["metrics"])
            content_metrics: dict = {}
            for choice in dumped.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message") or {}
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    for key in (
                        QWEN_IMAGE_STAGE_TRACE_KEY,
                        QWEN_IMAGE_DENOISE_BATCH_TRACE_KEY,
                        "stage_durations",
                        "peak_memory_mb",
                    ):
                        if key in item:
                            content_metrics[key] = item[key]
            merged.update(content_metrics)
    return merged


def _chat_completion_debug_keys(chat_completion) -> dict:
    debug = {
        "attrs": sorted(key for key in ("metrics", "model_extra") if hasattr(chat_completion, key)),
        "dump_keys": [],
        "metrics_keys": [],
    }
    if hasattr(chat_completion, "model_dump"):
        dumped = chat_completion.model_dump()
        if isinstance(dumped, dict):
            debug["dump_keys"] = sorted(dumped.keys())
            metrics = dumped.get("metrics")
            if isinstance(metrics, dict):
                debug["metrics_keys"] = sorted(metrics.keys())
            content_keys = set()
            for choice in dumped.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message") or {}
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            content_keys.update(item.keys())
            debug["content_item_keys"] = sorted(content_keys)
    return debug


def _stage_trace_from_metrics(metrics: dict) -> list[dict]:
    trace = metrics.get(QWEN_IMAGE_STAGE_TRACE_KEY)
    return trace if isinstance(trace, list) else []


def _denoise_batch_trace_from_metrics(metrics: dict) -> list[dict]:
    trace = metrics.get(QWEN_IMAGE_DENOISE_BATCH_TRACE_KEY)
    return trace if isinstance(trace, list) else []


def _stage_durations_from_metrics(metrics: dict) -> dict:
    stage_durations = metrics.get("stage_durations")
    return stage_durations if isinstance(stage_durations, dict) else {}


def _trace_overlaps(requests: list[dict]) -> dict[str, bool]:
    events: list[tuple[int, dict]] = []
    for request in requests:
        for event in request.get("stage_trace", []):
            if all(key in event for key in ("stage", "start_s", "end_s")):
                events.append((int(request["request_index"]), event))

    overlaps = {
        "encode_overlaps_denoise": False,
        "denoise_overlaps_decode": False,
        "encode_overlaps_decode": False,
        "any_cross_request_overlap": False,
    }
    for left_idx, left in events:
        for right_idx, right in events:
            if left_idx == right_idx:
                continue
            if float(left["start_s"]) < float(right["end_s"]) and float(right["start_s"]) < float(left["end_s"]):
                overlaps["any_cross_request_overlap"] = True
                pair = {str(left["stage"]), str(right["stage"])}
                if pair == {"encode", "denoise"}:
                    overlaps["encode_overlaps_denoise"] = True
                elif pair == {"denoise", "decode"}:
                    overlaps["denoise_overlaps_decode"] = True
                elif pair == {"encode", "decode"}:
                    overlaps["encode_overlaps_decode"] = True
    return overlaps


def _parse_steps_list(default_steps: int) -> list[int]:
    raw = os.environ.get("QWEN_IMAGE_STAGE_PIPELINE_STEPS_LIST")
    if not raw:
        return [default_steps]
    steps: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            steps.append(int(item))
    return steps or [default_steps]


def _make_request_config(
    model: str,
    request_index: int,
    *,
    height: int,
    width: int,
    num_inference_steps: int,
    true_cfg_scale: float,
) -> dict:
    return {
        "model": model,
        "messages": dummy_messages_from_mix_data(
            content_text=f"A red apple on a white plate, product photo. request {request_index}."
        ),
        "extra_body": {
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "negative_prompt": "blurry, low quality",
            "true_cfg_scale": true_cfg_scale,
            "seed": 123 + request_index,
        },
    }


def _send_traced_diffusion_request(
    client: OpenAIClientHandler,
    request_config: dict,
    *,
    request_index: int,
    stagger_s: float,
) -> dict:
    if stagger_s > 0:
        time.sleep(stagger_s * request_index)
    wall_start_s = time.time()
    perf_start_s = time.perf_counter()
    cfg = copy.deepcopy(request_config)
    chat_completion = client.client.chat.completions.create(
        model=cfg.get("model"),
        messages=cfg.get("messages"),
        extra_body=cfg.get("extra_body"),
        modalities=cfg.get("modalities", omit),
    )
    response = client._process_diffusion_response(chat_completion, wall_start=perf_start_s)
    metrics = _chat_completion_metrics(chat_completion)
    response_debug = _chat_completion_debug_keys(chat_completion)
    if len(response.images or []) != 1:
        raise RuntimeError(f"Smoke request {request_index} did not return one image.")
    image = response.images[0]
    wall_end_s = time.time()
    return {
        "request_index": request_index,
        "status": "passed",
        "wall_start_s": wall_start_s,
        "wall_end_s": wall_end_s,
        "wall_duration_s": wall_end_s - wall_start_s,
        "e2e_latency_s": response.e2e_latency,
        "image_count": len(response.images or []),
        "image_size": list(image.size),
        "response_debug": response_debug,
        "stage_trace": _stage_trace_from_metrics(metrics),
        "denoise_batch_trace": _denoise_batch_trace_from_metrics(metrics),
        "stage_durations": _stage_durations_from_metrics(metrics),
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    stage_config = Path(
        os.environ.get(
            "QWEN_IMAGE_STAGE_PIPELINE_CONFIG",
            repo / "vllm_omni/deploy/qwen_image_stage_pipeline.yaml",
        )
    )
    outdir = Path(
        os.environ.get(
            "QWEN_IMAGE_STAGE_PIPELINE_SMOKE_OUTDIR",
            repo / "benchmark_outputs/qwen_image_stage_pipeline_smoke",
        )
    )
    outdir.mkdir(parents=True, exist_ok=True)
    result_path = outdir / "smoke_result.json"

    model = os.environ.get("QWEN_IMAGE_STAGE_PIPELINE_MODEL", "Qwen/Qwen-Image")
    request_count = int(os.environ.get("QWEN_IMAGE_STAGE_PIPELINE_REQUESTS", "1"))
    warmup_count = int(os.environ.get("QWEN_IMAGE_STAGE_PIPELINE_WARMUP_REQUESTS", "0"))
    stagger_s = float(os.environ.get("QWEN_IMAGE_STAGE_PIPELINE_STAGGER_S", "0"))
    height = int(os.environ.get("QWEN_IMAGE_STAGE_PIPELINE_HEIGHT", "256"))
    width = int(os.environ.get("QWEN_IMAGE_STAGE_PIPELINE_WIDTH", "256"))
    num_inference_steps = int(os.environ.get("QWEN_IMAGE_STAGE_PIPELINE_STEPS", "1"))
    true_cfg_scale = float(os.environ.get("QWEN_IMAGE_STAGE_PIPELINE_TRUE_CFG_SCALE", "2.0"))
    steps_list = _parse_steps_list(num_inference_steps)
    env_dict = {
        "PYTHONPATH": _prepend_pythonpath(str(repo), os.environ.get("PYTHONPATH")),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
    }
    for name in (
        "ASCEND_RT_VISIBLE_DEVICES",
        "QUANTIZATION_CONFIG",
        "PR4024_DYNAMIC_ATTENTION_BACKEND",
        "MINDIE_SD_FA_TYPE",
        "DIFFUSION_ATTENTION_BACKEND",
        "VLLM_USE_MODELSCOPE",
        "QWEN_IMAGE_STAGE_FINE_TIMING",
        "QWEN_IMAGE_STAGE_TIMING_SYNC",
        "QWEN_IMAGE_DENOISE_BATCH_TRACE",
    ):
        if name in os.environ:
            env_dict[name] = os.environ[name]

    server_args = [
        "--stage-init-timeout",
        "900",
        "--init-timeout",
        "1200",
        "--log-stats",
    ]
    lb_policy = os.environ.get("QWEN_IMAGE_STAGE_PIPELINE_LB_POLICY", "round-robin")
    if lb_policy == "random":
        raise ValueError(
            "Qwen-Image stage pipeline experiments must not use random routing; "
            "use round-robin for balanced DiT replica assignment."
        )
    server_args.extend(["--omni-lb-policy", lb_policy])

    sample_request = _make_request_config(
        model,
        0,
        height=height,
        width=width,
        num_inference_steps=steps_list[0],
        true_cfg_scale=true_cfg_scale,
    )

    started = time.time()
    status = {
        "status": "running",
        "started_at": started,
        "model": model,
        "stage_config": str(stage_config),
        "request_count": request_count,
        "warmup_count": warmup_count,
        "steps_list": steps_list,
        "lb_policy": lb_policy,
        "request": sample_request["extra_body"],
    }
    result_path.write_text(json.dumps(status, indent=2, sort_keys=True))

    try:
        with OmniServerStageCli(
            model,
            str(stage_config),
            server_args,
            env_dict=env_dict,
        ) as server:
            client = OpenAIClientHandler(
                host=server.host,
                port=server.port,
                log_stats=server.log_stats,
            )
            scenarios = []
            for scenario_index, steps in enumerate(steps_list):
                scenario_started = time.time()
                warmup_requests = []
                warmup_started = time.time()
                for warmup_index in range(warmup_count):
                    warmup_request_index = 100_000 + scenario_index * 1_000 + warmup_index
                    warmup_config = _make_request_config(
                        model,
                        warmup_request_index,
                        height=height,
                        width=width,
                        num_inference_steps=steps,
                        true_cfg_scale=true_cfg_scale,
                    )
                    warmup_requests.append(
                        _send_traced_diffusion_request(
                            client,
                            warmup_config,
                            request_index=warmup_index,
                            stagger_s=0,
                        )
                    )
                warmup_elapsed_s = time.time() - warmup_started

                request_configs = [
                    _make_request_config(
                        model,
                        scenario_index * 10_000 + index,
                        height=height,
                        width=width,
                        num_inference_steps=steps,
                        true_cfg_scale=true_cfg_scale,
                    )
                    for index in range(request_count)
                ]
                measured_started = time.time()
                with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, request_count)) as executor:
                    futures = [
                        executor.submit(
                            _send_traced_diffusion_request,
                            client,
                            request_config,
                            request_index=index,
                            stagger_s=stagger_s,
                        )
                        for index, request_config in enumerate(request_configs)
                    ]
                    requests = [future.result() for future in futures]
                measured_elapsed_s = time.time() - measured_started
                requests.sort(key=lambda item: item["request_index"])
                warmup_requests.sort(key=lambda item: item["request_index"])
                scenarios.append(
                    {
                        "scenario_index": scenario_index,
                        "height": height,
                        "width": width,
                        "num_inference_steps": steps,
                        "true_cfg_scale": true_cfg_scale,
                        "elapsed_s": measured_elapsed_s,
                        "measured_elapsed_s": measured_elapsed_s,
                        "warmup_elapsed_s": warmup_elapsed_s,
                        "total_elapsed_s": time.time() - scenario_started,
                        "warmup_requests": warmup_requests,
                        "requests": requests,
                        "trace_overlaps": _trace_overlaps(requests),
                    }
                )
                status.update(
                    {
                        "status": "running",
                        "elapsed_s": time.time() - started,
                        "completed_scenarios": len(scenarios),
                        "scenarios": scenarios,
                    }
                )
                result_path.write_text(json.dumps(status, indent=2, sort_keys=True))

            requests = [request for scenario in scenarios for request in scenario["requests"]]
            traces_missing = [
                f"{scenario['scenario_index']}:{item['request_index']}"
                for scenario in scenarios
                for item in scenario["requests"]
                if not item.get("stage_trace")
            ]
            status.update(
                {
                    "status": "passed",
                    "elapsed_s": time.time() - started,
                    "measured_elapsed_s": sum(float(scenario["measured_elapsed_s"]) for scenario in scenarios),
                    "warmup_elapsed_s": sum(float(scenario["warmup_elapsed_s"]) for scenario in scenarios),
                    "image_count": sum(item["image_count"] for item in requests),
                    "image_size": requests[0]["image_size"] if requests else None,
                    "scenarios": scenarios,
                    "requests": scenarios[0]["requests"] if len(scenarios) == 1 else requests,
                    "trace_overlaps": _trace_overlaps(requests),
                    "traces_missing": traces_missing,
                }
            )
            result_path.write_text(json.dumps(status, indent=2, sort_keys=True))
            if traces_missing and os.environ.get("QWEN_IMAGE_STAGE_PIPELINE_REQUIRE_TRACE", "1") != "0":
                status["status"] = "failed"
                status["error"] = f"Smoke request(s) missing stage trace: {traces_missing}"
                result_path.write_text(json.dumps(status, indent=2, sort_keys=True))
                raise RuntimeError(status["error"])
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "elapsed_s": time.time() - started,
                "error": repr(exc),
            }
        )
        result_path.write_text(json.dumps(status, indent=2, sort_keys=True))
        raise

    result_path.write_text(json.dumps(status, indent=2, sort_keys=True))
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
