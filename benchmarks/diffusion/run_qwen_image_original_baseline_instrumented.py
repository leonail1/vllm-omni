from __future__ import annotations

import concurrent.futures
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI


MODEL = "Qwen/Qwen-Image"
REPO = Path("/home/lzg/vllm-omni-pr4024-stage-pipeline")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests.helpers.runtime import dummy_messages_from_mix_data
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import (
    QWEN_IMAGE_DENOISE_BATCH_TRACE_KEY,
    QWEN_IMAGE_STAGE_TRACE_KEY,
)

REQUESTS = int(
    os.environ.get(
        "ORIGINAL_BASELINE_REQUESTS",
        os.environ.get("ORIGINAL_8REPLICA_REQUESTS", "56"),
    )
)
WARMUP_REQUESTS_PER_REPLICA = int(os.environ.get("ORIGINAL_BASELINE_WARMUP_REQUESTS_PER_REPLICA", "1"))
DEVICE_IDS = [
    device.strip()
    for device in os.environ.get("ORIGINAL_BASELINE_DEVICE_IDS", "1,2,3,4,5,6,7").split(",")
    if device.strip()
]
REPLICAS = len(DEVICE_IDS)
def parse_steps_list() -> list[int]:
    raw = os.environ.get("ORIGINAL_BASELINE_STEPS_LIST")
    if not raw:
        return [4, 8, 12]
    steps = [int(item.strip()) for item in raw.split(",") if item.strip()]
    return steps or [4, 8, 12]


STEPS_LIST = parse_steps_list()
HEIGHT = 1024
WIDTH = 1024
TRUE_CFG_SCALE = 2.0
NEGATIVE_PROMPT = "blurry, low quality"


def get_open_ports(count: int) -> list[int]:
    sockets = []
    ports = []
    try:
        for _ in range(count):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", 0))
            sockets.append(s)
            ports.append(int(s.getsockname()[1]))
    finally:
        for s in sockets:
            s.close()
    return ports


def wait_port(port: int, *, timeout_s: float = 1200.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(2.0)
    raise TimeoutError(f"server on port {port} did not become ready within {timeout_s}s")


def request_payload(index: int, steps: int) -> dict:
    return {
        "model": MODEL,
        "messages": dummy_messages_from_mix_data(
            content_text=f"A red apple on a white plate, product photo. request {index}."
        ),
        "extra_body": {
            "height": HEIGHT,
            "width": WIDTH,
            "num_inference_steps": steps,
            "negative_prompt": NEGATIVE_PROMPT,
            "true_cfg_scale": TRUE_CFG_SCALE,
            "seed": 123 + index,
        },
    }


def send_request(port: int, request_index: int, steps: int) -> dict:
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="EMPTY", timeout=300.0)
    payload = request_payload(request_index, steps)
    start_wall = time.time()
    start = time.perf_counter()
    completion = client.chat.completions.create(**payload)
    latency = time.perf_counter() - start
    end_wall = time.time()
    completion_payload = completion.model_dump(mode="json") if hasattr(completion, "model_dump") else completion
    image_payload_count = count_image_payloads(completion_payload)
    if image_payload_count <= 0:
        raise RuntimeError(
            f"request {request_index} on port {port} returned no detectable image payload"
        )
    metrics = chat_completion_metrics(completion)
    return {
        "request_index": request_index,
        "port": port,
        "status": "passed",
        "latency_s": latency,
        "wall_start_s": start_wall,
        "wall_end_s": end_wall,
        "image_payload_count": image_payload_count,
        "stage_trace": metrics.get(QWEN_IMAGE_STAGE_TRACE_KEY, []),
        "denoise_batch_trace": metrics.get(QWEN_IMAGE_DENOISE_BATCH_TRACE_KEY, []),
        "stage_durations": metrics.get("stage_durations", {}),
        "response_debug": chat_completion_debug_keys(completion),
    }


def chat_completion_metrics(chat_completion) -> dict:
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


def chat_completion_debug_keys(chat_completion) -> dict:
    debug = {
        "attrs": sorted(key for key in ("metrics", "model_extra") if hasattr(chat_completion, key)),
        "dump_keys": [],
        "metrics_keys": [],
        "content_item_keys": [],
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


def count_image_payloads(value: object) -> int:
    if isinstance(value, dict):
        count = 0
        value_type = str(value.get("type", "")).lower()
        if value_type in {"image", "image_url", "output_image"}:
            count += 1
        if "image_url" in value or "b64_json" in value:
            count += 1
        return count + sum(count_image_payloads(child) for child in value.values())
    if isinstance(value, list):
        return sum(count_image_payloads(child) for child in value)
    if isinstance(value, str):
        lowered = value.lower()
        return int("data:image/" in lowered or "![image]" in lowered or "![generated image]" in lowered)
    return 0


def percentile(vals: list[float], q: float) -> float:
    vals = sorted(vals)
    if len(vals) == 1:
        return vals[0]
    rank = q * (len(vals) - 1)
    low = int(rank)
    high = min(low + 1, len(vals) - 1)
    frac = rank - low
    return vals[low] + (vals[high] - vals[low]) * frac


def run_batch(ports: list[int], steps: int, request_count: int) -> tuple[list[dict], float]:
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=request_count) as executor:
        futures = [
            executor.submit(send_request, ports[index % len(ports)], index, steps)
            for index in range(request_count)
        ]
        results = [future.result() for future in futures]
    elapsed = time.perf_counter() - start
    return results, elapsed


def launch_servers(outdir: Path, ports: list[int]) -> list[subprocess.Popen]:
    procs: list[subprocess.Popen] = []
    try:
        for replica_id, port in enumerate(ports):
            device_id = DEVICE_IDS[replica_id]
            env = os.environ.copy()
            env.update(
                {
                    "ASCEND_RT_VISIBLE_DEVICES": device_id,
                    "HF_HUB_OFFLINE": "1",
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                    "PYTHONPATH": str(REPO) + os.pathsep + env.get("PYTHONPATH", ""),
                    "VLLM_OMNI_STAGE_ID": "0",
                    "VLLM_OMNI_REPLICA_ID": str(replica_id),
                }
            )
            env.pop("VLLM_USE_MODELSCOPE", None)
            # Preserve acceleration-related envs when the parent run script sets them.
            log_path = outdir / f"server_{replica_id}.log"
            cmd = [
                sys.executable,
                "-m",
                "vllm_omni.entrypoints.cli.main",
                "serve",
                MODEL,
                "--omni",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--step-execution",
                "--max-num-seqs",
                str(REQUESTS),
                "--enforce-eager",
                "--cache-backend",
                "none",
                "--stage-init-timeout",
                "900",
                "--init-timeout",
                "1200",
                "--log-stats",
            ]
            with log_path.open("w") as log:
                procs.append(
                    subprocess.Popen(
                        cmd,
                        cwd=REPO,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
    except Exception:
        terminate_servers(procs)
        raise
    return procs


def terminate_servers(procs: list[subprocess.Popen]) -> None:
    for proc in procs:
        try:
            os.killpg(proc.pid, 15)
        except ProcessLookupError:
            pass
    deadline = time.time() + 20
    for proc in procs:
        remaining = max(0.0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, 9)
            except ProcessLookupError:
                pass
    for proc in procs:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, 9)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


def main() -> None:
    if not DEVICE_IDS:
        raise ValueError("ORIGINAL_BASELINE_DEVICE_IDS must contain at least one device id")
    ts = time.strftime("%Y%m%d-%H%M%S")
    outdir = Path(
        os.environ.get(
            "ORIGINAL_BASELINE_OUTDIR",
            REPO / "benchmark_outputs" / f"codex-qwen-image-original-{REPLICAS}replica-baseline-{ts}",
        )
    )
    outdir.mkdir(parents=True, exist_ok=True)
    latest = Path(
        os.environ.get(
            "ORIGINAL_BASELINE_LATEST",
            REPO / f"benchmark_outputs/codex-qwen-image-original-{REPLICAS}replica-baseline.latest",
        )
    )
    latest.unlink(missing_ok=True)
    latest.symlink_to(outdir)
    ports = get_open_ports(REPLICAS)
    result_path = outdir / f"original_{REPLICAS}replica_baseline_result.json"
    status = {
        "status": "starting",
        "model": MODEL,
        "replicas": REPLICAS,
        "device_ids": DEVICE_IDS,
        "resource_topology": (
            f"{REPLICAS} independent non-stage Qwen-Image servers; each full "
            "encode->DiT->decode pipeline uses one NPU"
        ),
        "ports": ports,
        "steps_list": STEPS_LIST,
        "request_count": REQUESTS,
        "warmup_requests_per_replica": WARMUP_REQUESTS_PER_REPLICA,
        "height": HEIGHT,
        "width": WIDTH,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "started_at": time.time(),
    }
    result_path.write_text(json.dumps(status, indent=2, sort_keys=True))
    procs: list[subprocess.Popen] = []
    try:
        procs = launch_servers(outdir, ports)
        status["server_pids"] = [proc.pid for proc in procs]
        status["status"] = "loading"
        result_path.write_text(json.dumps(status, indent=2, sort_keys=True))
        for replica_id, (proc, port) in enumerate(zip(procs, ports, strict=True)):
            if proc.poll() is not None:
                raise RuntimeError(f"server replica {replica_id} exited early with code {proc.returncode}")
            wait_port(port)
            print(f"server_ready replica={replica_id} port={port}", flush=True)

        status["status"] = "warming"
        result_path.write_text(json.dumps(status, indent=2, sort_keys=True))
        # Warm every full replica to avoid measuring first VAE decode cold start.
        warmup_count = REPLICAS * max(0, WARMUP_REQUESTS_PER_REPLICA)
        warmup, warmup_elapsed = run_batch(ports, 1, warmup_count) if warmup_count else ([], 0.0)
        print(f"warmup_done elapsed={warmup_elapsed:.3f}s", flush=True)

        scenarios = []
        total_elapsed = 0.0
        for steps in STEPS_LIST:
            print(f"running steps={steps}", flush=True)
            requests, elapsed = run_batch(ports, steps, REQUESTS)
            total_elapsed += elapsed
            latencies = [r["latency_s"] for r in requests]
            scenarios.append(
                {
                    "num_inference_steps": steps,
                    "request_count": REQUESTS,
                    "successful_requests": len(requests),
                    "failed_requests": 0,
                    "measured_elapsed_s": elapsed,
                    "throughput_img_s": len(requests) / elapsed,
                    "latency_mean_s": sum(latencies) / len(latencies),
                    "latency_p50_s": percentile(latencies, 0.50),
                    "latency_p95_s": percentile(latencies, 0.95),
                    "latency_p99_s": percentile(latencies, 0.99),
                    "requests": requests,
                }
            )
        status.update(
            {
                "status": "passed",
                "completed_scenarios": len(scenarios),
                "scenarios": scenarios,
                "image_count": sum(s["successful_requests"] for s in scenarios),
                "measured_elapsed_s": total_elapsed,
                "warmup_elapsed_s": warmup_elapsed,
                "warmup_requests": warmup,
                "elapsed_s": time.time() - status["started_at"],
            }
        )
        result_path.write_text(json.dumps(status, indent=2, sort_keys=True))
        print(
            json.dumps(
                {
                    "status": status["status"],
                    "image_count": status["image_count"],
                    "measured_elapsed_s": status["measured_elapsed_s"],
                    "outdir": str(outdir),
                },
                indent=2,
            ),
            flush=True,
        )
    except Exception as exc:
        status.update({"status": "failed", "error": repr(exc), "elapsed_s": time.time() - status["started_at"]})
        result_path.write_text(json.dumps(status, indent=2, sort_keys=True))
        raise
    finally:
        terminate_servers(procs)


if __name__ == "__main__":
    main()
