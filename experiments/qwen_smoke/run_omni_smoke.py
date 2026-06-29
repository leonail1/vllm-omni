import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import vllm_omni
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="a cup of coffee on the table")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--step-execution", action="store_true")
    parser.add_argument("--stage-configs-path")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--batch-trace")
    parser.add_argument("--max-num-seqs", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if bool(args.stage_configs_path) != bool(args.step_execution):
        raise SystemExit("--stage-configs-path and --step-execution must be used together.")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    result = {
        "model": args.model,
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "seed": args.seed,
        "step_execution": args.step_execution,
        "stage_configs_path": args.stage_configs_path,
        "warmup_runs": args.warmup_runs,
        "repeats": args.repeats,
        "num_prompts": args.num_prompts,
        "batch_trace": args.batch_trace,
        "max_num_seqs": args.max_num_seqs,
        "vllm_omni_path": str(Path(vllm_omni.__file__).resolve()),
    }
    omni = None
    try:
        if args.batch_trace:
            Path(args.batch_trace).parent.mkdir(parents=True, exist_ok=True)
            Path(args.batch_trace).write_text("")
            os.environ["VLLM_OMNI_DIFFUSION_BATCH_TRACE"] = args.batch_trace
        t0 = time.perf_counter()
        omni = Omni(
            model=args.model,
            mode="text-to-image",
            step_execution=args.step_execution,
            enforce_eager=True,
            log_stats=False,
            tensor_parallel_size=1,
            max_num_seqs=args.max_num_seqs,
            stage_configs_path=args.stage_configs_path,
        )
        t1 = time.perf_counter()
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        warmup_samples = []
        samples = []
        last_image = None

        def make_params():
            return OmniDiffusionSamplingParams(
                height=args.height,
                width=args.width,
                seed=args.seed,
                num_inference_steps=args.steps,
                true_cfg_scale=args.cfg_scale,
                guidance_scale=args.guidance_scale,
            )

        def make_generation_params():
            if args.stage_configs_path:
                return [make_params() for _ in range(omni.num_stages)]
            return make_params()

        def make_prompts():
            if args.num_prompts == 1:
                return args.prompt
            return [args.prompt for _ in range(args.num_prompts)]

        def measured_output_path(measure_idx, out_idx):
            if args.repeats == 1 and args.num_prompts == 1:
                return output_path
            suffix_parts = []
            if args.repeats > 1:
                suffix_parts.append(f"m{measure_idx:02d}")
            if args.num_prompts > 1:
                suffix_parts.append(f"req{out_idx:02d}")
            suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
            return output_path.with_name(
                f"{output_path.stem}{suffix}{output_path.suffix}"
            )

        def run_generate():
            params = make_generation_params()
            prompts = make_prompts()
            gen_start = time.perf_counter()
            outputs = omni.generate(prompts, params, use_tqdm=False)
            gen_end = time.perf_counter()
            return outputs, gen_end - gen_start

        for idx in range(args.warmup_runs):
            outputs, gen_s = run_generate()
            warmup_outputs = []
            for out_idx, output in enumerate(outputs):
                image = output.request_output.images[0]
                last_image = image
                warmup_outputs.append(
                    {
                        "idx": out_idx,
                        "request_id": getattr(output, "request_id", None),
                        "size": image.size,
                    }
                )
            warmup_samples.append(
                {
                    "idx": idx,
                    "gen_s": gen_s,
                    "outputs": warmup_outputs,
                }
            )

        primary_output = None
        for idx in range(args.repeats):
            outputs, gen_s = run_generate()
            saved_images = []
            for out_idx, output in enumerate(outputs):
                image = output.request_output.images[0]
                sample_path = measured_output_path(idx, out_idx)
                image.save(sample_path)
                last_image = image
                if primary_output is None:
                    primary_output = str(sample_path)
                saved_images.append(
                    {
                        "idx": out_idx,
                        "request_id": getattr(output, "request_id", None),
                        "output": str(sample_path),
                        "size": image.size,
                    }
                )
            samples.append(
                {
                    "idx": idx,
                    "gen_s": gen_s,
                    "outputs": saved_images,
                }
            )
        t2 = time.perf_counter()
        measured_gen_s = [sample["gen_s"] for sample in samples]
        warmup_gen_s = [sample["gen_s"] for sample in warmup_samples]
        result.update(
            {
                "ok": True,
                "init_s": t1 - t0,
                "warmup_gen_s": sum(warmup_gen_s),
                "warmup_samples": warmup_samples,
                "gen_s": sum(measured_gen_s),
                "measured_gen_s": sum(measured_gen_s),
                "measured_gen_s_mean": sum(measured_gen_s) / len(measured_gen_s),
                "measured_gen_s_min": min(measured_gen_s),
                "measured_gen_s_max": max(measured_gen_s),
                "gen_samples": samples,
                "total_s": t2 - t0,
                "output": str(output_path),
                "primary_output": primary_output,
                "size": last_image.size if last_image is not None else None,
            }
        )
    except Exception as exc:
        result.update(
            {
                "ok": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if omni is not None:
            try:
                omni.close()
            except Exception:
                pass
        metrics_path = Path(args.metrics)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    main()
