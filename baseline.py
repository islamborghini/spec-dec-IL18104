"""Run a small k=4 validation baseline on Modal; never opens the test set."""

import hashlib
import json
from pathlib import Path

import modal

from modal_app import ENTITY, PROJECT, WORKSPACE, image as setup_image, wandb_secret
from prepare_data import DATASET_REVISION, TARGET, TARGET_REVISION, TOKENIZER, TOKENIZER_REVISION

ROOT = Path(__file__).resolve().parent
app = modal.App("spec-dec-il18104-baseline")
cache = modal.Volume.from_name("spec-dec-benchmark-model-cache")
image = (
    setup_image.pip_install(
        "transformers==4.57.3", "tokenizers==0.22.2", "accelerate==1.11.0",
        "peft==0.17.1", "huggingface-hub==0.36.0", "safetensors==0.6.2",
    )
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("modal_app", "prepare_data")
    .add_local_file(ROOT / "decoding.py", "/project/decoding.py")
    .add_local_file(ROOT / "tests/test_decoding.py", "/project/test_decoding.py")
    .add_local_file(ROOT / "data/pilot/validation.jsonl", "/project/validation.jsonl")
    .add_local_file(ROOT / "data/pilot/manifest.json", "/project/manifest.json")
)


@app.function(image=image, cpu=(1, 1), memory=(4096, 4096), timeout=120,
              retries=0, max_containers=1, scaledown_window=2, single_use_containers=True)
def check_decoders():
    import sys
    import unittest

    sys.path.insert(0, "/project")
    suite = unittest.defaultTestLoader.discover("/project", pattern="test_decoding.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("Real-model decoder checks must all pass without skips")
    return {"passed": result.testsRun}


def summarize(records):
    totals = {}
    for mode in ("target", "speculative"):
        rows = [r for r in records if r["mode"] == mode]
        tokens = sum(len(r["token_ids"]) for r in rows)
        seconds = sum(r["seconds"] for r in rows)
        totals[mode] = {"generated_tokens": tokens, "seconds": seconds,
                        "tokens_per_second": tokens / seconds}
        if mode == "speculative":
            accepted = sum(r["accepted_draft_tokens"] for r in rows)
            proposed = sum(r["proposed_draft_tokens"] for r in rows)
            passes = sum(r["target_verification_passes"] for r in rows)
            totals[mode].update({"accepted_draft_tokens": accepted,
                                 "proposed_draft_tokens": proposed,
                                 "acceptance_rate": accepted / proposed,
                                 "generated_tokens_per_verification": tokens / passes,
                                 "accepted_draft_tokens_per_verification": accepted / passes})
    pairs = {}
    for record in records:
        pairs.setdefault((record["prompt_index"], record["repetition"]), {})[record["mode"]] = record
    mismatches, mismatched_tokens, positions = 0, 0, 0
    for pair in pairs.values():
        target, spec = pair["target"]["token_ids"], pair["speculative"]["token_ids"]
        mismatches += target != spec
        positions += max(len(target), len(spec))
        mismatched_tokens += sum((target[i] if i < len(target) else None) !=
                                 (spec[i] if i < len(spec) else None)
                                 for i in range(max(len(target), len(spec))))
    totals["checks"] = {"paired_sequences": len(pairs), "mismatched_sequences": mismatches,
                        "mismatched_positions": mismatched_tokens,
                        "position_mismatch_rate": mismatched_tokens / positions}
    ratio = totals["speculative"]["tokens_per_second"] / totals["target"]["tokens_per_second"]
    totals["throughput_ratio"] = ratio
    totals["verified_speedup"] = ratio if mismatches == 0 else None
    return totals


@app.function(image=image, gpu="A10", cpu=(2, 2), memory=(24576, 24576),
              volumes={"/cache": cache}, secrets=[wandb_secret], timeout=600,
              startup_timeout=300, retries=0, max_containers=1,
              scaledown_window=2, single_use_containers=True)
def run_baseline(code_revision):
    import importlib.metadata
    import subprocess
    import sys
    import time

    import torch
    import wandb
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/project")
    from decoding import greedy_decode, speculative_decode

    started = time.monotonic()
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    gpu = torch.cuda.get_device_name(0)
    assert "A10" in gpu, gpu
    cache.reload()
    manifest = json.loads(Path("/project/manifest.json").read_text())
    payload = Path("/project/validation.jsonl").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == manifest["file_sha256"]["validation.jsonl"]
    rows = [json.loads(line) for line in payload.decode().splitlines()][:5]
    assert len(rows) == 5 and len({r["article_title"] for r in rows}) == 5
    assert manifest["dataset_revision"] == DATASET_REVISION
    assert manifest["tokenizer_revision"] == TOKENIZER_REVISION
    model_specs = {"target": (TARGET, TARGET_REVISION), "drafter": (TOKENIZER, TOKENIZER_REVISION)}
    models, paths = {}, {}
    for role, (model_id, revision) in model_specs.items():
        # These pinned weights were staged by the earlier hardware benchmark on CPU.
        paths[role] = snapshot_download(model_id, revision=revision, local_files_only=True)
        models[role] = AutoModelForCausalLM.from_pretrained(
            paths[role], torch_dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="sdpa", local_files_only=True,
        ).eval().requires_grad_(False)
        print(f"Loaded {role}: {model_id}@{revision}", flush=True)
    target, drafter = models["target"], models["drafter"]
    tokenizer = AutoTokenizer.from_pretrained(paths["target"], local_files_only=True)
    draft_tokenizer = AutoTokenizer.from_pretrained(paths["drafter"], local_files_only=True)
    assert tokenizer.backend_tokenizer.to_str() == draft_tokenizer.backend_tokenizer.to_str()
    eos = target.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    prompts = [torch.tensor([r["input_ids"][:128]], device="cuda") for r in rows]
    assert all(p.shape == (1, 128) for p in prompts)
    config = {
        "code_revision": code_revision, "gpu": gpu, "dtype": "bfloat16", "attention": "sdpa",
        "batch_size": 1, "k": 4, "max_new_tokens": 64, "prompt_tokens": 128,
        "prompts": 5, "repetitions": 2, "seed": 42, "split": "validation",
        "validation_sha256": manifest["file_sha256"]["validation.jsonl"],
        "models": {role: {"id": spec[0], "revision": spec[1]} for role, spec in model_specs.items()},
        "timing": "synchronized wall time including KV prefill, both models, verification and rollback; excludes loading/tokenization",
        "eos_ids": sorted(eos_ids), "greedy": True, "test_set_used": False,
    }
    result = {"config": config, "records": [],
              "hardware": subprocess.check_output([
                  "nvidia-smi", "--query-gpu=name,driver_version,power.limit,clocks.max.sm",
                  "--format=csv,noheader"], text=True).strip(),
              "versions": {p: importlib.metadata.version(p) for p in
                           ("torch", "transformers", "tokenizers", "modal", "wandb")},
              "cuda": torch.version.cuda}

    def decode(mode, prompt, budget):
        if mode == "target":
            return greedy_decode(target, prompt, budget, eos_ids)
        return speculative_decode(target, drafter, prompt, budget, 4, eos_ids)

    with wandb.init(entity=ENTITY, project=PROJECT, name="a10-k4-validation-baseline",
                    job_type="baseline", mode="online", config=config,
                    settings=wandb.Settings(init_timeout=30)) as run:
        result["wandb_url"] = run.url
        # Check our target loop against Transformers on the actual checkpoint, untimed.
        reference = target.generate(prompts[0], max_new_tokens=16, do_sample=False,
                                    temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                                    eos_token_id=sorted(eos_ids), pad_token_id=tokenizer.pad_token_id,
                                    use_cache=True)[0, 128:].tolist()
        assert decode("target", prompts[0], 16)["token_ids"] == reference
        for mode in ("target", "speculative"):
            decode(mode, prompts[0], 8)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for repetition in range(2):
            for index, prompt in enumerate(prompts):
                modes = ("target", "speculative") if (index + repetition) % 2 == 0 else ("speculative", "target")
                for mode in modes:
                    if time.monotonic() - started > 510:
                        raise TimeoutError("Stopping before the ten-minute GPU execution limit")
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    record = decode(mode, prompt, 64)
                    torch.cuda.synchronize()
                    record.update({"seconds": time.perf_counter() - tick, "mode": mode,
                                   "prompt_index": index, "repetition": repetition,
                                   "article_title": rows[index]["article_title"]})
                    result["records"].append(record)
                    run.log({f"timing/{mode}_tokens_per_second": len(record["token_ids"]) / record["seconds"]})
                    print(f"rep={repetition} prompt={index} mode={mode} tokens={len(record['token_ids'])} seconds={record['seconds']:.3f}", flush=True)
        result["summary"] = summarize(result["records"])
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        table = wandb.Table(columns=["article", "prompt", "target", "speculative", "tokens_match"])
        for index, prompt in enumerate(prompts):
            pair = {r["mode"]: r for r in result["records"] if r["prompt_index"] == index and r["repetition"] == 0}
            table.add_data(rows[index]["article_title"], tokenizer.decode(prompt[0]),
                           tokenizer.decode(pair["target"]["token_ids"]),
                           tokenizer.decode(pair["speculative"]["token_ids"]),
                           pair["target"]["token_ids"] == pair["speculative"]["token_ids"])
        run.log({"examples": table, "gpu/peak_allocated_gib": result["peak_allocated_gib"],
                 "baseline/throughput_ratio": result["summary"]["throughput_ratio"],
                 "checks/mismatched_sequences": result["summary"]["checks"]["mismatched_sequences"],
                 "speculative/acceptance_rate": result["summary"]["speculative"]["acceptance_rate"],
                 "speculative/generated_tokens_per_verification": result["summary"]["speculative"]["generated_tokens_per_verification"]})
        result["gpu_function_seconds"] = time.monotonic() - started
        # Resource estimate only; the Modal billing page is authoritative.
        result["estimated_compute_usd"] = result["gpu_function_seconds"] * (0.000306 + 2 * 0.0000131 + 24 * 0.00000222)
        report = Path("/tmp/baseline.json")
        report.write_text(json.dumps(result, indent=2) + "\n")
        artifact = wandb.Artifact("k4-validation-baseline", type="evaluation")
        artifact.add_file(str(report))
        run.log_artifact(artifact)
    print(json.dumps(result["summary"], indent=2), flush=True)
    return result


def main():
    import subprocess

    workspace = modal.Workspace.from_context()
    workspace.hydrate()
    if workspace.name != WORKSPACE:
        raise RuntimeError(f"Expected workspace {WORKSPACE}; use MODAL_PROFILE={WORKSPACE}")
    wandb_secret.hydrate()
    cache.hydrate()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit the code first so the run identifies the exact implementation")
    with modal.enable_output(), app.run():
        print("CPU decoder checks:", check_decoders.remote(), flush=True)
        result = run_baseline.remote(revision)
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    (output / "baseline.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"wandb_url": result["wandb_url"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
