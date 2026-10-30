"""One-epoch LoRA KL pilot and a paired k=4 validation comparison on one A10."""

import hashlib
import json
from pathlib import Path

import modal

from baseline import ROOT, cache, image as baseline_image, summarize
from modal_app import ENTITY, PROJECT, WORKSPACE, wandb_secret
from prepare_data import TARGET, TARGET_REVISION, TOKENIZER, TOKENIZER_REVISION

app = modal.App("spec-dec-il18104-training")
image = (
    baseline_image.add_local_python_source("baseline", "distillation")
    .add_local_file(ROOT / "data/pilot/train.jsonl", "/project/train.jsonl")
    .add_local_file(ROOT / "tests/test_distillation.py", "/project/test_distillation.py")
)


@app.function(image=image, cpu=(1, 1), memory=(4096, 4096), timeout=120,
              retries=0, max_containers=1, scaledown_window=2, single_use_containers=True)
def check_training():
    import unittest

    suite = unittest.defaultTestLoader.discover("/project", pattern="test_distillation.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("Training checks must pass without skips before GPU allocation")
    return {"passed": result.testsRun}


@app.function(image=image, gpu="A10", cpu=(2, 2), memory=(24576, 24576),
              volumes={"/cache": cache}, secrets=[wandb_secret], timeout=600,
              startup_timeout=300, retries=0, max_containers=1,
              scaledown_window=2, single_use_containers=True)
def train_and_evaluate(code_revision, continuation=None):
    import importlib.metadata
    import random
    import subprocess
    import sys
    import time

    import torch
    import wandb
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, "/project")
    from decoding import greedy_decode, speculative_decode
    from distillation import distillation_loss

    started = time.monotonic()
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    gpu = torch.cuda.get_device_name(0)
    assert "A10" in gpu, gpu
    cache.reload()
    start_step = continuation["start_step"] if continuation else 0
    train_count = continuation["train_chunks"] if continuation else 500
    steps = train_count // 4
    interval = 250 if continuation else 25
    job_limit = 3600 if continuation else 600
    manifest_path = "/project/expanded/manifest.json" if continuation else "/project/manifest.json"
    manifest = json.loads(Path(manifest_path).read_text())
    if continuation:
        assert manifest["file_sha256"]["train.jsonl"] == continuation["data_sha256"]
    datasets = {}
    for split, count in (("train", train_count), ("validation", 50)):
        path = f"/project/expanded/{split}.jsonl" if continuation and split == "train" else f"/project/{split}.jsonl"
        payload = Path(path).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == manifest["file_sha256"][f"{split}.jsonl"]
        rows = [json.loads(line) for line in payload.decode().splitlines()]
        assert len(rows) == count and all(len(row["input_ids"]) == 256 for row in rows)
        datasets[split] = rows
    assert not ({r["article_title"] for r in datasets["train"]} &
                {r["article_title"] for r in datasets["validation"]})
    paths = {"target": snapshot_download(TARGET, revision=TARGET_REVISION, local_files_only=True),
             "drafter": snapshot_download(TOKENIZER, revision=TOKENIZER_REVISION, local_files_only=True)}

    def load_model(path):
        return AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa",
            local_files_only=True,
        ).eval().requires_grad_(False)

    target = load_model(paths["target"])
    student = get_peft_model(load_model(paths["drafter"]), LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM", revision=TOKENIZER_REVISION,
    ))
    student.peft_config["default"].base_model_name_or_path = TOKENIZER
    if continuation:
        checkpoint_path = Path(continuation["checkpoint_dir"])
        for name, expected in continuation["checkpoint_sha256"].items():
            assert hashlib.sha256((checkpoint_path / name).read_bytes()).hexdigest() == expected
        set_peft_model_state_dict(student, load_file(str(checkpoint_path / "adapter_model.safetensors")))
    params = [p for p in student.parameters() if p.requires_grad]
    assert params and all("lora_" in n for n, p in student.named_parameters() if p.requires_grad)
    assert not any(p.requires_grad for p in target.parameters())
    b_probe = next(p for n, p in student.named_parameters() if "lora_B" in n)
    initial_b = b_probe.detach().clone()
    teacher_probe = next(target.parameters()).detach().flatten()[:16].clone()
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
    config = {
        "code_revision": code_revision, "gpu": gpu, "dtype": "bfloat16", "attention": "sdpa",
        "target": TARGET, "target_revision": TARGET_REVISION, "drafter": TOKENIZER,
        "drafter_revision": TOKENIZER_REVISION, "seed": 42, "epochs": 1,
        "train_chunks": train_count, "validation_chunks": 50, "chunk_tokens": 256,
        "microbatch": 1, "gradient_accumulation": 4, "optimizer_steps": steps,
        "starting_optimizer_step": start_step, "ending_optimizer_step": start_step + steps,
        "optimizer_state_restored": False,
        "learning_rate": 1e-4, "weight_decay": 0.0, "max_grad_norm": 1.0,
        "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.0,
        "lora_modules": ["q_proj", "v_proj"], "trainable_parameters": sum(p.numel() for p in params),
        "loss": "mean per-position KL(target || drafter), full vocabulary, temperature 1",
        "positions": "all 256 prefixes per chunk; no padding, labels, chat template, or activation matching",
        "checkpoint_selection": "lowest KL on all 50 validation chunks, including phase-start weights",
        "validation_interval": interval, "test_set_used": False, "k": 4,
        "eval_prompts": 5, "eval_prompt_tokens": 128, "eval_max_new_tokens": 64, "eval_repetitions": 2,
        "timing": "synchronized wall time including prefill and all decode work; both drafters merged/plain",
        "dataset_manifest": manifest,
    }
    result = {"config": config, "train_history": [], "validation_history": [], "records": [],
              "hardware": subprocess.check_output([
                  "nvidia-smi", "--query-gpu=name,driver_version,power.limit,clocks.max.sm",
                  "--format=csv,noheader"], text=True).strip(),
              "versions": {p: importlib.metadata.version(p) for p in
                           ("torch", "transformers", "peft", "wandb")}}

    def validation(model):
        model.eval()
        total, matches = 0.0, 0
        with torch.no_grad():
            for row in datasets["validation"]:
                ids = torch.tensor([row["input_ids"]], device="cuda")
                teacher_logits = target(ids, use_cache=False).logits
                student_logits = model(ids, use_cache=False).logits
                loss = distillation_loss(student_logits, teacher_logits)
                assert torch.isfinite(loss).item()
                total += loss.item()
                matches += (student_logits.argmax(-1) == teacher_logits.argmax(-1)).sum().item()
                del teacher_logits, student_logits, loss
        return {"kl": total / 50, "teacher_forced_top1_agreement": matches / (50 * 256)}

    init_options = ({"id": continuation["run_id"], "resume": "must"} if continuation else
                    {"name": "a10-lora-kl-pilot", "config": config})
    with wandb.init(entity=ENTITY, project=PROJECT,
                    job_type="train-and-evaluate", mode="online", **init_options,
                    settings=wandb.Settings(init_timeout=30)) as run:
        if continuation:
            assert run.id == continuation["run_id"] and run.resumed
            # Preserve the original pilot configuration and mark the larger-data phase explicitly.
            run.config.update({f"continuation_from_step_{start_step}": config})
        run.define_metric("optimizer_step")
        run.define_metric("train/*", step_metric="optimizer_step")
        run.define_metric("validation/*", step_metric="optimizer_step")
        output = Path("/cache/project-runs") / run.id
        if continuation:
            output /= f"continuation-{start_step + 1}-{start_step + steps}"
        output.mkdir(parents=True, exist_ok=False)
        result.update({"wandb_url": run.url, "run_id": run.id, "storage_path": str(output)})
        best_loss, best_step = float("inf"), start_step

        def checkpoint(step):
            nonlocal best_loss, best_step
            metrics = validation(student)
            result["validation_history"].append({"step": step, **metrics})
            run.log({"optimizer_step": step, **{f"validation/{k}": v for k, v in metrics.items()}})
            if metrics["kl"] < best_loss:
                best_loss, best_step = metrics["kl"], step
                student.save_pretrained(output / "best", safe_serialization=True, save_embedding_layers=False)
                cache.commit()
            if continuation:
                student.save_pretrained(output / "last", safe_serialization=True, save_embedding_layers=False)
                torch.save({"optimizer": optimizer.state_dict(), "optimizer_step": step,
                            "phase_updates_completed": step - start_step,
                            "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state()},
                           output / "training_state.pt")
                cache.commit()
            print(f"Validation step={step} KL={metrics['kl']:.6f} agreement={metrics['teacher_forced_top1_agreement']:.4f}", flush=True)
            student.train()

        checkpoint(start_step)
        order = list(range(train_count))
        random.Random(42).shuffle(order)
        assert len(set(order)) == train_count
        torch.cuda.reset_peak_memory_stats()
        for local_step in range(1, steps + 1):
            step = start_step + local_step
            if time.monotonic() - started > job_limit - 150:
                raise TimeoutError("Training stopped to respect the GPU execution limit")
            optimizer.zero_grad(set_to_none=True)
            mean_loss = 0.0
            for index in order[(local_step - 1) * 4:local_step * 4]:
                ids = torch.tensor([datasets["train"][index]["input_ids"]], device="cuda")
                with torch.no_grad():
                    teacher_logits = target(ids, use_cache=False).logits
                student_logits = student(ids, use_cache=False).logits
                loss = distillation_loss(student_logits, teacher_logits)
                assert torch.isfinite(loss).item()
                (loss / 4).backward()
                mean_loss += loss.item() / 4
                del teacher_logits, student_logits, loss
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            assert torch.isfinite(grad_norm).item()
            optimizer.step()
            record = {"step": step, "kl": mean_loss, "gradient_norm": grad_norm.item(), "tokens_seen": step * 1024}
            result["train_history"].append(record)
            run.log({"optimizer_step": step, "train/kl": mean_loss,
                     "train/gradient_norm": record["gradient_norm"], "train/tokens_seen": record["tokens_seen"]})
            if local_step % 25 == 0:
                print(f"Train step={step}/{start_step + steps} batch_KL={mean_loss:.6f}", flush=True)
            if local_step % interval == 0 or local_step == steps:
                checkpoint(step)
        assert not torch.equal(b_probe.detach(), initial_b), "Adapter weights did not change"
        assert torch.equal(next(target.parameters()).detach().flatten()[:16], teacher_probe)
        assert all(p.grad is None for p in target.parameters())
        student.eval().save_pretrained(output / "last", safe_serialization=True, save_embedding_layers=False)
        result.update({"best_step": best_step, "best_validation_kl": best_loss,
                       "training_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                       "adapter_updated": True, "teacher_frozen": True})
        best_weights = load_file(str(output / "best/adapter_model.safetensors"))
        set_peft_model_state_dict(student, best_weights)
        restored = get_peft_model_state_dict(student)
        assert all(torch.equal(restored[name].cpu(), value) for name, value in best_weights.items())
        # Merge removes adapter execution overhead; both drafters then use the same architecture.
        trained = student.eval().merge_and_unload(safe_merge=True).eval()
        del optimizer, params, student, initial_b, b_probe, restored, best_weights
        torch.cuda.empty_cache()
        result["merged_validation"] = validation(trained)
        original = load_model(paths["drafter"])
        tokenizer = AutoTokenizer.from_pretrained(paths["target"], local_files_only=True)
        draft_tokenizer = AutoTokenizer.from_pretrained(paths["drafter"], local_files_only=True)
        assert tokenizer.backend_tokenizer.to_str() == draft_tokenizer.backend_tokenizer.to_str()
        eos = target.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        prompts = [torch.tensor([r["input_ids"][:128]], device="cuda") for r in datasets["validation"][:5]]

        def decode(mode, prompt, budget):
            if mode == "target":
                return greedy_decode(target, prompt, budget, eos_ids)
            return speculative_decode(target, original if mode == "original" else trained, prompt, budget, 4, eos_ids)

        for mode in ("target", "original", "trained"):
            decode(mode, prompts[0], 8)
        torch.cuda.synchronize()
        for repetition in range(2):
            for index, prompt in enumerate(prompts):
                modes = ["target", "original", "trained"]
                rotation = (index + repetition) % 3
                for mode in modes[rotation:] + modes[:rotation]:
                    if time.monotonic() - started > job_limit - 60:
                        raise TimeoutError("Evaluation stopped before the GPU execution limit")
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    record = decode(mode, prompt, 64)
                    torch.cuda.synchronize()
                    record.update({"seconds": time.perf_counter() - tick, "mode": mode,
                                   "prompt_index": index, "repetition": repetition,
                                   "article_title": datasets["validation"][index]["article_title"]})
                    result["records"].append(record)
                    print(f"Eval rep={repetition} prompt={index} mode={mode} seconds={record['seconds']:.3f}", flush=True)
        result["comparison"] = {
            variant: summarize([{**r, "mode": "speculative" if r["mode"] == variant else "target"}
                                for r in result["records"] if r["mode"] in ("target", variant)])
            for variant in ("original", "trained")
        }
        table = wandb.Table(columns=["article", "prompt", "target", "original_drafter", "trained_drafter"])
        for index, prompt in enumerate(prompts):
            pair = {r["mode"]: r for r in result["records"] if r["prompt_index"] == index and r["repetition"] == 0}
            table.add_data(datasets["validation"][index]["article_title"], tokenizer.decode(prompt[0]),
                           *(tokenizer.decode(pair[m]["token_ids"]) for m in ("target", "original", "trained")))
        run.log({"examples": table})
        run.summary.update({"selection/best_step": best_step, "selection/best_validation_kl": best_loss,
                            "validation/merged_kl": result["merged_validation"]["kl"],
                            "checks/adapter_updated": True, "checks/teacher_frozen": True})
        for variant, summary in result["comparison"].items():
            for key in ("tokens_per_second", "acceptance_rate", "generated_tokens_per_verification"):
                run.summary[f"comparison/{variant}/{key}"] = summary["speculative"][key]
            run.summary[f"comparison/{variant}/mismatched_sequences"] = summary["checks"]["mismatched_sequences"]
        run.summary["comparison/target_tokens_per_second"] = result["comparison"]["original"]["target"]["tokens_per_second"]
        for name in ("best", "last"):
            artifact = wandb.Artifact(f"qwen3-06b-wikitext-pilot-{name}", type="model")
            artifact.add_dir(str(output / name))
            run.log_artifact(artifact)
        result["gpu_function_seconds"] = time.monotonic() - started
        result["files"] = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(output.rglob("*")) if p.is_file()}
        report = output / "training.json"
        report.write_text(json.dumps(result, indent=2) + "\n")
        cache.commit()
        artifact = wandb.Artifact("lora-pilot-evaluation", type="evaluation")
        artifact.add_file(str(report))
        run.log_artifact(artifact)
        print(json.dumps({"best_step": best_step, "best_validation_kl": best_loss,
                          "merged_validation": result["merged_validation"], "comparison": result["comparison"]}, indent=2), flush=True)
    return result


def main():
    import subprocess

    workspace = modal.Workspace.from_context()
    workspace.hydrate()
    if workspace.name != WORKSPACE:
        raise RuntimeError(f"Use MODAL_PROFILE={WORKSPACE}")
    wandb_secret.hydrate()
    cache.hydrate()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit the implementation before launching the pilot")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with modal.enable_output(), app.run():
        print("CPU training checks:", check_training.remote(), flush=True)
        result = train_and_evaluate.remote(revision)
    save_result(result)


def save_result(result, name="training"):
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    adapters = output / f"adapters-{result['run_id']}"
    if name != "training":
        adapters /= name
    volume_path = Path(result["storage_path"]).relative_to("/cache")
    for name, expected in result["files"].items():
        path = adapters / name
        assert path.resolve().is_relative_to(adapters.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as destination:
            for chunk in cache.read_file(str(volume_path / name)):
                destination.write(chunk)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    print(json.dumps({"wandb_url": result["wandb_url"], "best_step": result["best_step"],
                      "best_validation_kl": result["best_validation_kl"], "local_adapters": str(adapters)}, indent=2))


if __name__ == "__main__":
    main()
