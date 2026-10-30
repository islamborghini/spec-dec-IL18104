"""Replay recorded validation divergences at identical prefixes, without test data."""

import json
from pathlib import Path

import modal

from baseline import app, cache, image, ROOT, WORKSPACE
from modal_app import ENTITY, PROJECT, wandb_secret


@app.function(image=image.add_local_python_source("baseline"), gpu="A10", cpu=(2, 2), memory=(24576, 24576),
              volumes={"/cache": cache}, timeout=300, startup_timeout=300,
              retries=0, max_containers=1, scaledown_window=2, single_use_containers=True)
def diagnose(records):
    import sys
    import time

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    from prepare_data import TARGET, TARGET_REVISION, TOKENIZER, TOKENIZER_REVISION

    sys.path.insert(0, "/project")
    from decoding import greedy_decode, speculative_decode

    started = time.monotonic()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    cache.reload()
    target = AutoModelForCausalLM.from_pretrained(
        snapshot_download(TARGET, revision=TARGET_REVISION, local_files_only=True),
        dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa", local_files_only=True,
    ).eval().requires_grad_(False)
    drafter = AutoModelForCausalLM.from_pretrained(
        snapshot_download(TOKENIZER, revision=TOKENIZER_REVISION, local_files_only=True),
        dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa", local_files_only=True,
    ).eval().requires_grad_(False)
    eos = target.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    rows = [json.loads(line) for line in Path("/project/validation.jsonl").read_text().splitlines()][:5]
    evidence = []

    class TraceTarget:
        config = target.config

        def __init__(self, expected, prompt, position):
            self.expected_prefix = prompt + expected[:position]
            self.position = position
            self.prompt_length = len(prompt)
            self.history, self.logits = [], None

        def __call__(self, **kwargs):
            past = kwargs.get("past_key_values")
            size = past.get_seq_length() if past is not None else 0
            self.history = self.history[:size] + kwargs["input_ids"][0].tolist()
            out = target(**kwargs)
            first_position = len(self.history) - out.logits.shape[1] + 1 - self.prompt_length
            index = self.position - first_position
            prefix_length = self.prompt_length + self.position
            if 0 <= index < out.logits.shape[1] and self.history[:prefix_length] == self.expected_prefix:
                self.logits = out.logits[0, index].float().cpu()
            return out

    for index, row in enumerate(rows):
        pair = {r["mode"]: r["token_ids"] for r in records if r["prompt_index"] == index and r["repetition"] == 0}
        differences = [j for j, (a, b) in enumerate(zip(pair["target"], pair["speculative"])) if a != b]
        if not differences:
            continue
        position = differences[0]
        prompt_ids = row["input_ids"][:128]
        prompt = torch.tensor([prompt_ids], device="cuda")
        baseline_trace = TraceTarget(pair["target"], prompt_ids, position)
        spec_trace = TraceTarget(pair["target"], prompt_ids, position)
        a = greedy_decode(baseline_trace, prompt, 64, eos_ids)["token_ids"]
        b = speculative_decode(spec_trace, drafter, prompt, 64, 4, eos_ids)["token_ids"]
        assert a == pair["target"] and b == pair["speculative"], "Recorded divergence did not reproduce"
        assert baseline_trace.logits is not None and spec_trace.logits is not None, "Cache prefix did not match"
        greedy_logits, spec_logits = baseline_trace.logits, spec_trace.logits
        greedy_values, greedy_ids = greedy_logits.topk(2)
        spec_values, spec_ids = spec_logits.topk(2)
        item = {"prompt_index": index, "first_divergence_position": position,
                "identical_prefix": True, "target_top2_ids": greedy_ids.tolist(),
                "target_top2_logits": greedy_values.tolist(), "spec_top2_ids": spec_ids.tolist(),
                "spec_top2_logits": spec_values.tolist(),
                "max_abs_logit_difference": (greedy_logits - spec_logits).abs().max().item(),
                "mean_abs_logit_difference": (greedy_logits - spec_logits).abs().mean().item()}
        evidence.append(item)
        print(json.dumps(item), flush=True)

    # Change target precision only; drafter stays BF16. This tests the rounding hypothesis.
    target.float()
    torch.cuda.empty_cache()
    fp32_checks = []
    for index, row in enumerate(rows):
        prompt = torch.tensor([row["input_ids"][:128]], device="cuda")
        a = greedy_decode(target, prompt, 64, eos_ids)["token_ids"]
        b = speculative_decode(target, drafter, prompt, 64, 4, eos_ids)["token_ids"]
        fp32_checks.append({"prompt_index": index, "matches": a == b})
        print("FP32 target check", fp32_checks[-1], flush=True)
    result = {"bf16_identical_prefix_evidence": evidence,
              "fp32_target_bf16_drafter_checks": fp32_checks,
              "seconds": time.monotonic() - started,
              "scope": "validation only; untimed diagnostic, not a speed benchmark"}
    return result


@app.function(image=image.add_local_python_source("baseline"), cpu=(1, 1), memory=(2048, 2048),
              secrets=[wandb_secret], timeout=90, retries=0, max_containers=1,
              scaledown_window=2, single_use_containers=True)
def log_diagnostic(result, baseline):
    import wandb

    checks = result["fp32_target_bf16_drafter_checks"]
    run_id = baseline["wandb_url"].rsplit("/", 1)[-1]
    with wandb.init(entity=ENTITY, project=PROJECT, id=run_id, resume="must", mode="online",
                    settings=wandb.Settings(init_timeout=30)) as run:
        for mode in ("target", "speculative"):
            run.summary[f"baseline/{mode}_tokens_per_second"] = baseline["summary"][mode]["tokens_per_second"]
        run.summary["numerics/fp32_checked_sequences"] = len(checks)
        run.summary["numerics/fp32_mismatched_sequences"] = sum(not row["matches"] for row in checks)
        run.summary["numerics/identical_prefix_divergences"] = len(result["bf16_identical_prefix_evidence"])
        run.summary["numerics/note"] = "BF16 ties differ at identical prefixes; FP32 check is untimed and separate from BF16 timings."
        path = Path("/tmp/numerics.json")
        path.write_text(json.dumps(result, indent=2) + "\n")
        artifact = wandb.Artifact("k4-numerical-diagnostic", type="diagnostics")
        artifact.add_file(str(path))
        run.log_artifact(artifact)
        return run.url


def main():
    workspace = modal.Workspace.from_context()
    workspace.hydrate()
    if workspace.name != WORKSPACE:
        raise RuntimeError(f"Use MODAL_PROFILE={WORKSPACE}")
    baseline = json.loads((ROOT / "results/baseline.json").read_text())
    records = baseline["records"]
    wandb_secret.hydrate()
    with modal.enable_output(), app.run():
        result = diagnose.remote(records)
        log_diagnostic.remote(result, baseline)
    (ROOT / "results/numerics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
