"""Validation k sweep for the update-9,875 drafter: k in 1/2/4/8 vs target-only, same A10 job."""

import json
from pathlib import Path

import modal

from baseline import image as baseline_image, cache, summarize
from modal_app import WORKSPACE

ROOT = Path(__file__).resolve().parent
ADAPTER = "/cache/project-runs/hkays15c/continuation-2626-10000/best"
KS = [1, 2, 4, 8]
app = modal.App("spec-dec-k-sweep")
image = baseline_image.add_local_python_source("baseline")


@app.function(image=image, gpu="A10", cpu=(2, 2), memory=(24576, 24576),
              volumes={"/cache": cache}, timeout=1200, startup_timeout=120,
              retries=0, max_containers=1, scaledown_window=2, single_use_containers=True)
def sweep(adapter_sha256):
    import hashlib
    import sys
    import time

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from prepare_data import TARGET, TARGET_REVISION, TOKENIZER, TOKENIZER_REVISION

    sys.path.insert(0, "/project")
    from decoding import greedy_decode, speculative_decode

    started = time.monotonic()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert "A10" in torch.cuda.get_device_name(0)
    cache.reload()
    assert hashlib.sha256(Path(ADAPTER, "adapter_model.safetensors").read_bytes()).hexdigest() == adapter_sha256
    payload = Path("/project/validation.jsonl").read_bytes()
    manifest = json.loads(Path("/project/manifest.json").read_text())
    assert hashlib.sha256(payload).hexdigest() == manifest["file_sha256"]["validation.jsonl"]
    # Same first-20 validation prompts as the 10k comparison; test set stays untouched.
    rows = [json.loads(line) for line in payload.decode().splitlines()][:20]
    prompts = [torch.tensor([row["input_ids"][:128]], device="cuda") for row in rows]

    def load(model, revision):
        return AutoModelForCausalLM.from_pretrained(
            snapshot_download(model, revision=revision, local_files_only=True),
            dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa",
            local_files_only=True).eval().requires_grad_(False)

    target = load(TARGET, TARGET_REVISION)
    drafter = PeftModel.from_pretrained(load(TOKENIZER, TOKENIZER_REVISION), ADAPTER,
                                        is_trainable=False).merge_and_unload(safe_merge=True).eval()
    eos = target.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])

    def timed(mode, index, rep, budget=64):
        assert time.monotonic() - started < 1100, "Sweep execution limit"
        torch.cuda.synchronize()
        tick = time.perf_counter()
        record = (greedy_decode(target, prompts[index], budget, eos_ids) if mode == "target" else
                  speculative_decode(target, drafter, prompts[index], budget, mode, eos_ids))
        torch.cuda.synchronize()
        record.update(mode=mode, prompt_index=index, repetition=rep, seconds=time.perf_counter() - tick)
        return record

    for mode in ("target", *KS):
        timed(mode, 0, -1, budget=8)
    records = []
    modes = ["target", *KS]
    for rep in range(2):
        for index in range(20):
            shift = (rep + index) % len(modes)  # rotate order so no setting always runs first
            for mode in modes[shift:] + modes[:shift]:
                records.append(timed(mode, index, rep))
        print(f"k sweep rep={rep} complete", flush=True)
    target_rows = [r for r in records if r["mode"] == "target"]
    summary = {k: summarize(target_rows + [{**r, "mode": "speculative"} for r in records if r["mode"] == k])
               for k in KS}
    return {"gpu": torch.cuda.get_device_name(0), "adapter": ADAPTER, "adapter_sha256": adapter_sha256,
            "test_set_used": False, "prompts": 20, "repetitions": 2, "prompt_tokens": 128,
            "new_tokens": 64, "summary": summary, "records": records,
            "gpu_function_seconds": time.monotonic() - started}


def main():
    workspace = modal.Workspace.from_context()
    workspace.hydrate()
    assert workspace.name == WORKSPACE
    report = json.loads((ROOT / "reports/training-to-10k-steps.json").read_text())
    with modal.enable_output(), app.run():
        result = sweep.remote(report["files"]["best/adapter_model.safetensors"])
    (ROOT / "reports/k-sweep.json").write_text(json.dumps(result, indent=2) + "\n")
    for k, s in result["summary"].items():
        spec = s["speculative"]
        print(f"k={k}: {spec['tokens_per_second']:.2f} tok/s, acceptance {spec['acceptance_rate']:.1%}, "
              f"vs target {s['throughput_ratio']:.3f}x")
    print(f"target-only: {result['summary'][KS[0]]['target']['tokens_per_second']:.2f} tok/s")


if __name__ == "__main__":
    main()
