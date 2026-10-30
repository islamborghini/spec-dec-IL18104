"""Bounded A10 diagnosis: paired checkpoints, validation k sweep, forward timings."""

import json
from pathlib import Path

import modal

from baseline import image as baseline_image, cache
from modal_app import WORKSPACE

ROOT = Path(__file__).resolve().parent
app = modal.App("spec-dec-speed-diagnosis")
image = baseline_image.add_local_python_source("baseline")


@app.function(image=image, gpu="A10", cpu=(2, 2), memory=(24576, 24576),
              volumes={"/cache": cache}, timeout=600, startup_timeout=120,
              retries=0, max_containers=1, scaledown_window=2, single_use_containers=True)
def diagnose():
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
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert "A10" in torch.cuda.get_device_name(0)
    cache.reload()
    payload = Path("/project/validation.jsonl").read_bytes()
    manifest = json.loads(Path("/project/manifest.json").read_text())
    assert hashlib.sha256(payload).hexdigest() == manifest["file_sha256"]["validation.jsonl"]
    rows = [json.loads(line) for line in payload.decode().splitlines()][:5]
    prompts = [torch.tensor([row["input_ids"][:128]], device="cuda") for row in rows]

    def load(model, revision):
        return AutoModelForCausalLM.from_pretrained(
            snapshot_download(model, revision=revision, local_files_only=True),
            dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa",
            local_files_only=True).eval().requires_grad_(False)

    target = load(TARGET, TARGET_REVISION)
    original = load(TOKENIZER, TOKENIZER_REVISION)
    models = {"original": original}
    for name, path in (("pilot", "/cache/project-runs/hkays15c/best"),
                       ("trained10k", "/cache/project-runs/hkays15c/continuation-126-2625/best")):
        models[name] = PeftModel.from_pretrained(load(TOKENIZER, TOKENIZER_REVISION), path,
                                                is_trainable=False).merge_and_unload(safe_merge=True).eval()
    eos = target.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    result = {"gpu": torch.cuda.get_device_name(0), "test_set_used": False,
              "prompt_tokens": 128, "new_tokens": 64, "records": [], "k_records": [],
              "layers": {"target": target.config.num_hidden_layers,
                         "drafter": original.config.num_hidden_layers},
              "checkpoint_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                  for name, path in (("pilot", "/cache/project-runs/hkays15c/best/adapter_model.safetensors"),
                                     ("trained10k", "/cache/project-runs/hkays15c/continuation-126-2625/best/adapter_model.safetensors"))}}

    def decode(mode, prompt, k=4, budget=64):
        if mode == "target":
            return greedy_decode(target, prompt, budget, eos_ids)
        return speculative_decode(target, models[mode], prompt, budget, k, eos_ids)

    def timed(mode, index, repetition, k=4):
        assert time.monotonic() - started < 510, "Diagnostic execution limit"
        torch.cuda.synchronize()
        tick = time.perf_counter()
        record = decode(mode, prompts[index], k)
        torch.cuda.synchronize()
        record.update(mode=mode, prompt_index=index, repetition=repetition,
                      k=k, seconds=time.perf_counter() - tick)
        return record

    for mode in ("target", *models):
        decode(mode, prompts[0], budget=8)
    for rep in range(3):
        for index in range(5):
            modes = ["target", "original", "pilot", "trained10k"]
            shift = (rep + index) % len(modes)
            for mode in modes[shift:] + modes[:shift]:
                record = timed(mode, index, rep)
                result["records"].append(record)
            print(f"Paired timing rep={rep} prompt={index} complete", flush=True)

    # This small validation sweep is diagnostic, not a held-out performance claim.
    for rep in range(2):
        for index in range(3):
            ks = [1, 2, 4, 8]
            shift = (rep + index) % len(ks)
            for k in ks[shift:] + ks[:shift]:
                result["k_records"].append(timed("trained10k", index, rep, k))
            print(f"k sweep rep={rep} prompt={index} complete", flush=True)

    # CUDA events bracket forward calls without synchronizing each call. These
    # are stream elapsed times (including launch gaps), not pure kernel durations.
    events = []
    pending = {}
    handles = []

    def before(role):
        def hook(module, args, kwargs):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            pending[role] = (event, kwargs["input_ids"].shape[1])
        return hook

    def after(role):
        def hook(module, args, output):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            begin, tokens = pending.pop(role)
            events.append((role, tokens, begin, event))
        return hook

    for role, model in (("target", target), ("drafter", models["trained10k"])):
        handles.extend([model.register_forward_pre_hook(before(role), with_kwargs=True),
                        model.register_forward_hook(after(role))])
    profiles = []
    try:
        for index in range(3):
            events.clear()
            row = timed("trained10k", index, 0)
            groups = {}
            for role, tokens, begin, end in events:
                key = f"{role}/{'prefill' if tokens > 8 else 'decode'}"
                item = groups.setdefault(key, {"calls": 0, "stream_ms": 0.0})
                item["calls"] += 1
                item["stream_ms"] += begin.elapsed_time(end)
            row["forward_groups"] = groups
            profiles.append(row)
    finally:
        for handle in handles:
            handle.remove()
    result["profiles"] = profiles
    result["gpu_function_seconds"] = time.monotonic() - started
    assert len(result["records"]) == 60 and len(result["k_records"]) == 24
    assert all(len(row["token_ids"]) == 64 for row in result["records"] + result["k_records"])
    print(f"Diagnosis finished in {result['gpu_function_seconds']:.1f}s", flush=True)
    return result


def main():
    workspace = modal.Workspace.from_context()
    workspace.hydrate()
    assert workspace.name == WORKSPACE
    with modal.enable_output(), app.run():
        result = diagnose.remote()
    path = ROOT / "results/speed-diagnosis.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
