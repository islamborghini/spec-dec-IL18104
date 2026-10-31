"""Cache actual four-method outputs and synchronized emission timestamps on A10."""

import hashlib
import json
from pathlib import Path
import subprocess
import uuid

import modal

from baseline import cache, image as baseline_image
from modal_app import WORKSPACE

ROOT = Path(__file__).resolve().parent
ADAPTER = "/cache/project-runs/hkays15c/continuation-2626-10000/best"
app = modal.App("spec-dec-demo-capture")
image = baseline_image.add_local_python_source("baseline")
PROMPT = (
    "Write a detailed tutorial of about 1,500 words for a computer science student explaining "
    "speculative decoding. Explain autoregressive generation, the small drafter and large target, "
    "block verification, rejection and correction, the bonus token, and KV cache reuse. "
    "Walk through a concrete example with four draft tokens, including a rejection. "
    "Explain why high acceptance does not guarantee higher tokens per second, how draft length "
    "changes the tradeoff, and how to make a fair benchmark. Finish with practical limitations. "
    "Use clear paragraphs and descriptive section headings. Give the tutorial directly."
)


@app.function(image=image, cpu=(1, 1), memory=(4096, 4096), timeout=120,
              startup_timeout=120, retries=0, max_containers=1,
              scaledown_window=2, single_use_containers=True)
def check_capture():
    import sys
    import unittest

    sys.path.insert(0, "/project")
    suite = unittest.defaultTestLoader.discover("/project", pattern="test_decoding.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("Decoder and streaming callback checks must pass without skips")
    return {"passed": result.testsRun}


@app.function(image=image, gpu="A10", cpu=(2, 2), memory=(24576, 24576),
              volumes={"/cache": cache}, timeout=1200, startup_timeout=120,
              retries=0, max_containers=1, scaledown_window=2, single_use_containers=True)
def capture(prompt_text, adapter_sha256, revision, recording_id):
    import importlib.metadata
    import math
    import sys
    import time
    from datetime import datetime, timezone

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from prepare_data import TARGET, TARGET_REVISION, TOKENIZER, TOKENIZER_REVISION

    sys.path.insert(0, "/project")
    from decoding import greedy_decode, speculative_decode

    started = time.monotonic()
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert "A10" in torch.cuda.get_device_name(0)
    cache.reload()
    assert hashlib.sha256(Path(ADAPTER, "adapter_model.safetensors").read_bytes()).hexdigest() == adapter_sha256

    def load(model, pin):
        path = snapshot_download(model, revision=pin, local_files_only=True)
        return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
            device_map="cuda:0", attn_implementation="sdpa", local_files_only=True).eval().requires_grad_(False), path

    target, target_path = load(TARGET, TARGET_REVISION)
    original, draft_path = load(TOKENIZER, TOKENIZER_REVISION)
    trained_base, _ = load(TOKENIZER, TOKENIZER_REVISION)
    trained = PeftModel.from_pretrained(trained_base, ADAPTER, is_trainable=False).merge_and_unload(safe_merge=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=True)
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_path, local_files_only=True)
    assert tokenizer.backend_tokenizer.to_str() == draft_tokenizer.backend_tokenizer.to_str()
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               enable_thinking=False)
    prompt = torch.tensor([prompt_ids], device="cuda")
    eos = target.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])

    def decode(mode, budget, callback=None):
        if mode in ("base", "target"):
            return greedy_decode(original if mode == "base" else target, prompt, budget,
                                 eos_ids, on_tokens=callback)
        return speculative_decode(target, original if mode == "original" else trained,
                                  prompt, budget, 4, eos_ids, on_tokens=callback)

    modes = ["base", "target", "original", "trained"]
    for mode in modes:
        decode(mode, 8)
    result = {"recording_id": recording_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": revision, "prompt": prompt_text, "prompt_ids": prompt_ids,
        "chat_messages": messages, "chat_template": True, "enable_thinking": False,
        "prompt_tokens": len(prompt_ids), "max_new_tokens": 2000, "k": 4,
        "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16", "attention": "sdpa",
        "target": TARGET, "target_revision": TARGET_REVISION,
        "drafter": TOKENIZER, "drafter_revision": TOKENIZER_REVISION,
        "adapter": ADAPTER, "adapter_sha256": adapter_sha256, "selected_step": 9875,
        "test_set_used": False, "execution": "methods recorded sequentially on one GPU, not concurrently",
        "timing": "synchronized emission timestamps include prefill, decoding, and capture callback overhead; excludes loading and warmup",
        "scope": "one new chat-formatted demo prompt; not a replacement for the WikiText validation benchmark",
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")},
        "records": {}}
    output = Path("/cache/demo-recordings") / recording_id
    output.mkdir(parents=True, exist_ok=False)
    for mode in modes:
        events = []
        generated = 0
        torch.cuda.synchronize()
        tick = time.perf_counter()
        last_print = tick

        def emit(tokens):
            nonlocal generated, last_print
            torch.cuda.synchronize()
            now = time.perf_counter()
            generated += len(tokens)
            events.append({"at_s": now - tick, "token_count": generated, "token_ids": list(tokens)})
            if now - last_print > 15:
                print(f"Capture {mode}: {generated}/2000 tokens, {now - tick:.1f}s", flush=True)
                last_print = now
            if time.monotonic() - started > 1110:
                raise TimeoutError("Stopping before the capture GPU limit")

        record = decode(mode, 2000, emit)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - tick
        assert [t for event in events for t in event["token_ids"]] == record["token_ids"]
        assert all(math.isfinite(e["at_s"]) and e["at_s"] >= 0 for e in events)
        assert all(a["at_s"] <= b["at_s"] for a, b in zip(events, events[1:]))
        assert generated == len(record["token_ids"]) and 0 < generated <= 2000
        record.update({"mode": mode, "events": events, "seconds": seconds,
            "tokens_per_second": generated / seconds,
            "text": tokenizer.decode(record["token_ids"], skip_special_tokens=True),
            "stop_reason": "eos" if record["token_ids"][-1] in eos_ids else "token_limit"})
        if mode == "base":
            record["base_forward_calls"] = record.pop("target_forward_calls")
        result["records"][mode] = record
        (output / "recording.json").write_text(json.dumps(result, indent=2) + "\n")
        cache.commit()
        print(f"Capture {mode} complete: {generated} tokens, {seconds:.2f}s, {generated / seconds:.2f} tok/s, {record['stop_reason']}", flush=True)
    target_ids = result["records"]["target"]["token_ids"]
    result["checks"] = {mode: {"matches_target": result["records"][mode]["token_ids"] == target_ids}
                        for mode in ("original", "trained")}
    result["gpu_function_seconds"] = time.monotonic() - started
    result["storage_path"] = str(output)
    (output / "recording.json").write_text(json.dumps(result, indent=2) + "\n")
    cache.commit()
    return result


def main():
    workspace = modal.Workspace.from_context()
    workspace.hydrate()
    assert workspace.name == WORKSPACE
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit capture implementation before recording")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    training = json.loads((ROOT / "reports/training-to-10k-steps.json").read_text())
    recording_id = "demo-" + uuid.uuid4().hex[:12]
    with modal.enable_output(), app.run():
        print("CPU decoder checks:", check_capture.remote(), flush=True)
        result = capture.remote(PROMPT, training["files"]["best/adapter_model.safetensors"], revision, recording_id)
    path = ROOT / "reports/demo-recording.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path, flush=True)


if __name__ == "__main__":
    main()
