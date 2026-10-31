"""Build offline playback from actual token IDs and emission timestamps."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def build_events(record, tokenizer):
    ids, previous, result = [], "", []
    last_time = 0
    for event in record["events"]:
        at = event["at_s"]
        if not 0 <= last_time <= at <= record["seconds"]:
            raise ValueError("Emission times must be ordered and inside the capture duration")
        ids.extend(event["token_ids"])
        if event["token_count"] != len(ids):
            raise ValueError("Emission counts must match captured tokens")
        current = tokenizer.decode(ids, skip_special_tokens=True)
        common = 0
        for a, b in zip(previous, current):
            if a != b:
                break
            common += 1
        # JavaScript slices UTF-16 code units, including partially decoded Unicode.
        keep = len(previous[:common].encode("utf-16-le")) // 2
        result.append({"at_s": at, "keep": keep, "append": current[common:],
                       "token_count": len(ids)})
        previous, last_time = current, at
    if not ids or ids != record["token_ids"] or previous != record["text"]:
        raise ValueError("Playback must exactly match recorded output")
    return result


def main():
    from transformers import AutoTokenizer
    recording = json.loads((ROOT / "reports/demo-recording.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(recording["drafter"],
        revision=recording["drafter_revision"], local_files_only=True)
    models = {}
    for mode in ("base", "target", "original", "trained"):
        row = recording["records"][mode]
        models[mode] = {"events": build_events(row, tokenizer),
            "tokens": len(row["token_ids"]), "seconds": row["seconds"],
            "tokens_per_second": row["tokens_per_second"], "stop_reason": row["stop_reason"]}
    payload = {key: recording[key] for key in ("recording_id", "created_utc", "gpu", "dtype",
        "prompt", "prompt_tokens", "max_new_tokens", "k", "selected_step", "checks", "code_revision")}
    payload["models"] = models
    (ROOT / "demo/cached-data.js").write_text("window.CACHED_DEMO = " + json.dumps(payload, ensure_ascii=False) + ";\n")
    print("Built exact recorded-timestamp replay for all four methods.")


if __name__ == "__main__":
    main()
