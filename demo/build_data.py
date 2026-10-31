"""Create offline replay assets; no inference, credentials, or cloud requests."""

import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent


def main():
    report = json.loads((ROOT / "reports/training-to-10k-steps.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(report["config"]["drafter"],
        revision=report["config"]["drafter_revision"], local_files_only=True)
    text = (ROOT / "demo/illustrative-answer.md").read_text()
    ids = tokenizer.encode(text, add_special_tokens=False)[:2000]
    assert len(ids) == 2000, "Write enough illustrative text for the requested playback"
    pieces = [tokenizer.decode([token]) for token in ids]
    assert "".join(pieces) == tokenizer.decode(ids), "Single-token pieces must preserve text"
    rows = [json.loads(line) for line in (ROOT / "data/pilot/validation.jsonl").read_text().splitlines()]
    recorded = {}
    for mode in ("target", "original", "trained"):
        row = next(r for r in report["records"] if r["mode"] == mode
                   and r["prompt_index"] == 0 and r["repetition"] == 0)
        recorded[mode] = [tokenizer.decode(row["token_ids"][:n]) for n in range(len(row["token_ids"]) + 1)]
    data = {"illustrativePieces": pieces,
        "illustrativePrompt": "Explain speculative decoding, why a smaller drafter may fail to accelerate a target model, and what our experiments show.",
        "recordedPrompt": tokenizer.decode(rows[0]["input_ids"][:128]),
        "recordedArticle": rows[0]["article_title"], "recorded": recorded,
        "rates": {"base": 45,
            "target": report["comparison"]["original"]["target"]["tokens_per_second"],
            "original": report["comparison"]["original"]["speculative"]["tokens_per_second"],
            "trained": report["comparison"]["trained"]["speculative"]["tokens_per_second"]},
        "source": "reports/training-to-10k-steps.json", "selectedStep": report["best_step"]}
    (ROOT / "demo/data.js").write_text("window.REPLAY_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n")
    print("Prepared 2,000-token illustrative text and three real 64-token recorded outputs.")


if __name__ == "__main__":
    main()
