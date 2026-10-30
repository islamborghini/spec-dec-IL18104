"""Extend the frozen 500-chunk pilot to 10,000 article-diverse training chunks."""

import importlib.metadata
import json
from pathlib import Path

from prepare_data import (
    DATASET, DATASET_REVISION, SEED, SUBSET, TARGET, TARGET_REVISION,
    TOKENIZER, TOKENIZER_REVISION, WINDOW, articles, digest, normalized,
    sample, shingles, validate,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "wikitext-10k"
COUNT = 10000


def load_pilot():
    root = ROOT / "data" / "pilot"
    manifest = json.loads((root / "manifest.json").read_text())
    splits = {}
    for split in ("validation", "test", "train"):
        content = (root / f"{split}.jsonl").read_text()
        if digest(content) != manifest["file_sha256"][f"{split}.jsonl"]:
            raise ValueError(f"Frozen pilot {split} checksum changed")
        splits[split] = [json.loads(line) for line in content.splitlines()]
    validate(splits)
    return manifest, splits


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    pilot_manifest, pilot = load_pilot()
    data = load_dataset(DATASET, SUBSET, revision=DATASET_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    target_tokenizer = AutoTokenizer.from_pretrained(TARGET, revision=TARGET_REVISION)
    if tokenizer.backend_tokenizer.to_str() != target_tokenizer.backend_tokenizer.to_str():
        raise ValueError("Drafter and target tokenizer backends differ")
    reconstructed = {}
    for split in ("validation", "test", "train"):
        rows = (text for batch in data[split].iter(batch_size=10000) for text in batch["text"])
        reconstructed[split] = articles(rows)
        print(f"Reconstructed {split}: {len(reconstructed[split])} articles", flush=True)
    assert {k: len(v) for k, v in reconstructed.items()} == {"train": 28472, "validation": 60, "test": 60}
    evaluation = reconstructed["validation"] + reconstructed["test"]
    eval_titles = {normalized(a["title"]) for a in evaluation}
    eval_hashes = {a["article_sha256"] for a in evaluation}
    source = [a for a in reconstructed["train"] if normalized(a["title"]) not in eval_titles
              and a["article_sha256"] not in eval_hashes]
    blocked = set()
    for split in ("validation", "test"):
        for row in pilot[split]:
            blocked.update(shingles(row["input_ids"] + row.get("reference_ids", [])))
    print(f"Sampling {COUNT} distinct training articles with seed {SEED}", flush=True)
    train, rejected = sample(source, COUNT, tokenizer, SEED, forbidden=blocked)
    if train[:500] != pilot["train"]:
        raise ValueError("Expanded data must preserve the exact 500-chunk pilot prefix")
    splits = {"validation": pilot["validation"], "test": pilot["test"], "train": train}
    validate(splits, train_count=COUNT)
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in train)
    manifest = {
        key: pilot_manifest[key] for key in (
            "dataset", "dataset_revision", "subset", "tokenizer", "tokenizer_revision",
            "target_tokenizer", "target_tokenizer_revision", "tokenizer_backends_identical",
            "add_special_tokens", "chat_template",
        )
    }
    manifest.update({
        "seed": SEED, "window_tokens": WINDOW, "train_chunks": COUNT,
        "train_tokens": COUNT * WINDOW, "unique_training_articles": len({r["article_title"] for r in train}),
        "equivalent_optimizer_steps_per_epoch_at_effective_batch_4": COUNT // 4,
        "sampling": "one random 256-token window per article, same seeded shuffled source order as pilot",
        "pilot_training_is_exact_prefix": True, "pilot_chunks_retained": 500,
        "additional_chunks": COUNT - 500, "sampling_rejections": rejected,
        "source_article_counts": {k: len(v) for k, v in reconstructed.items()},
        "excluded_training_articles": len(reconstructed["train"]) - len(source),
        "evaluation_files": {"validation": "../pilot/validation.jsonl", "test": "../pilot/test.jsonl"},
        "sample_counts": {"train": COUNT, "validation": 50, "test": 100},
        "file_sha256": {"train.jsonl": digest(content),
                        **{f"{s}.jsonl": pilot_manifest["file_sha256"][f"{s}.jsonl"] for s in ("validation", "test")}},
        "pilot_train_sha256": pilot_manifest["file_sha256"]["train.jsonl"],
        "checks": {"cross_split_article_overlap": 0, "duplicate_windows": 0,
                   "cross_split_shared_32_token_passages": 0, "within_article_window_overlap": 0},
        "versions": {p: importlib.metadata.version(p) for p in ("datasets", "transformers", "tokenizers")},
        "limitations": ["Exact passage checks cover the fixed evaluation windows, not semantic paraphrases.",
                        "Official evaluation articles are excluded by normalized title and whole-article hash.",
                        "Article-balanced Wikipedia sampling does not establish topic balance or code/chat performance.",
                        "No training or final-test evaluation is performed by this CPU preparation step."],
    })
    payloads = {"train.jsonl": content, "manifest.json": json.dumps(manifest, indent=2) + "\n"}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        path = OUTPUT / name
        if path.exists() and path.read_text() != payload:
            raise FileExistsError(f"Refusing to replace a different fixed dataset: {path}")
    # Verify evaluation inputs again before writing; preparation never edits the pilot.
    load_pilot()
    for name, payload in payloads.items():
        (OUTPUT / name).write_text(payload)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
