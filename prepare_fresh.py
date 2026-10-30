"""Prepare 29,500 fresh random windows for optimizer updates 2,626–10,000."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import struct

from prepare_10k import load_pilot
from prepare_data import (DATASET, DATASET_REVISION, SUBSET, TOKENIZER,
                          TOKENIZER_REVISION, TARGET, TARGET_REVISION,
                          WINDOW, SHINGLE, articles, digest, normalized)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/wikitext-fresh-to-10k-steps"
COUNT = 29500
SEED = 20261029


def passage_hashes(ids):
    """Fixed-width token packing avoids ambiguous token concatenations."""
    packed = struct.pack(f"<{len(ids)}I", *ids)
    return {hashlib.blake2b(packed[i * 4:(i + SHINGLE) * 4], digest_size=16).digest()
            for i in range(len(ids) - SHINGLE + 1)}


def fresh_starts(length, used, rng):
    """Random-offset tiling of unused gaps; neither old nor new windows overlap."""
    previous = 0
    starts = []
    for left, right in sorted(used) + [(length, length)]:
        if not 0 <= previous <= left <= right <= length:
            raise ValueError("Invalid or overlapping prior source intervals")
        available = left - previous
        if available >= WINDOW:
            offset = rng.randint(0, min(WINDOW - 1, available - WINDOW))
            starts.extend(range(previous + offset, left - WINDOW + 1, WINDOW))
        previous = right
    rng.shuffle(starts)
    return starts


def select_fresh(source, tokenizer, prior, forbidden, count, seed):
    rng = random.Random(seed)
    order = list(source)
    rng.shuffle(order)
    spans, prior_passages = {}, set()
    for row in prior:
        spans.setdefault(row["article_sha256"], set()).add((row["token_start"], row["token_end"]))
        prior_passages.update(passage_hashes(row["input_ids"]))
    blocked = set(forbidden) | prior_passages
    titles, hashes, pending = set(), set(), []
    for index, article in enumerate(order):
        title = normalized(article["title"])
        if title in titles or article["article_sha256"] in hashes:
            continue
        titles.add(title)
        hashes.add(article["article_sha256"])
        ids = tokenizer.encode(article["text"], add_special_tokens=False)
        starts = fresh_starts(len(ids), spans.get(article["article_sha256"], ()), rng)
        if starts:
            pending.append((article, ids, starts))
        if index and index % 2000 == 0:
            print(f"Tokenized {index}/{len(order)} training articles", flush=True)
    selected, seen_windows, rejected = [], set(), 0
    # One window per shuffled article per round gives short articles a chance.
    while pending:
        rng.shuffle(pending)
        following = []
        for article, ids, starts in pending:
            while starts:
                start = starts.pop()
                window = ids[start:start + WINDOW]
                window_hash = digest(json.dumps(window, separators=(",", ":")))
                pieces = passage_hashes(window)
                if window_hash in seen_windows or pieces & blocked:
                    rejected += 1
                    continue
                selected.append({"article_title": article["title"],
                    "source_row_start": article["row_start"], "source_row_end": article["row_end"],
                    "article_sha256": article["article_sha256"], "token_start": start,
                    "token_end": start + WINDOW, "window_sha256": window_hash,
                    "input_ids": window, "text": tokenizer.decode(window)})
                seen_windows.add(window_hash)
                blocked.update(pieces)
                break
            if starts:
                following.append((article, ids, starts))
            if len(selected) == count:
                rng.shuffle(selected)
                return selected, rejected
        print(f"Selected {len(selected)}/{count} fresh chunks", flush=True)
        pending = following
    raise ValueError(f"Only {len(selected)} clean fresh chunks available; need {count}")


def audit_fresh(rows, prior, forbidden, eval_titles, eval_hashes, count):
    assert len(rows) == count
    blocked, windows, spans = set(forbidden), set(), {}
    for row in prior:
        blocked.update(passage_hashes(row["input_ids"]))
        spans.setdefault(row["article_sha256"], set()).add((row["token_start"], row["token_end"]))
    for row in rows:
        ids = row["input_ids"]
        assert len(ids) == WINDOW and all(type(t) is int and t >= 0 for t in ids)
        assert normalized(row["article_title"]) not in eval_titles
        assert row["article_sha256"] not in eval_hashes
        start, end = row["token_start"], row["token_end"]
        assert start >= 0 and end - start == WINDOW
        assert all(end <= a or start >= b for a, b in spans.get(row["article_sha256"], ())), "Reused source tokens"
        expected = digest(json.dumps(ids, separators=(",", ":")))
        assert expected == row["window_sha256"] and expected not in windows
        pieces = passage_hashes(ids)
        assert not pieces & blocked, "Repeated training or held-out passage"
        blocked.update(pieces)
        windows.add(expected)
        spans.setdefault(row["article_sha256"], set()).add((start, end))


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    pilot_manifest, pilot = load_pilot()
    previous_manifest = json.loads((ROOT / "data/wikitext-10k/manifest.json").read_text())
    previous_text = (ROOT / "data/wikitext-10k/train.jsonl").read_text()
    assert digest(previous_text) == previous_manifest["file_sha256"]["train.jsonl"]
    prior = [json.loads(line) for line in previous_text.splitlines()]
    assert len(prior) == 10000 and prior[:500] == pilot["train"]
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    target_tokenizer = AutoTokenizer.from_pretrained(TARGET, revision=TARGET_REVISION)
    assert tokenizer.backend_tokenizer.to_str() == target_tokenizer.backend_tokenizer.to_str()
    data = load_dataset(DATASET, SUBSET, revision=DATASET_REVISION)
    reconstructed = {}
    for split in ("validation", "test", "train"):
        reconstructed[split] = articles(text for batch in data[split].iter(batch_size=10000)
                                        for text in batch["text"])
    assert {k: len(v) for k, v in reconstructed.items()} == {"validation": 60, "test": 60, "train": 28472}
    evaluation = reconstructed["validation"] + reconstructed["test"]
    eval_titles = {normalized(a["title"]) for a in evaluation}
    eval_hashes = {a["article_sha256"] for a in evaluation}
    forbidden = set()
    for article in evaluation:
        forbidden.update(passage_hashes(tokenizer.encode(article["text"], add_special_tokens=False)))
    for split in ("validation", "test"):
        for row in pilot[split]:
            forbidden.update(passage_hashes(row["input_ids"] + row.get("reference_ids", [])))
    source = [a for a in reconstructed["train"] if normalized(a["title"]) not in eval_titles
              and a["article_sha256"] not in eval_hashes]
    print(f"Blocked all 120 official evaluation articles and {len(prior)} prior training chunks", flush=True)
    rows, rejected = select_fresh(source, tokenizer, prior, forbidden, COUNT, SEED)
    print("Independently auditing every selected window", flush=True)
    audit_fresh(rows, prior, forbidden, eval_titles, eval_hashes, COUNT)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    assert len(payload.encode()) < 100 * 1024**2, "Dataset exceeds GitHub's individual-file limit"
    manifest = {key: pilot_manifest[key] for key in ("dataset", "dataset_revision", "subset", "tokenizer",
        "tokenizer_revision", "target_tokenizer", "target_tokenizer_revision", "tokenizer_backends_identical",
        "add_special_tokens", "chat_template")}
    manifest.update({"seed": SEED, "window_tokens": WINDOW, "train_chunks": COUNT,
        "train_tokens": COUNT * WINDOW, "starting_optimizer_step": 2625, "ending_optimizer_step": 10000,
        "optimizer_updates": COUNT // 4, "unique_training_articles": len({normalized(r['article_title']) for r in rows}),
        "sampling": "seeded shuffled articles, random-offset nonoverlapping gap tiling, randomized article-balanced rounds and final order",
        "prior_training_sha256": previous_manifest["file_sha256"]["train.jsonl"],
        "prior_training_chunks": len(prior), "rejected_repeated_passage_windows": rejected,
        "official_evaluation_articles_excluded": len(evaluation),
        "evaluation_files": {"validation": "../pilot/validation.jsonl", "test": "../pilot/test.jsonl"},
        "file_sha256": {"train.jsonl": digest(payload), **{f"{s}.jsonl": pilot_manifest["file_sha256"][f"{s}.jsonl"]
                                                                   for s in ("validation", "test")}},
        "checks": {"overlap_with_prior_source_spans": 0, "duplicate_windows": 0,
                   "shared_32_token_passages_with_prior_training": 0,
                   "shared_32_token_passages_with_entire_official_evaluation": 0,
                   "shared_32_token_passages_between_new_windows": 0, "new_source_span_overlap": 0},
        "versions": {p: importlib.metadata.version(p) for p in ("datasets", "transformers", "tokenizers")},
        "limitations": ["Excludes all known project and official WikiText evaluation data, not unknown external benchmarks.",
                       "128-bit passage fingerprints are conservative collision checks; matches are rejected.",
                       "No semantic/paraphrase or model-pretraining contamination guarantee.",
                       "Some training articles recur with new disjoint text; prior training chunks never recur."]})
    outputs = {"train.jsonl": payload, "manifest.json": json.dumps(manifest, indent=2) + "\n"}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        path = OUTPUT / name
        if path.exists() and path.read_text() != value:
            raise FileExistsError(f"Refusing to replace fixed prepared data: {path}")
    load_pilot()
    for name, value in outputs.items():
        (OUTPUT / name).write_text(value)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
