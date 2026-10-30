"""Build the fixed, article-separated WikiText pilot on CPU; no model weights."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import re

DATASET = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SUBSET = "wikitext-103-raw-v1"
TOKENIZER = "Qwen/Qwen3-0.6B"
TOKENIZER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
TARGET = "Qwen/Qwen3-4B"
TARGET_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
SEED = 42
WINDOW = 256
PROMPT = 128
SHINGLE = 32
OUTPUT = Path(__file__).resolve().parent / "data" / "pilot"


def normalized(text):
    return " ".join(text.casefold().split())


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def articles(rows):
    """Rejoin HF rows at level-one headings; section headings stay in the article."""
    result, parts, title, start = [], [], None, None
    iterator = iter(rows)
    text, previous_blank, index = next(iterator, None), True, 0
    while text is not None:
        following = next(iterator, None)
        # WikiText also splits inline equations at '='; real headings have blank neighbors.
        heading = (re.fullmatch(r"= ([^=].*?) =", text.strip())
                   if previous_blank and (following is None or not following.strip()) else None)
        if heading:
            if title is not None:
                result.append(make_article(title, start, index, parts))
            title, start, parts = heading[1], index, []
        if title is not None:
            parts.append(text)
        elif text.strip():
            raise ValueError("Nonempty text before the first article heading")
        previous_blank, text = not text.strip(), following
        index += 1
    if title is not None:
        result.append(make_article(title, start, index, parts))
    return result


def make_article(title, start, end, parts):
    text = "".join(parts)
    return {
        "title": title, "row_start": start, "row_end": end, "text": text,
        "article_sha256": digest(normalized(text)),
    }


def shingles(ids):
    return {tuple(ids[i:i + SHINGLE]) for i in range(len(ids) - SHINGLE + 1)}


def sample(source, count, tokenizer, seed, per_article=1, forbidden=()):
    """One window per shuffled article per round; at most two for test articles."""
    rng = random.Random(seed)
    order = list(source)
    rng.shuffle(order)
    selected, pending, seen = [], [], set()
    rejected = {"too_short_articles": 0, "duplicate_or_shared_passage_windows": 0}
    blocked = set(forbidden)
    for article in order:
        ids = tokenizer.encode(article["text"], add_special_tokens=False)
        if len(ids) < WINDOW:
            rejected["too_short_articles"] += 1
            continue
        if per_article == 2 and len(ids) >= 2 * WINDOW:
            # Separate halves guarantee the two reserved evaluation regions cannot overlap.
            middle = len(ids) // 2
            starts = [rng.randint(0, middle - WINDOW), rng.randint(middle, len(ids) - WINDOW)]
            rng.shuffle(starts)
        else:
            starts = [rng.randint(0, len(ids) - WINDOW)]
        for round_index, start in enumerate(starts):
            window = ids[start:start + WINDOW]
            row = {
                "article_title": article["title"],
                "source_row_start": article["row_start"],
                "source_row_end": article["row_end"],
                "article_sha256": article["article_sha256"],
                "token_start": start, "token_end": start + WINDOW,
                "window_sha256": digest(json.dumps(window, separators=(",", ":"))),
                "input_ids": window, "text": tokenizer.decode(window),
            }
            if round_index:
                pending.append(row)
            elif not accept(row, blocked, seen):
                rejected["duplicate_or_shared_passage_windows"] += 1
            else:
                selected.append(row)
        if len(selected) == count:
            return selected, rejected
    rng.shuffle(pending)
    for row in pending:
        if accept(row, blocked, seen):
            selected.append(row)
            if len(selected) == count:
                return selected, rejected
        else:
            rejected["duplicate_or_shared_passage_windows"] += 1
    raise ValueError(f"Only found {len(selected)} clean windows; need {count}")


def accept(row, blocked, seen):
    if row["window_sha256"] in seen or shingles(row["input_ids"]) & blocked:
        return False
    seen.add(row["window_sha256"])
    return True


def validate(splits):
    """Fail on source leakage, shared passages, bad lengths, or overlapping windows."""
    expected = {"train": 500, "validation": 50, "test": 100}
    titles, hashes, passages = set(), set(), set()
    for split, rows in splits.items():
        assert len(rows) == expected[split], (split, len(rows))
        current_titles, current_hashes, current_passages = set(), set(), set()
        windows, spans = set(), {}
        for row in rows:
            ids = row["input_ids"] + row.get("reference_ids", [])
            assert len(ids) == WINDOW and all(type(token) is int and token >= 0 for token in ids)
            assert row["token_end"] - row["token_start"] == WINDOW
            if split == "test":
                assert len(row["input_ids"]) == len(row["reference_ids"]) == PROMPT
            key = normalized(row["article_title"])
            assert key not in titles and row["article_sha256"] not in hashes
            assert row["window_sha256"] == digest(json.dumps(ids, separators=(",", ":")))
            assert row["window_sha256"] not in windows
            windows.add(row["window_sha256"])
            for start, end in spans.setdefault(key, []):
                assert row["token_end"] <= start or row["token_start"] >= end
            spans[key].append((row["token_start"], row["token_end"]))
            pieces = shingles(ids)
            assert not pieces & passages, f"Shared {SHINGLE}-token passage in {split}"
            current_titles.add(key)
            current_hashes.add(row["article_sha256"])
            current_passages.update(pieces)
        assert max(map(len, spans.values())) <= (2 if split == "test" else 1)
        titles.update(current_titles)
        hashes.update(current_hashes)
        passages.update(current_passages)


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer

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
    expected_articles = {"train": 28472, "validation": 60, "test": 60}
    if {k: len(v) for k, v in reconstructed.items()} != expected_articles:
        raise ValueError("Reconstructed article counts differ from the pinned WikiText release")

    # Evaluation articles take priority; remove training conflicts before random sampling.
    eval_articles = reconstructed["validation"] + reconstructed["test"]
    eval_titles = {normalized(a["title"]) for a in eval_articles}
    eval_hashes = {a["article_sha256"] for a in eval_articles}
    train = [a for a in reconstructed["train"]
             if normalized(a["title"]) not in eval_titles and a["article_sha256"] not in eval_hashes]
    splits, stats = {}, {}
    splits["validation"], stats["validation"] = sample(reconstructed["validation"], 50, tokenizer, SEED + 1)
    blocked = set().union(*(shingles(row["input_ids"]) for row in splits["validation"]))
    splits["test"], stats["test"] = sample(reconstructed["test"], 100, tokenizer, SEED + 2, 2, blocked)
    blocked.update(set().union(*(shingles(row["input_ids"]) for row in splits["test"])))
    splits["train"], stats["train"] = sample(train, 500, tokenizer, SEED, forbidden=blocked)
    for row in splits["test"]:
        ids = row["input_ids"]
        row["input_ids"], row["reference_ids"] = ids[:PROMPT], ids[PROMPT:]
        row["text"] = tokenizer.decode(row["input_ids"])
        row["reference_text"] = tokenizer.decode(row["reference_ids"])
    validate(splits)
    payloads = {f"{split}.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
                for split, rows in splits.items()}
    manifest = {
        "dataset": DATASET, "dataset_revision": DATASET_REVISION, "subset": SUBSET,
        "tokenizer": TOKENIZER, "tokenizer_revision": TOKENIZER_REVISION,
        "target_tokenizer": TARGET, "target_tokenizer_revision": TARGET_REVISION,
        "tokenizer_backends_identical": True, "add_special_tokens": False, "chat_template": False,
        "seed": SEED, "split_seeds": {"train": SEED, "validation": SEED + 1, "test": SEED + 2},
        "window_tokens": WINDOW, "test_prompt_tokens": PROMPT, "test_reference_tokens": WINDOW - PROMPT,
        "sampling": "article-balanced; one random window/article, test up to two in separate article halves",
        "article_counts": {k: len(v) for k, v in reconstructed.items()},
        "excluded_training_articles": len(reconstructed["train"]) - len(train),
        "selected_articles": {k: len({r["article_title"] for r in v}) for k, v in splits.items()},
        "sample_counts": {k: len(v) for k, v in splits.items()}, "sampling_rejections": stats,
        "checks": {"cross_split_article_overlap": 0, "duplicate_windows": 0,
                   "cross_split_shared_32_token_passages": 0, "within_article_window_overlap": 0},
        "file_sha256": {name: digest(content) for name, content in payloads.items()},
        "versions": {p: importlib.metadata.version(p) for p in ("datasets", "transformers", "tokenizers")},
        "limitations": ["Pilot sizes are a resource budget, not a statistically optimized split.",
                        "Long-enough articles only; article-balanced, not token-weighted corpus sampling.",
                        "Exact checks do not rule out paraphrases, topic overlap, or pretraining exposure.",
                        "Test prompts share articles: aggregate uncertainty by article, not by prompt.",
                        "Wikipedia reference continuations are not the target model's generated answers."],
    }
    payloads["manifest.json"] = json.dumps(manifest, indent=2) + "\n"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        path = OUTPUT / name
        if path.exists() and path.read_text() != content:
            raise FileExistsError(f"Refusing to replace a different fixed dataset: {path}")
    for name, content in payloads.items():
        (OUTPUT / name).write_text(content)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
