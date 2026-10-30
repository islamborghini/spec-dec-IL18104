"""Audit the expanded data and catch cross-split leakage in either split order."""

import json
from pathlib import Path
import unittest

from prepare_10k import load_pilot, OUTPUT
from prepare_data import digest, validate


class ExpandedDataTest(unittest.TestCase):
    def test_expanded_data_is_audited_and_preserves_pilot_prefix(self):
        _, pilot = load_pilot()
        manifest = json.loads((OUTPUT / "manifest.json").read_text())
        content = (OUTPUT / "train.jsonl").read_text()
        self.assertEqual(digest(content), manifest["file_sha256"]["train.jsonl"])
        rows = [json.loads(line) for line in content.splitlines()]
        self.assertEqual(rows[:500], pilot["train"])
        self.assertEqual(len({r["article_title"] for r in rows}), 10000)
        validate({"validation": pilot["validation"], "test": pilot["test"], "train": rows}, train_count=10000)

    def test_leakage_guard_works_with_large_training_split_last_or_first(self):
        _, pilot = load_pilot()
        pilot["train"] = pilot["train"][:3]
        for order in (("validation", "test", "train"), ("train", "validation", "test")):
            splits = {key: pilot[key] for key in order}
            validate(splits, train_count=3)
        row = pilot["train"][0]
        row["input_ids"] = pilot["validation"][0]["input_ids"]
        row["window_sha256"] = digest(json.dumps(row["input_ids"], separators=(",", ":")))
        for order in (("validation", "test", "train"), ("train", "validation", "test")):
            with self.assertRaisesRegex(AssertionError, "Shared 32-token passage"):
                validate({key: pilot[key] for key in order}, train_count=3)


if __name__ == "__main__":
    unittest.main()
