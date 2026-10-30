"""Exercise article boundaries, fixed sampling, and leakage guards without downloads."""

import json
from pathlib import Path
import unittest

import prepare_data as prep


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(map(int, text.split()))

    def decode(self, ids):
        return " ".join(map(str, ids))


class DataTest(unittest.TestCase):
    def test_sections_and_empty_rows_do_not_split_articles(self):
        rows = ["", " = First = \n", "", "body\n", "", " = = Section = = \n", "more\n",
                " = 1 henry and capacitance C = \n", "1 farad\n", "", " = Second = \n", "", "last\n"]
        articles = prep.articles(rows)
        self.assertEqual([a["title"] for a in articles], ["First", "Second"])
        self.assertEqual((articles[0]["row_start"], articles[0]["row_end"]), (1, 10))
        self.assertIn("Section", articles[0]["text"])
        self.assertIn("capacitance", articles[0]["text"])
        self.assertEqual(articles[1]["row_end"], len(rows))

    def test_sampling_repeats_and_test_windows_do_not_overlap(self):
        source = [prep.make_article(str(i), i, i + 1, [" ".join(str(j + i * 1000) for j in range(800))])
                  for i in range(4)]
        rows, _ = prep.sample(source, 7, FakeTokenizer(), 42, per_article=2)
        self.assertEqual(rows, prep.sample(source, 7, FakeTokenizer(), 42, per_article=2)[0])
        self.assertEqual(len({r["article_title"] for r in rows[:4]}), 4)
        for first in rows:
            for second in rows:
                if first is not second and first["article_title"] == second["article_title"]:
                    self.assertTrue(first["token_end"] <= second["token_start"] or
                                    second["token_end"] <= first["token_start"])

    def test_shared_passage_is_rejected_even_when_window_start_differs(self):
        source = [prep.make_article("copy", 0, 1, [" ".join(map(str, range(500)))])]
        blocked = prep.shingles(list(range(100, 356)))
        with self.assertRaisesRegex(ValueError, "Only found 0 clean"):
            prep.sample(source, 1, FakeTokenizer(), 42, forbidden=blocked)

    def test_repeated_article_titles_cannot_supply_multiple_training_windows(self):
        source = [prep.make_article("Same title", i, i + 1,
                                   [" ".join(str(j + i * 1000) for j in range(800))])
                  for i in range(3)]
        with self.assertRaisesRegex(ValueError, "Only found 1 clean"):
            prep.sample(source, 2, FakeTokenizer(), 42)

    def test_committed_artifacts_pass_audit_and_hash_checks(self):
        root = Path(__file__).resolve().parents[1] / "data" / "pilot"
        self.assertTrue((root / "manifest.json").exists(), "Run prepare_data.py first")
        manifest = json.loads((root / "manifest.json").read_text())
        splits = {}
        for name, expected_hash in manifest["file_sha256"].items():
            content = (root / name).read_text()
            self.assertEqual(prep.digest(content), expected_hash)
            splits[Path(name).stem] = [json.loads(line) for line in content.splitlines()]
        prep.validate(splits)
        # A copied source article must fail even if its token window is different.
        splits["train"][0]["article_title"] = splits["test"][0]["article_title"]
        with self.assertRaises(AssertionError):
            prep.validate(splits)


if __name__ == "__main__":
    unittest.main()
