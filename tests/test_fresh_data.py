import json
import random
import unittest

from prepare_data import WINDOW, digest
from prepare_fresh import audit_fresh, fresh_starts, passage_hashes, select_fresh


class Tokens:
    def encode(self, text, **kwargs):
        return list(text)

    def decode(self, ids):
        return str(ids)


def article(name, ids):
    return {"title": name, "text": ids, "article_sha256": digest(name), "row_start": 0, "row_end": 1}


def row(name, ids, start=0):
    return {"article_title": name, "article_sha256": digest(name), "token_start": start,
            "token_end": start + WINDOW, "input_ids": ids,
            "window_sha256": digest(json.dumps(ids, separators=(",", ":")))}


class FreshDataTest(unittest.TestCase):
    def test_gap_tiling_never_reuses_old_or_new_source_tokens(self):
        used = [(100, 356), (1200, 1456)]
        starts = fresh_starts(2400, used, random.Random(42))
        occupied = used + [(s, s + WINDOW) for s in starts]
        for index, (a, b) in enumerate(occupied):
            for c, d in occupied[index + 1:]:
                self.assertTrue(b <= c or a >= d)
        with self.assertRaises(ValueError):
            fresh_starts(1000, [(10, 300), (200, 456)], random.Random(42))

    def test_random_selection_is_repeatable_and_avoids_prior_chunks(self):
        source = [article(f"a{i}", list(range(i * 10000, i * 10000 + 3000))) for i in range(4)]
        prior = [row("a0", source[0]["text"][700:956], 700)]
        first, _ = select_fresh(source, Tokens(), prior, set(), 8, 42)
        second, _ = select_fresh(source, Tokens(), prior, set(), 8, 42)
        self.assertEqual(first, second)
        self.assertEqual(len({r['article_title'] for r in first}), 4)
        audit_fresh(first, prior, set(), set(), set(), 8)

    def test_audit_rejects_partial_overlap_even_without_32_shared_tokens(self):
        prior = [row("same", list(range(256)), 0)]
        overlapping = row("same", list(range(1000, 1256)), 250)
        with self.assertRaisesRegex(AssertionError, "Reused source tokens"):
            audit_fresh([overlapping], prior, set(), set(), set(), 1)

    def test_any_heldout_passage_is_rejected_even_under_another_title(self):
        reserved = list(range(10000, 10300))
        candidate = row("different", reserved[:256])
        with self.assertRaisesRegex(AssertionError, "held-out passage"):
            audit_fresh([candidate], [], passage_hashes(reserved), set(), set(), 1)
        with self.assertRaises(AssertionError):
            audit_fresh([candidate], [], set(), {"different"}, set(), 1)

    def test_repeated_passages_between_fresh_articles_are_rejected(self):
        ids = list(range(256))
        with self.assertRaisesRegex(AssertionError, "Repeated training"):
            audit_fresh([row("one", ids), row("two", [9999] + ids[:255])], [], set(), set(), set(), 2)

    def test_insufficient_clean_data_fails_instead_of_recycling(self):
        source = [article("one", list(range(256)))]
        with self.assertRaisesRegex(ValueError, "Only 1"):
            select_fresh(source, Tokens(), [], set(), 2, 42)


if __name__ == "__main__":
    unittest.main()
