"""Check pooled throughput and that output mismatches cannot count as verified speedup."""

import unittest

from baseline import summarize


class SummaryTest(unittest.TestCase):
    def records(self):
        return [
            {"prompt_index": 0, "repetition": rep, "mode": mode, "token_ids": [1, 2],
             "seconds": (rep + 1) * (2 if mode == "target" else 1),
             "accepted_draft_tokens": 1, "proposed_draft_tokens": 4, "target_verification_passes": 1}
            for rep in range(2) for mode in ("target", "speculative")
        ]

    def test_pooled_throughput_and_acceptance(self):
        result = summarize(self.records())
        self.assertAlmostEqual(result["target"]["tokens_per_second"], 4 / 6)
        self.assertEqual(result["verified_speedup"], 2)
        self.assertEqual(result["speculative"]["acceptance_rate"], 0.25)
        self.assertEqual(result["speculative"]["generated_tokens_per_verification"], 2)

    def test_mismatch_blocks_verified_speedup(self):
        rows = self.records()
        rows[1]["token_ids"] = [1, 9]
        result = summarize(rows)
        self.assertIsNone(result["verified_speedup"])
        self.assertEqual(result["checks"]["mismatched_sequences"], 1)
        self.assertEqual(result["checks"]["position_mismatch_rate"], 0.25)


if __name__ == "__main__":
    unittest.main()
