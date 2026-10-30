import copy
import json
from pathlib import Path
import unittest

from continue_to_10k import resume_plan


class ResumeTo10kTest(unittest.TestCase):
    def setUp(self):
        self.report = json.loads((Path(__file__).resolve().parents[1] / "reports/training-10k.json").read_text())
        self.manifest = {
            "train_chunks": 29500, "optimizer_updates": 7375, "window_tokens": 256,
            "starting_optimizer_step": 2625, "ending_optimizer_step": 10000,
            "prior_training_sha256": self.report["config"]["dataset_manifest"]["file_sha256"]["train.jsonl"],
            "file_sha256": {**self.report["config"]["dataset_manifest"]["file_sha256"], "train.jsonl": "new-data"},
            "official_evaluation_articles_excluded": 120,
            "checks": {"overlap_with_prior_source_spans": 0, "duplicate_windows": 0,
                       "shared_32_token_passages_with_prior_training": 0,
                       "shared_32_token_passages_with_entire_official_evaluation": 0,
                       "shared_32_token_passages_between_new_windows": 0, "new_source_span_overlap": 0},
        }

    def test_resumes_same_run_with_optimizer_through_exactly_10000(self):
        plan = resume_plan(self.report, 2625, self.manifest)
        self.assertEqual(plan["run_id"], "hkays15c")
        self.assertEqual(plan["start_step"] + plan["train_chunks"] // 4, 10000)
        self.assertTrue(plan["optimizer_state_restored"])
        self.assertEqual(plan["optimizer_sha256"], self.report["files"]["training_state.pt"])
        self.assertTrue(plan["checkpoint_dir"].endswith("continuation-126-2625/last"))

    def test_refuses_stale_weights_changed_tests_and_failed_or_missing_audits(self):
        with self.assertRaises(ValueError):
            resume_plan(self.report, 2626, self.manifest)
        for mutation in (lambda m: m["file_sha256"].update({"test.jsonl": "changed"}),
                         lambda m: m["checks"].update({"overlap_with_prior_source_spans": 1}),
                         lambda m: m.update({"checks": {}}),
                         lambda m: m.update({"prior_training_sha256": "wrong"})):
            changed = copy.deepcopy(self.manifest)
            mutation(changed)
            with self.assertRaises(ValueError):
                resume_plan(self.report, 2625, changed)


if __name__ == "__main__":
    unittest.main()
