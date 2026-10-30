"""Prevent restarting an existing W&B graph from stale weights or the wrong data."""

import copy
import json
import unittest

from continue_training import ROOT, continuation_plan


class ContinuationTest(unittest.TestCase):
    def setUp(self):
        self.report = json.loads((ROOT / "reports/training.json").read_text())
        self.manifest = json.loads((ROOT / "data/wikitext-10k/manifest.json").read_text())

    def test_continues_same_run_after_125_with_2500_more_updates(self):
        plan = continuation_plan(self.report, 125, self.manifest)
        self.assertEqual(plan["run_id"], "hkays15c")
        self.assertEqual((plan["start_step"] + 1, plan["end_step"]), (126, 2625))
        self.assertFalse(plan["optimizer_state_restored"])
        self.assertTrue(plan["checkpoint_dir"].endswith("hkays15c/last"))
        self.assertIn("adapter_model.safetensors", plan["checkpoint_sha256"])

    def test_refuses_history_that_has_advanced_beyond_saved_weights(self):
        with self.assertRaisesRegex(ValueError, "stale weights"):
            continuation_plan(self.report, 126, self.manifest)

    def test_refuses_changed_validation_data(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["file_sha256"]["validation.jsonl"] = "changed"
        with self.assertRaisesRegex(ValueError, "Validation data changed"):
            continuation_plan(self.report, 125, manifest)


if __name__ == "__main__":
    unittest.main()
