"""Prevent restarting an existing W&B graph from stale weights or the wrong data."""

import copy
import json
import unittest
from unittest.mock import patch

from continue_training import ROOT, continuation_plan, last_optimizer_step, resume_training


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

    def test_sparse_wandb_rows_do_not_reset_or_break_the_step_counter(self):
        self.assertEqual(last_optimizer_step([{}, {"optimizer_step": None},
                                              {"optimizer_step": 124}, {"optimizer_step": 125}]), 125)
        self.assertEqual(last_optimizer_step([{"optimizer_step": None}]), -1)

    def test_gpu_wrapper_forwards_the_checkpoint_plan_to_the_shared_trainer(self):
        plan = continuation_plan(self.report, 125, self.manifest)
        with patch("continue_training.train.train_and_evaluate.local", return_value={"ok": True}) as trainer:
            self.assertEqual(resume_training.local("revision", plan), {"ok": True})
            trainer.assert_called_once_with("revision", plan)


if __name__ == "__main__":
    unittest.main()
