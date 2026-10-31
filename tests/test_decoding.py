"""Greedy verification cases plus tiny real-Qwen cache regression checks."""

import importlib.util
import unittest

from decoding import greedy_decode, speculative_decode, verified_tokens


class VerificationTest(unittest.TestCase):
    def test_reject_first_middle_last_and_accept_all(self):
        proposals = [1, 2, 3]
        for predictions, expected, accepted in [
            ([9, 2, 3, 4], [9], 0), ([1, 9, 3, 4], [1, 9], 1),
            ([1, 2, 9, 4], [1, 2, 9], 2), ([1, 2, 3, 4], [1, 2, 3, 4], 3),
        ]:
            self.assertEqual(verified_tokens(proposals, predictions, 10), (expected, accepted))

    def test_budget_and_eos_truncate_acceptance_counts(self):
        self.assertEqual(verified_tokens([1, 2, 3], [1, 2, 3, 4], 1), ([1], 1))
        self.assertEqual(verified_tokens([1, 2, 3], [1, 2, 3, 4], 8, {2}), ([1, 2], 2))
        self.assertEqual(verified_tokens([1, 9], [1, 2, 3], 8, {2}), ([1, 2], 1))

    def test_invalid_verification_inputs(self):
        with self.assertRaises(ValueError):
            verified_tokens([1], [1], 3)
        with self.assertRaises(ValueError):
            verified_tokens([1], [1, 2], 0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Real-model tests run in the Modal CPU check")
class TinyQwenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.set_num_threads(1)
        config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=8, eos_token_id=None,
                             pad_token_id=0, attention_dropout=0.0)
        config._attn_implementation = "eager"
        torch.manual_seed(42)
        cls.target = Qwen3ForCausalLM(config).eval()
        torch.manual_seed(43)
        cls.drafter = Qwen3ForCausalLM(config).eval()

    def test_cached_decoders_match_transformers_for_several_k_and_budgets(self):
        import torch

        for prefix in ([1], [1, 7, 3, 9]):
            prompt = torch.tensor([prefix])
            for budget in (1, 2, 9, 17):
                with torch.inference_mode():
                    expected = self.target.generate(prompt, max_new_tokens=budget, do_sample=False,
                                                    eos_token_id=None, pad_token_id=0)[0, len(prefix):].tolist()
                self.assertEqual(greedy_decode(self.target, prompt, budget)["token_ids"], expected)
                for k in (1, 2, 4, 8):
                    actual = speculative_decode(self.target, self.drafter, prompt, budget, k)
                    self.assertEqual(actual["token_ids"], expected, (prefix, budget, k))
                    self.assertGreater(actual["target_verification_passes"], 0)

    def test_identical_drafter_accepts_every_used_proposal_and_bonus(self):
        import torch

        result = speculative_decode(self.target, self.target, torch.tensor([[1, 2, 3]]), 12, 4)
        self.assertEqual(result["accepted_draft_tokens"], 10)
        self.assertEqual(result["target_verification_passes"], 3)
        self.assertEqual(len(result["token_ids"]), 12)

    def test_eos_stops_both_decoders(self):
        import torch

        prompt = torch.tensor([[1, 7, 3, 9]])
        reference = greedy_decode(self.target, prompt, 12)["token_ids"]
        for eos in (reference[0], reference[2]):
            expected = reference[:reference.index(eos) + 1]
            self.assertEqual(greedy_decode(self.target, prompt, 12, {eos})["token_ids"], expected)
            for drafter in (self.target, self.drafter):
                actual = speculative_decode(self.target, drafter, prompt, 12, 4, {eos})
                self.assertEqual(actual["token_ids"], expected)

    def test_zero_budget_and_invalid_batch(self):
        import torch

        prompt = torch.tensor([[1]])
        self.assertEqual(greedy_decode(self.target, prompt, 0)["token_ids"], [])
        self.assertEqual(speculative_decode(self.target, self.drafter, prompt, 0)["token_ids"], [])
        with self.assertRaises(ValueError):
            speculative_decode(self.target, self.drafter, prompt, 1, 0)
        with self.assertRaises(ValueError):
            greedy_decode(self.target, torch.tensor([[1], [2]]), 2)

    def test_stream_callbacks_match_outputs_and_keep_speculative_bursts(self):
        import torch

        prompt = torch.tensor([[1, 2, 3]])
        chunks = []
        result = speculative_decode(self.target, self.target, prompt, 12, 4, on_tokens=chunks.append)
        self.assertEqual([token for chunk in chunks for token in chunk], result["token_ids"])
        self.assertEqual(list(map(len, chunks)), [5, 5, 2])
        singles = []
        greedy = greedy_decode(self.target, prompt, 12, on_tokens=singles.append)
        self.assertEqual([token for chunk in singles for token in chunk], greedy["token_ids"])
        self.assertEqual(greedy["token_ids"], result["token_ids"])
        self.assertTrue(all(len(chunk) == 1 for chunk in singles))
        empty = []
        speculative_decode(self.target, self.target, prompt, 0, 4, on_tokens=empty.append)
        greedy_decode(self.target, prompt, 0, on_tokens=empty.append)
        self.assertEqual(empty, [])


if __name__ == "__main__":
    unittest.main()
