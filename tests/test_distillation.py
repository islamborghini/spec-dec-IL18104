"""Check KL direction, frozen weights, and saved-adapter inference on tiny Qwen."""

import importlib.util
import math
import tempfile
import unittest

from distillation import distillation_loss


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("peft"),
                     "Real training checks run on Modal CPU")
class DistillationTest(unittest.TestCase):
    def test_kl_direction_and_per_token_normalization(self):
        import torch

        teacher = torch.tensor([[[0.8, 0.2]]], requires_grad=True).log()
        student = torch.tensor([[[0.5, 0.5]]], requires_grad=True).log()
        loss = distillation_loss(student, teacher)
        self.assertAlmostEqual(loss.item(), 0.8 * math.log(1.6) + 0.2 * math.log(0.4), places=6)
        repeated = distillation_loss(student.repeat(2, 3, 1), teacher.repeat(2, 3, 1))
        self.assertAlmostEqual(loss.item(), repeated.item(), places=6)
        self.assertEqual(torch.autograd.grad(loss, teacher, allow_unused=True), (None,))

    def test_shape_validation(self):
        import torch

        with self.assertRaises(ValueError):
            distillation_loss(torch.zeros(1, 2, 3), torch.zeros(1, 2, 4))

    def test_only_lora_updates_and_save_reload_merge_preserve_logits(self):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.set_num_threads(1)
        config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=8, attention_dropout=0.0)
        config._attn_implementation = "eager"
        torch.manual_seed(42)
        base = Qwen3ForCausalLM(config).eval()
        base_state = {name: value.clone() for name, value in base.state_dict().items()}
        torch.manual_seed(43)
        teacher = Qwen3ForCausalLM(config).eval().requires_grad_(False)
        student = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
                                                 target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        params = [p for p in student.parameters() if p.requires_grad]
        self.assertTrue(all("lora_" in n for n, p in student.named_parameters() if p.requires_grad))
        ids = torch.tensor([[1, 7, 3, 9]])
        optimizer = torch.optim.AdamW(params, lr=1e-2, weight_decay=0.0)
        with torch.no_grad():
            target_logits = teacher(ids, use_cache=False).logits
        loss = distillation_loss(student(ids, use_cache=False).logits, target_logits)
        loss.backward()
        self.assertTrue(torch.isfinite(torch.nn.utils.clip_grad_norm_(params, 1.0)))
        optimizer.step()
        self.assertTrue(any(torch.count_nonzero(p).item() for n, p in student.named_parameters() if "lora_B" in n))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        for name, value in student.get_base_model().state_dict().items():
            if "lora_" not in name:
                self.assertTrue(torch.equal(value, base_state[name.replace(".base_layer", "")]))
        with tempfile.TemporaryDirectory() as directory:
            student.eval().save_pretrained(directory, save_embedding_layers=False)
            fresh = Qwen3ForCausalLM(config)
            fresh.load_state_dict(base_state)
            loaded = PeftModel.from_pretrained(fresh, directory).eval()
            with torch.no_grad():
                expected = student(ids, use_cache=False).logits
                torch.testing.assert_close(loaded(ids, use_cache=False).logits, expected)
                merged = loaded.merge_and_unload(safe_merge=True)
                torch.testing.assert_close(merged(ids, use_cache=False).logits, expected, atol=1e-6, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
