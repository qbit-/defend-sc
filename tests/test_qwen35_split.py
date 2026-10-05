"""Split-hook checks for Qwen2 and Qwen3.5 text decoders."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from transformers import (
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import models as M
from src import split_model as SM


def _tiny_qwen35() -> Qwen3_5ForConditionalGeneration:
    """Build a random Qwen3.5 model small enough for a CPU test.

    Returns:
        Eval-mode conditional-generation model.
    """
    text = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        out_hidden_size=64,
        num_position_embeddings=16,
    )
    cfg = Qwen3_5Config(
        text_config=text,
        vision_config=vision,
        tie_word_embeddings=False,
    )
    model = Qwen3_5ForConditionalGeneration(cfg)
    model.eval()
    return model


def _tiny_qwen2() -> Qwen2ForCausalLM:
    """Build a random Qwen2 model small enough for a CPU test.

    Returns:
        Eval-mode causal LM.
    """
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    return model


class ModelOptionTest(unittest.TestCase):
    """Check ids, dtypes, and split hooks without downloading weights."""

    def test_qwen35_id_and_dtype(self) -> None:
        """Qwen3.5 ids select the 4B option and a shared dtype."""
        self.assertTrue(M.is_qwen35_id(M.QWEN35_4B))
        self.assertTrue(M.is_qwen35_id("local/Qwen3_5-4B"))
        self.assertFalse(M.is_qwen35_id(M.QWEN25_0_5B))
        self.assertEqual(
            M.default_dtype(M.QWEN35_4B, "cpu"),
            torch.float32,
        )
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            cuda_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            cuda_dtype = torch.float16
        else:
            cuda_dtype = torch.float32
        self.assertEqual(
            M.default_dtype(M.QWEN25_0_5B, "cuda:0"),
            cuda_dtype,
        )
        self.assertIn(M.QWEN35_4B, M.SUPPORTED)
        self.assertEqual(M.calibration_batch_size(M.QWEN35_4B), 4)
        self.assertEqual(M.calibration_batch_size(M.QWEN25_0_5B), 4)

    def test_qwen35_split_matches_full_forward(self) -> None:
        """An unperturbed Qwen3.5 split matches the full forward."""
        model = _tiny_qwen35()
        M._drop_vision_tower(model)
        self.assertFalse(hasattr(model.model, "visual"))
        self.assertEqual(M.family(model), "qwen35")
        self.assertEqual(M.hidden_size(model), 64)
        self.assertEqual(M.vocab_size(model), 128)
        self.assertEqual(M.n_layers(model), 4)
        self.assertEqual(M.forward_batch_size(model), 4)
        self.assertEqual(M.cloud_chunk_size(model), 32)
        ids = torch.randint(0, 128, (2, 8))
        mask = torch.ones_like(ids)
        with torch.no_grad():
            full = model(input_ids=ids, attention_mask=mask).logits
            captured = SM.capture_a_k(model, ids, k=2, attention_mask=mask)
            plain = SM.split_run(model, ids, k=2, attention_mask=mask)
        self.assertEqual(tuple(captured.shape), (2, 8, 64))
        self.assertTrue(torch.allclose(plain["logits"], full))
        self.assertTrue(torch.allclose(plain["a_k"], captured))

    def test_qwen35_override_changes_logits(self) -> None:
        """Replacing the cut activation changes Qwen3.5 logits."""
        model = _tiny_qwen35()
        ids = torch.randint(0, 128, (2, 6))
        mask = torch.ones_like(ids)
        with torch.no_grad():
            plain = SM.split_run(model, ids, k=1, attention_mask=mask)
            zeros = torch.zeros_like(plain["a_k"])
            edited = SM.split_run(
                model, ids, k=1, hidden_override=zeros, attention_mask=mask,
            )
        self.assertFalse(torch.allclose(edited["logits"], plain["logits"]))

    def test_qwen2_family_still_splits(self) -> None:
        """Qwen2.5-style modules keep the qwen family and split hooks."""
        model = _tiny_qwen2()
        self.assertEqual(M.family(model), "qwen")
        self.assertEqual(M.hidden_size(model), 32)
        self.assertEqual(M.vocab_size(model), 64)
        self.assertEqual(M.forward_batch_size(model), 4)
        self.assertEqual(M.cloud_chunk_size(model), 32)
        ids = torch.randint(0, 64, (2, 6))
        mask = torch.ones_like(ids)
        with torch.no_grad():
            full = model(input_ids=ids, attention_mask=mask).logits
            plain = SM.split_run(model, ids, k=2, attention_mask=mask)
        self.assertTrue(torch.allclose(plain["logits"], full, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
