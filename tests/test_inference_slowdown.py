"""Hook-based slowdown timing for Qwen2 and Qwen3.5."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import models as M
from tests.test_qwen35_split import _tiny_qwen2, _tiny_qwen35


def _load_script():
    """Import the slowdown script as a module.

    Returns:
        Loaded module.
    """
    path = ROOT / "scripts" / "18_measure_inference_slowdown.py"
    spec = importlib.util.spec_from_file_location("slowdown18", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SlowdownHookTest(unittest.TestCase):
    """Check split timing without downloading checkpoints."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the benchmark module once."""
        cls.mod = _load_script()

    def test_qwen35_is_a_default_model(self) -> None:
        """The standalone sweep includes Qwen3.5-4B."""
        self.assertIn(M.QWEN35_4B, self.mod.DEFAULT_MODELS)
        self.assertEqual(self.mod.M.DEFAULT_BATCH, 4)
        self.assertEqual(self.mod.model_label(M.QWEN35_4B), "Qwen3.5-4B")
        self.assertEqual(
            self.mod.resolve_dtype("auto", "cpu"),
            torch.float32,
        )
        self.assertEqual(
            self.mod.resolve_dtype("auto", "cuda:0"),
            M.default_dtype(M.QWEN35_4B, "cuda:0"),
        )

    def test_qwen35_hook_splits_one_forward(self) -> None:
        """A Qwen3.5 forward reports a cut activation at the split."""
        model = _tiny_qwen35()
        ids = torch.randint(0, 128, (2, 6))
        mask = torch.ones_like(ids)
        head, tail, hidden = self.mod.time_one_forward(
            model, ids, mask, 2, "cpu",
        )
        self.assertGreaterEqual(head, 0.0)
        self.assertGreaterEqual(tail, 0.0)
        self.assertGreater(head + tail, 0.0)
        self.assertEqual(tuple(hidden.shape), (2, 6, 64))

    def test_qwen2_hook_splits_one_forward(self) -> None:
        """A Qwen2 forward reports a cut activation at the split."""
        model = _tiny_qwen2()
        ids = torch.randint(0, 64, (2, 6))
        mask = torch.ones_like(ids)
        _head, _tail, hidden = self.mod.time_one_forward(
            model, ids, mask, 1, "cpu",
        )
        self.assertEqual(tuple(hidden.shape), (2, 6, 32))

    def test_noise_and_suppressor_keep_shape(self) -> None:
        """Placeholder noise and suppression preserve the activation shape."""
        hidden = torch.zeros(2, 4, 8)
        state = self.mod.make_placeholder_state(
            8, 2, "cpu", torch.float32, 0,
        )
        noisy = self.mod.add_lowrank_noise(hidden, state)
        suppressed = self.mod.apply_suppressor(noisy, state)
        self.assertFalse(torch.allclose(noisy, hidden))
        self.assertEqual(tuple(suppressed.shape), tuple(hidden.shape))


if __name__ == "__main__":
    unittest.main()
