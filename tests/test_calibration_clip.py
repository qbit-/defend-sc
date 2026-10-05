"""Clip-norm quantiles accept half-precision activations."""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_calibration() -> types.ModuleType:
    """Import the calibration script as a module.

    Returns:
        Loaded module.
    """
    path = ROOT / "scripts" / "01_collect_calibration.py"
    spec = importlib.util.spec_from_file_location(
        "collect_calibration", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClipQuantileTest(unittest.TestCase):
    """Half-precision norms can set the clip threshold."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the calibration module once."""
        cls.mod = _load_calibration()

    def test_quantile_accepts_half_dtypes(self) -> None:
        """Quantile of half-precision norms returns a float."""
        for dtype in (torch.bfloat16, torch.float16):
            norms = torch.randn(4, 8, 16, dtype=dtype).norm(dim=-1)
            value = self.mod.activation_norm_quantile(norms, 0.95)
            self.assertIsInstance(value, float)
            self.assertGreater(value, 0.0)

    def test_per_position_clip_keeps_activation_dtype(self) -> None:
        """Clipping bfloat16 activations keeps their dtype."""
        activation = torch.randn(4, 6, 8, dtype=torch.bfloat16)
        clipped, diag = self.mod.per_position_clip(activation, 1.0)
        self.assertEqual(clipped.dtype, torch.bfloat16)
        self.assertEqual(clipped.shape, activation.shape)
        self.assertIsInstance(diag["norm_p95"], float)
