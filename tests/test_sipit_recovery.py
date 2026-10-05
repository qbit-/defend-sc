"""SIPIT recovered-prompt conditioning."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.attacks import sipit as SIP
from src.noise import GaussianCov

VOCAB = 5


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ids, a privacy mask, and cut observations.

    Position 1 is observed so Euclidean search prefers token 0.
    Later positions match the true token plus the true prefix sum.

    Returns:
        ``ids``, ``mask``, and observations ``[B, T, 1]``.
    """
    ids = torch.tensor([[1, 2, 3], [1, 2, 3]])
    mask = torch.tensor([[1, 1, 1], [1, 0, 1]])
    obs = torch.zeros(2, 3, 1)
    for position in range(1, 3):
        prefix_sum = ids[:, :position].sum(dim=1).float()
        obs[:, position, 0] = ids[:, position].float() + prefix_sum
    obs[:, 1, 0] = 0
    return ids, mask, obs


def _true_clouds(ids: torch.Tensor) -> dict[int, dict]:
    """Build true-prefix clouds for the independent-recovery path.

    Args:
        ids: True prompt tokens, ``[B, T]``.

    Returns:
        Cache keyed by absolute position.
    """
    clouds: dict[int, dict] = {}
    for position in (1, 2):
        prefix_sum = ids[:, :position].sum(dim=1).float()
        cand = torch.arange(VOCAB).view(1, -1).expand(ids.shape[0], -1)
        cand = cand.contiguous()
        values = cand.float() + prefix_sum.unsqueeze(1)
        clouds[position] = {
            "cand": cand,
            "cloud": values.unsqueeze(-1),
            "prior_lp": torch.zeros(ids.shape[0], VOCAB),
        }
    return clouds


class _Recorder:
    """Language model and cloud that record attacker prefixes."""

    def __init__(self) -> None:
        """Start with an empty call log."""
        self.seen: list[tuple[str, torch.Tensor]] = []

    def model(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        """Return logits peaked at the prefix sum.

        Args:
            input_ids: Attacker prefix.
            attention_mask: Ignored.

        Returns:
            Namespace whose ``logits`` are ``[B, T, V]``.
        """
        del attention_mask
        self.seen.append(("model", input_ids.detach().clone()))
        batch, length = input_ids.shape
        logits = torch.arange(VOCAB).float().view(1, 1, -1)
        logits = logits.repeat(batch, length, 1)
        peak = (input_ids.sum(dim=1) % VOCAB).long()
        rows = torch.arange(batch)
        logits[rows, -1, peak] += 1000
        return SimpleNamespace(logits=logits)

    def cloud(
        self,
        model: object,
        prefix_ids: torch.Tensor,
        position: int,
        candidate_ids: torch.Tensor,
        k_split: int,
        chunk_size: int,
    ) -> torch.Tensor:
        """Return cut states equal to the candidate plus the prefix sum.

        Args:
            model: Unused.
            prefix_ids: Full attacker prompt.
            position: Index being guessed.
            candidate_ids: Candidate token ids, ``[B, V']``.
            k_split: Unused split depth.
            chunk_size: Unused chunk size.

        Returns:
            States ``[B, V', 1]``.
        """
        del model, k_split, chunk_size
        prefix = prefix_ids[:, :position].detach().clone()
        self.seen.append(("cloud", prefix))
        prefix_sum = prefix.sum(dim=1).float()
        values = candidate_ids.float() + prefix_sum.unsqueeze(1)
        return values.unsqueeze(-1)


def _install(monkeypatch: pytest.MonkeyPatch, recorder: _Recorder) -> None:
    """Route SIPIT cloud construction through ``recorder``.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        recorder: Prefix recorder.
    """
    monkeypatch.setattr(SIP.M, "vocab_size", lambda _model: VOCAB)
    monkeypatch.setattr(SIP.CS, "candidate_cloud", recorder.cloud)


def _distance(cloud: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Score candidates by negative squared error.

    Args:
        cloud: Candidate states, ``[B, V', H]``.
        obs: Observation, ``[B, H]``.

    Returns:
        Scores, ``[B, V']``.
    """
    delta = cloud - obs.unsqueeze(1)
    return -(delta * delta).sum(dim=-1)


def _prefixes(recorder: _Recorder, kind: str) -> list[torch.Tensor]:
    """Return recorded prefixes of one kind.

    Args:
        recorder: Prefix recorder.
        kind: ``model`` or ``cloud``.

    Returns:
        Prefix tensors in call order.
    """
    return [item for name, item in recorder.seen if name == kind]


def _attack(
    recorder: _Recorder,
    ids: torch.Tensor,
    mask: torch.Tensor,
    obs: torch.Tensor,
    cov: GaussianCov | None,
    prior_weight: float,
    score_fn: Callable[..., torch.Tensor] | None,
    independent_recovery: bool,
    clouds: dict | None,
) -> dict:
    """Run one attack and check that the true ids stay unchanged.

    Args:
        recorder: Stub model and cloud.
        ids: True prompt tokens.
        mask: Privacy mask.
        obs: Cut observations.
        cov: Mahalanobis covariance, or ``None``.
        prior_weight: Log-prior mixture weight.
        score_fn: Optional custom scorer.
        independent_recovery: Use the true prefix when true.
        clouds: Optional true-prefix cache.

    Returns:
        SIPIT metric dict.
    """
    recorder.seen.clear()
    original = ids.clone()
    result = SIP.attack_positions(
        recorder.model, ids, mask, obs, k=1, positions=(2, 1),
        cov=cov, prior_weight=prior_weight, top_k=VOCAB, n_rand=0,
        chunk=4, clouds=clouds, score_fn=score_fn,
        independent_recovery=independent_recovery,
    )
    assert torch.equal(ids, original)
    return result


def test_guess_conditions_distance_attacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vanilla, Mahalanobis, and exact attacks condition on the guess.

    A masked-off row keeps its original token.
    """
    recorder = _Recorder()
    _install(monkeypatch, recorder)
    ids, mask, obs = _batch()
    cov = GaussianCov("iso", hidden=1, sigma0=1.0)
    expected = torch.tensor([[1, 0], [1, 2]])
    calls = (
        (None, 0.0, None),
        (cov, 0.0, None),
        (None, 0.0, _distance),
    )
    for cov_i, prior, score_fn in calls:
        result = _attack(
            recorder, ids, mask, obs, cov_i, prior, score_fn, False, None,
        )
        assert _prefixes(recorder, "model")[0].shape[1] == 1
        assert torch.equal(_prefixes(recorder, "model")[1], expected)
        assert torch.equal(_prefixes(recorder, "cloud")[1], expected)
        assert result["by_position"][2] == 0.5


def test_sequence_map_prior_uses_the_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sequence-MAP's log-prior is conditioned on the recovered token."""
    recorder = _Recorder()
    _install(monkeypatch, recorder)
    ids, mask, obs = _batch()
    cov = GaussianCov("iso", hidden=1, sigma0=1.0)
    result = _attack(
        recorder, ids, mask, obs, cov, 1.0, None, False, None,
    )
    expected = torch.tensor([[1, 1], [1, 2]])
    assert torch.equal(_prefixes(recorder, "model")[1], expected)
    assert torch.equal(_prefixes(recorder, "cloud")[1], expected)
    assert result["by_position"][1] == 0.0
    assert result["by_position"][2] == 0.5


def test_default_attack_is_sequential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting the flag feeds the guess into the next position."""
    recorder = _Recorder()
    _install(monkeypatch, recorder)
    ids, mask, obs = _batch()
    SIP.attack_positions(
        recorder.model, ids, mask, obs, 1, (2, 1),
        top_k=VOCAB, n_rand=0, chunk=4,
    )
    expected = torch.tensor([[1, 0], [1, 2]])
    assert torch.equal(_prefixes(recorder, "cloud")[1], expected)


def test_independent_recovery_keeps_the_true_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent recovery scores every position from the true prefix."""
    recorder = _Recorder()
    _install(monkeypatch, recorder)
    ids, mask, obs = _batch()
    result = _attack(
        recorder, ids, mask, obs, None, 0.0, None, True, _true_clouds(ids),
    )
    assert recorder.seen == []
    assert result["by_position"][1] == 0.0
    assert result["by_position"][2] == 1.0


def _eval_module():
    """Load the TNSC evaluation script.

    Returns:
        Imported module.
    """
    path = ROOT / "scripts" / "15_eval_tnsc.py"
    spec = importlib.util.spec_from_file_location("tnsc_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_recovery_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The eval flag defaults to false and accepts boolean text."""
    module = _eval_module()
    monkeypatch.setattr(sys, "argv", ["15_eval_tnsc.py"])
    assert module.parse_args().independent_recovery is True
    monkeypatch.setattr(
        sys, "argv",
        ["15_eval_tnsc.py", "--independent_recovery", "false"],
    )
    assert module.parse_args().independent_recovery is False
    monkeypatch.setattr(
        sys, "argv",
        ["15_eval_tnsc.py", "--independent_recovery", "0"],
    )
    assert module.parse_args().independent_recovery is False
