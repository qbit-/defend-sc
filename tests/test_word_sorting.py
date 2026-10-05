"""Word-sorting prompts, metrics, paths, and prompt splicing."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import metrics as MET
from src import split_model as SM
from src import util
from src.generate import right_to_left_pad
from src.split_model import splice_prompt_hidden
from src.tasks import get_task
from src.tasks.privacy import positions_from_mask
from src.tasks.word_sorting import (
    answer_only_prompt,
    as_question,
    exclude_targets,
    fit_rows,
    list_char_span,
    subsample,
)


def test_exact_match_ignores_only_space_and_newline() -> None:
    """Exact match keeps case and drops surrounding space."""
    assert MET.exact_match("  oven costume\nextra", "oven costume") == 1.0
    assert MET.exact_match("Oven costume", "oven costume") == 0.0


def test_edit_similarity_is_one_for_equal_and_empty() -> None:
    """Identical strings and two empty strings score 1."""
    assert MET.char_edit_similarity("ab", "ab") == 1.0
    assert MET.char_edit_similarity("", "") == 1.0
    assert MET.char_edit_similarity("ab", "a") == 0.5


def test_answer_only_prompt_ends_at_the_cue() -> None:
    """The 3-shot prompt contains the exemplars and a final cue."""
    question = as_question("stick gelatine")
    prompt = answer_only_prompt(question)
    assert prompt.startswith("Sort a list of words.")
    assert "A: costume counterpart oven" in prompt
    assert prompt.endswith("Q: " + question + "\nA:")
    start, end = list_char_span(prompt)
    assert prompt[start:end] == "stick gelatine"


def test_fit_rows_drop_eval_targets() -> None:
    """Calibration rows exclude BBH targets and stay reproducible."""
    bigbench = [
        {"input": "b a", "target": "a b"},
        {"input": "d c", "target": "c d"},
        {"input": "f e", "target": "e f"},
    ]
    chosen = fit_rows(bigbench, {"a b"}, n=None, seed=0)
    assert [row["target"] for row in chosen] == ["c d", "e f"]
    assert chosen[0]["question"].endswith("List: d c")
    blocked = exclude_targets(bigbench, {"c d"})
    assert subsample(blocked, 1, 0) == subsample(blocked, 1, 0)


def test_privacy_offset_zero_is_kept() -> None:
    """A private span that starts at token 0 still contributes."""
    mask = torch.zeros(1, 4)
    mask[0, 0:3] = 1
    assert positions_from_mask(mask, (0, 1, 5)) == [0, 1]


def test_privacy_offsets_stay_inside_the_span() -> None:
    """Offsets past a short private span are dropped."""
    mask = torch.zeros(2, 8)
    mask[0, 3:5] = 1
    mask[1, 3:8] = 1
    positions = positions_from_mask(mask, (0, 1, 2, 4))
    assert positions == [3, 4, 5, 7]


def test_art_path_groups_model_and_task() -> None:
    """Artifact paths use the model slug and the benchmark name."""
    path = util.art_path(
        "plots", "Qwen/Qwen2.5-0.5B", "word_sorting", "figure.png",
    )
    assert path == (
        util.ART / "plots" / "Qwen_Qwen2.5-0.5B"
        / "word_sorting" / "figure.png"
    )
    assert get_task("sst2").kind == "classification"
    assert get_task("word_sorting").utility_metrics == (
        "exact_match",
        "char_edit_similarity",
    )


def test_splice_replaces_only_the_prompt_span() -> None:
    """Continuation positions are left unchanged."""
    hidden = torch.zeros(2, 6, 3)
    prompt = torch.arange(24, dtype=torch.float32).view(2, 4, 3)
    starts = torch.tensor([1, 0])
    lengths = torch.tensor([2, 3])
    spliced = splice_prompt_hidden(hidden, prompt, starts, lengths)
    assert torch.equal(spliced[0, 1:3], prompt[0, 1:3])
    assert torch.equal(spliced[0, 0], torch.zeros(3))
    assert torch.equal(spliced[0, 3:], torch.zeros(3, 3))
    assert torch.equal(spliced[1, 0:3], prompt[1, 0:3])


def test_right_to_left_pad_keeps_real_tokens() -> None:
    """Real tokens move to the right edge of the batch."""
    ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    hidden = torch.arange(18, dtype=torch.float32).view(2, 3, 3)
    packed_ids, packed_mask, packed_h, starts, lengths = right_to_left_pad(
        ids, mask, hidden, pad_id=0,
    )
    assert torch.equal(packed_ids[0], torch.tensor([0, 1, 2]))
    assert torch.equal(packed_mask[1], torch.tensor([1, 1, 1]))
    assert int(starts[0]) == 1
    assert int(lengths[0]) == 2
    assert torch.equal(packed_h[0, 1:], hidden[0, :2])


def test_metric_columns_share_one_layout() -> None:
    """SST-2 accuracy uses the same clean/raw/private columns."""
    import importlib.util

    path = ROOT / "scripts" / "15_eval_tnsc.py"
    spec = importlib.util.spec_from_file_location("eval15", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    scores = {
        "accuracy": {"clean": 0.8, "raw": 0.4, "private": 0.6},
    }
    fields = module._metric_fields("b1_lowrank_struct", scores)
    assert fields == {
        "accuracy_clean": 0.8,
        "accuracy_raw": 0.4,
        "accuracy_private": "",
    }
    assert "acc_clean_test" not in fields


def test_plot_task_comes_from_the_eval_path() -> None:
    """Both benchmarks are read from artifacts/evals/<model>/<task>."""
    import importlib.util

    path = ROOT / "scripts" / "17_plot_lowrank_private_suppressor.py"
    spec = importlib.util.spec_from_file_location("plot17", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    csv_path = (
        ROOT / "artifacts" / "evals" / "Qwen_Qwen2.5-0.5B"
        / "sst2" / "tnsc_eval.csv"
    )
    assert module._task_from_eval_csv(csv_path) == "sst2"


def test_split_splice_runs_on_tiny_qwen() -> None:
    """A spliced forward returns logits of the full sequence."""
    from tests.test_qwen35_split import _tiny_qwen2

    model = _tiny_qwen2()
    ids = torch.randint(1, 60, (2, 5))
    mask = torch.ones_like(ids)
    hidden = torch.randn(2, 4, 32)
    starts = torch.tensor([0, 0])
    lengths = torch.tensor([4, 4])
    out = SM.split_run_splice(model, ids, 1, hidden, starts, lengths, mask)
    assert tuple(out["logits"].shape) == (2, 5, 64)
