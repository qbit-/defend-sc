"""BIG-bench / BBH word-sorting generation task."""
from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path
from typing import Any

import torch

from src import util
from src.tasks.privacy import resolve_task_positions


def few_shot_exemplars() -> tuple[tuple[str, str], ...]:
    """Return the official BBH answer-only exemplars.

    Returns:
        Three ``(question, answer)`` pairs. None of these answers
        appear in the BBH word-sorting eval set.
    """
    return (
        (
            "Sort the following words alphabetically: "
            "List: oven costume counterpart",
            "costume counterpart oven",
        ),
        (
            "Sort the following words alphabetically: "
            "List: hypochlorite ponderosa phone credulity",
            "credulity hypochlorite phone ponderosa",
        ),
        (
            "Sort the following words alphabetically: "
            "List: newt arson parthia seismography mugho aspect census",
            "arson aspect census mugho newt parthia seismography",
        ),
    )


def as_question(text: str) -> str:
    """Return a BBH-formatted sorting question.

    Args:
        text: Either a bare word list or a question that already
            starts with the BBH instruction.

    Returns:
        Question beginning with ``Sort the following words``.
    """
    stripped = text.strip()
    if stripped.startswith("Sort the following words"):
        return stripped
    return f"Sort the following words alphabetically: List: {stripped}"


def answer_only_prompt(question: str) -> str:
    """Wrap a question in the official answer-only 3-shot prompt.

    Args:
        question: Full sorting question, not a bare word list.

    Returns:
        Prompt ending in ``A:`` so the model continues with the answer.
    """
    lines = ["Sort a list of words.", ""]
    for shot_question, shot_answer in few_shot_exemplars():
        lines.append(f"Q: {shot_question}")
        lines.append(f"A: {shot_answer}")
        lines.append("")
    lines.append(f"Q: {question}")
    lines.append("A:")
    return "\n".join(lines)


def list_char_span(prompt: str) -> tuple[int, int]:
    """Return the character span of the last ``List:`` word list.

    Args:
        prompt: Answer-only prompt.

    Returns:
        Half-open ``[start, end)`` span of the query words.

    Raises:
        ValueError: If the prompt has no ``List:`` marker.
    """
    marker = "List: "
    start_at = prompt.rfind(marker)
    if start_at < 0:
        raise ValueError("prompt has no 'List:' marker")
    start = start_at + len(marker)
    end = prompt.find("\n", start)
    if end < 0:
        end = len(prompt)
    return start, end


def exclude_targets(
    rows: list[dict], blocked: set[str],
) -> list[dict]:
    """Drop rows whose target string is in ``blocked``.

    Args:
        rows: Dicts with a ``target`` field.
        blocked: Target strings that must not be used for fitting.

    Returns:
        Rows whose target is outside ``blocked``.
    """
    return [row for row in rows if row["target"] not in blocked]


def subsample(rows: list[dict], n: int | None, seed: int) -> list[dict]:
    """Return up to ``n`` rows, shuffled with ``seed``.

    Args:
        rows: Candidate rows.
        n: Cap. ``None`` or a cap above ``len(rows)`` keeps every row.
        seed: Shuffle seed.

    Returns:
        A new list. The input order is unchanged when nothing is dropped.
    """
    if n is None or n >= len(rows):
        return list(rows)
    copied = list(rows)
    random.Random(seed).shuffle(copied)
    return copied[:n]


def gold_continuation(target: str) -> str:
    """Return the teacher-forced continuation for a target.

    The prompt ends at ``A:`` and the few-shot lines use ``A: answer``,
    so the gold tokens include the separating space.

    Args:
        target: Sorted word list, without a leading space.

    Returns:
        Continuation text beginning with a space.
    """
    return " " + target.strip()


def _bigbench_cache_path() -> Path:
    """Return the local cache path for the BIG-bench task file.

    Returns:
        JSON path under the project Hugging Face cache.
    """
    return util.ROOT / ".hf_cache" / "bigbench" / "word_sorting_task.json"


def _bigbench_url() -> str:
    """Return the raw BIG-bench word-sorting task URL.

    Returns:
        URL of ``task.json``.
    """
    return (
        "https://raw.githubusercontent.com/google/BIG-bench/main/"
        "bigbench/benchmark_tasks/word_sorting/task.json"
    )


def load_bigbench_rows() -> list[dict]:
    """Download, cache, and return BIG-bench word-sorting examples.

    Returns:
        Dicts with ``input`` and ``target``.
    """
    path = _bigbench_cache_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_bigbench_url(), path)
    payload = json.loads(path.read_text())
    return list(payload["examples"])


def load_bbh_rows() -> list[dict]:
    """Load the 250 BBH word-sorting eval examples.

    Returns:
        Dicts with ``question`` and ``target``.
    """
    from datasets import get_dataset_split_names, load_dataset

    dataset_id = "Joschka/big_bench_hard"
    config = "word_sorting"
    splits = get_dataset_split_names(dataset_id, config)
    split = "train" if "train" in splits else splits[0]
    dataset = load_dataset(dataset_id, config, split=split)
    rows = []
    for row in dataset:
        question = row.get("question", row.get("input"))
        rows.append({"question": question, "target": row["target"]})
    return rows


def fit_rows(
    bigbench_rows: list[dict],
    eval_targets: set[str],
    n: int | None,
    seed: int,
) -> list[dict]:
    """Build the calibration pool from BIG-bench, minus BBH targets.

    Args:
        bigbench_rows: Full BIG-bench example list.
        eval_targets: BBH target strings to hold out.
        n: Optional cap applied after the hold-out.
        seed: Subsample seed.

    Returns:
        Prompt-ready dicts with ``question`` and ``target``.
    """
    blocked = set(eval_targets)
    blocked.update(answer for _question, answer in few_shot_exemplars())
    kept = exclude_targets(bigbench_rows, blocked)
    chosen = subsample(kept, n, seed)
    return [
        {"question": as_question(row["input"]), "target": row["target"]}
        for row in chosen
    ]


def _pad_token_id(tokenizer) -> int:
    """Return a pad id, falling back to EOS or 0.

    Args:
        tokenizer: Hugging Face tokenizer.

    Returns:
        Integer pad id.
    """
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return 0


def _encode_one(tokenizer, prompt: str, max_len: int) -> dict:
    """Tokenize one prompt and mark the query word-list tokens.

    Args:
        tokenizer: Hugging Face tokenizer with offset mapping.
        prompt: Full answer-only prompt.
        max_len: Truncation length.

    Returns:
        Ids, a privacy bit per token, and whether truncation happened.
    """
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_len,
    )
    char_start, char_end = list_char_span(prompt)
    privacy = []
    for start, end in encoded["offset_mapping"]:
        hit = end > start and end > char_start and start < char_end
        privacy.append(1 if hit else 0)
    full = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return {
        "ids": list(encoded["input_ids"]),
        "privacy": privacy,
        "truncated": len(full) > max_len,
    }


def _pad_rows(
    rows: list[dict], pad_id: int, device: str,
) -> dict[str, torch.Tensor]:
    """Right-pad variable-length token rows.

    Args:
        rows: Dicts with ``ids`` and ``privacy`` lists.
        pad_id: Pad token id.
        device: Destination device.

    Returns:
        ``input_ids``, ``attention_mask``, and ``privacy_mask``.
    """
    width = max(len(row["ids"]) for row in rows)
    batch = len(rows)
    ids = torch.full((batch, width), pad_id, dtype=torch.long)
    mask = torch.zeros(batch, width, dtype=torch.long)
    privacy = torch.zeros(batch, width, dtype=torch.long)
    for index, row in enumerate(rows):
        length = len(row["ids"])
        ids[index, :length] = torch.tensor(row["ids"], dtype=torch.long)
        mask[index, :length] = 1
        privacy[index, :length] = torch.tensor(
            row["privacy"], dtype=torch.long,
        )
    return {
        "input_ids": ids.to(device),
        "attention_mask": mask.to(device),
        "privacy_mask": privacy.to(device),
    }


def _gold_batch(
    tokenizer, targets: list[str], device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad teacher-forced continuation ids.

    Args:
        tokenizer: Hugging Face tokenizer.
        targets: Reference answers without a leading space.
        device: Destination device.

    Returns:
        ``gold_ids`` and ``gold_mask``, both ``[B, G]``.
    """
    encoded = [
        tokenizer.encode(
            gold_continuation(target), add_special_tokens=False,
        )
        for target in targets
    ]
    width = max(len(row) for row in encoded)
    batch = len(encoded)
    ids = torch.zeros(batch, width, dtype=torch.long)
    mask = torch.zeros(batch, width, dtype=torch.long)
    for index, row in enumerate(encoded):
        if not row:
            continue
        ids[index, :len(row)] = torch.tensor(row, dtype=torch.long)
        mask[index, :len(row)] = 1
    return ids.to(device), mask.to(device)


class WordSortingTask:
    """BBH word sorting, fit on BIG-bench and scored on BBH."""

    def __init__(self) -> None:
        """Set the benchmark id and string metrics."""
        self.name = "word_sorting"
        self.kind = "generation"
        self.utility_metrics = ("exact_match", "char_edit_similarity")
        self.max_len = 256
        self.max_new_tokens = 64
        self.privacy_offsets = (0, 1, 2, 3, 5, 8, 12, 16)
        self.offset_pattern = ""
        self.privacy_from_mask = True

    def load_split(
        self, split: str, n: int | None, seed: int,
    ) -> list[dict]:
        """Load fit or eval examples.

        Args:
            split: ``train`` for BIG-bench calibration, ``test`` for BBH.
            n: Optional row cap.
            seed: Subsample seed. Eval uses it only when ``n`` is set.

        Returns:
            Dicts with ``question`` and ``target``.
        """
        eval_rows = [
            {
                "question": as_question(row["question"]),
                "target": row["target"],
            }
            for row in load_bbh_rows()
        ]
        if split == "test":
            return subsample(eval_rows, n, seed)
        targets = {row["target"] for row in eval_rows}
        return fit_rows(load_bigbench_rows(), targets, n, seed)

    def encode(
        self,
        tokenizer,
        examples: list[dict],
        max_len: int,
        device: str,
    ) -> dict:
        """Encode answer-only prompts and the gold continuations.

        Args:
            tokenizer: Model tokenizer. Must support offset mappings.
            examples: Dicts with ``question`` and ``target``.
            max_len: Maximum prompt length.
            device: Torch device for the tensors.

        Returns:
            Right-padded ids, masks, targets, and gold token ids.
        """
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        prompts = [
            answer_only_prompt(example["question"]) for example in examples
        ]
        encoded = [
            _encode_one(tokenizer, prompt, max_len) for prompt in prompts
        ]
        if any(row["truncated"] for row in encoded):
            print(
                f"  warning: word-sorting prompt exceeded max_len {max_len}",
                flush=True,
            )
        padded = _pad_rows(encoded, _pad_token_id(tokenizer), device)
        targets = [example["target"] for example in examples]
        gold_ids, gold_mask = _gold_batch(tokenizer, targets, device)
        answer_pos = padded["attention_mask"].sum(dim=1) - 1
        return {
            **padded,
            "answer_pos": answer_pos.to(device),
            "labels": None,
            "targets": targets,
            "gold_ids": gold_ids,
            "gold_mask": gold_mask,
        }

    def resolve_privacy_positions(
        self, encoded: dict, tokenizer: Any = None,
    ) -> list[int]:
        """Return the configured attack positions for this batch.

        Args:
            encoded: Batch from ``encode``, including ``privacy_mask``.
            tokenizer: Required when ``offset_pattern`` is set.

        Returns:
            Absolute token positions the attack should score.
        """
        positions, _score = resolve_task_positions(
            self, encoded, tokenizer,
        )
        return positions

    def estimate_task_subspace(
        self,
        model,
        batch: dict,
        probes: torch.Tensor,
        k: int,
        label_token_ids: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Estimate ``U_T`` from teacher-forced gold-token logits.

        Args:
            model: Causal language model.
            batch: Train encodings, including gold ids.
            probes: Probe directions, ``[n_dirs, H]``.
            k: Split depth.
            label_token_ids: Unused. Present so both tasks share the call.

        Returns:
            ``U_T``, singular values, and the kept rank.
        """
        del label_token_ids
        from src.tasks.generation_geometry import gold_logit_jacobian

        return gold_logit_jacobian(
            model,
            batch["input_ids"],
            batch["attention_mask"],
            batch["clean_a"],
            batch["gold_ids"],
            batch["gold_mask"],
            probes,
            k=k,
        )
