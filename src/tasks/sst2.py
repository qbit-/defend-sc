"""SST-2 classification task adapter."""
from __future__ import annotations

from typing import Any

import torch

from src import sst2_data as data
from src import sst2_geometry as geometry
from src.tasks.privacy import resolve_task_positions


class SST2Task:
    """GLUE SST-2 verbalized as a two-way next-token classifier."""

    def __init__(self) -> None:
        """Set the benchmark id and classification metrics."""
        self.name = "sst2"
        self.kind = "classification"
        self.utility_metrics = ("accuracy",)
        self.max_len = 64
        self.max_new_tokens = 0
        self.privacy_offsets = (2, 5, 10, 20)
        self.offset_pattern = ""
        self.privacy_from_mask = False

    def load_split(
        self, split: str, n: int | None, seed: int,
    ) -> list[tuple[str, int]]:
        """Load SST-2 examples.

        Args:
            split: ``train`` or ``test``. ``test`` reads GLUE validation.
            n: Optional cap on the number of rows.
            seed: Shuffle seed used when ``n`` subsamples the split.

        Returns:
            ``(sentence, label)`` pairs.
        """
        source = "validation" if split == "test" else split
        return data.load_sst2(split=source, n=n, seed=seed)

    def encode(
        self,
        tokenizer,
        examples: list[tuple[str, int]],
        max_len: int,
        device: str,
    ) -> dict:
        """Encode reviews and mark every real token as private.

        Args:
            tokenizer: Model tokenizer.
            examples: ``(sentence, label)`` pairs.
            max_len: Maximum prompt length.
            device: Torch device for the encoded batch.

        Returns:
            Ids, mask, answer positions, labels, and a privacy mask.
        """
        ids, mask, answer_pos = data.encode_prompts(
            tokenizer, examples, max_len=max_len, device=device,
        )
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "answer_pos": answer_pos,
            "labels": data.labels_tensor(examples, device=device),
            "privacy_mask": mask.clone(),
            "targets": None,
            "gold_ids": None,
            "gold_mask": None,
        }

    def label_token_ids(self, tokenizer) -> list[int]:
        """Return the negative and positive verbalizer token ids.

        Args:
            tokenizer: Model tokenizer.

        Returns:
            ``[negative_id, positive_id]``.
        """
        verbalizer = data.build_verbalizer(tokenizer)
        return list(verbalizer.label_token_ids)

    def label_words(self, tokenizer) -> tuple[str, str]:
        """Return the verbalizer strings.

        Args:
            tokenizer: Model tokenizer.

        Returns:
            Negative and positive label words.
        """
        verbalizer = data.build_verbalizer(tokenizer)
        return verbalizer.label_words

    def gather_answer_hidden(
        self, hidden: torch.Tensor, answer_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Gather the hidden state at the answer position.

        Args:
            hidden: ``[B, T, H]`` states.
            answer_pos: Index per row.

        Returns:
            ``[B, H]`` answer-position states.
        """
        return data.gather_answer_position(hidden, answer_pos)

    def resolve_privacy_positions(
        self, encoded: dict, tokenizer: Any = None,
    ) -> list[int]:
        """Return the configured attack positions for this batch.

        Args:
            encoded: Batch from ``encode``.
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
        label_token_ids: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Estimate ``U_T`` from the two label logits.

        Args:
            model: Causal language model.
            batch: Train encodings with ids, mask, and answer positions.
            probes: Probe directions, ``[n_dirs, H]``.
            k: Split depth.
            label_token_ids: Verbalizer token ids.

        Returns:
            ``U_T``, singular values, and the kept rank.
        """
        return geometry.label_logit_jacobian_subspace(
            model,
            batch["input_ids"],
            batch["attention_mask"],
            batch["answer_pos"],
            label_token_ids,
            k=k,
            probe_dirs=probes,
            tau=1e-2,
            energy=0.9,
            max_rank=64,
        )
