"""SST-2 data loading + verbalizer + label-token resolution."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from datasets import load_dataset


PROMPT_TEMPLATE = "Review: {sentence}\nSentiment:"


@dataclass
class Verbalizer:
    label_words: tuple[str, str]              # (negative, positive)
    label_token_ids: list[int]                # [neg_id, pos_id]
    label_token_strs: list[str]               # decoded back, for sanity

    @property
    def n_classes(self) -> int:
        return 2


def build_verbalizer(tokenizer, label_words=(" negative", " positive")) -> Verbalizer:
    """Pick label tokens by encoding ' negative' and ' positive' (leading space matters
    for BPE alignment after a colon). Return *first* token of each (the LM head
    chooses the next single token; if the verbalizer needs >1 tokens, the comparison
    becomes multi-step which we avoid in this simple variant)."""
    ids = []
    strs = []
    for w in label_words:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if len(toks) < 1:
            raise ValueError(f"verbalizer {w!r} encoded to no tokens")
        ids.append(toks[0])
        strs.append(tokenizer.decode([toks[0]]))
    if ids[0] == ids[1]:
        raise ValueError(f"verbalizer collision: {label_words} -> same token id {ids[0]}")
    return Verbalizer(label_words=label_words, label_token_ids=ids, label_token_strs=strs)


def load_sst2(split: str = "train", n: int | None = None, seed: int = 0):
    """Return list of (sentence, label) pairs. Label 0=negative, 1=positive."""
    ds = load_dataset("glue", "sst2", split=split)
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    return [(r["sentence"].strip(), int(r["label"])) for r in ds]


def encode_prompts(tokenizer, examples, max_len: int = 64, device="cpu"):
    """Encode examples for batched forward at the answer position.

    Returns (input_ids, attention_mask, answer_position) where answer_position[b]
    is the index of the *last non-pad token* in row b (i.e., the token whose
    next-position prediction is the label). Right-padding with pad_token.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = [PROMPT_TEMPLATE.format(sentence=s) for (s, _) in examples]
    enc = tokenizer(prompts, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=max_len)
    ids = enc["input_ids"]
    mask = enc["attention_mask"]
    answer_pos = mask.sum(dim=1) - 1                  # index of last real token
    return ids.to(device), mask.to(device), answer_pos.to(device)


def labels_tensor(examples, device="cpu") -> torch.Tensor:
    return torch.tensor([lab for (_, lab) in examples], dtype=torch.long, device=device)


def gather_answer_position(hidden: torch.Tensor, answer_pos: torch.Tensor) -> torch.Tensor:
    """hidden: [B, T, H]; answer_pos: [B]. Returns [B, H] at each row's answer_pos."""
    B, T, H = hidden.shape
    idx = answer_pos.view(B, 1, 1).expand(B, 1, H)
    return hidden.gather(1, idx).squeeze(1)
