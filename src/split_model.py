"""Split execution via forward-pre-hooks.

Approach:
  - We never re-implement the model forward; we always call `model(input_ids).logits`.
  - To capture the cut activation a_k, we install a forward_pre_hook on block k that
    records hidden_states (the first positional arg passed to the block).
  - To inject noise / replace a_k, we install a forward_pre_hook on block k that
    returns a *new* hidden_states tensor.
  - This guarantees the split run is *exactly* the full run when a_k is not
    perturbed, and degrades gracefully when it is. No rotary plumbing required.

Special case k = 0: the cut activation is the embedding output. We hook block 0.

If `k == n_layers`, the cut activation is the input to the final norm (no further
blocks). We hook the final norm with a forward_pre_hook.
"""
from __future__ import annotations
import torch
from . import models as M


class _Capture:
    def __init__(self):
        self.hidden = None

    def hook(self, module, args, kwargs):
        # Block forward pre-hook: first positional arg is hidden_states.
        if args and isinstance(args[0], torch.Tensor):
            self.hidden = args[0].detach()
        else:
            # some HF blocks pass hidden_states as kwarg
            h = kwargs.get("hidden_states", None)
            if h is not None:
                self.hidden = h.detach()


class _Replace:
    """Hook that replaces hidden_states (first positional or hidden_states kwarg)."""
    def __init__(self, new_hidden: torch.Tensor):
        self.new_hidden = new_hidden

    def hook(self, module, args, kwargs):
        if args and isinstance(args[0], torch.Tensor):
            new_args = (self.new_hidden,) + args[1:]
            return new_args, kwargs
        if "hidden_states" in kwargs:
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = self.new_hidden
            return args, kwargs
        return args, kwargs


def _hook_target(model, k: int):
    """Return the module to hook to capture/replace the cut activation at depth k.

    k = 0 .. L-1: hook block k (intercepts what arrives at block k).
    k = L:        hook final norm (intercepts what arrives at final norm).
    """
    L = M.n_layers(model)
    assert 0 <= k <= L
    if k < L:
        return M.get_blocks(model)[k]
    return M.final_norm(model)


@torch.no_grad()
def capture_a_k(model, input_ids, k: int, attention_mask=None) -> torch.Tensor:
    cap = _Capture()
    h = _hook_target(model, k).register_forward_pre_hook(cap.hook, with_kwargs=True)
    try:
        model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        h.remove()
    assert cap.hidden is not None, f"failed to capture a_k at k={k}"
    return cap.hidden


@torch.no_grad()
def split_run(model, input_ids, k: int, hidden_override=None,
              attention_mask=None) -> dict:
    """Run split protocol; if hidden_override is given, replace a_k with it.

    Returns dict with keys: a_k, logits, server_out (final hidden before norm).
    server_out is captured from the input to the final norm.
    """
    cap_ak = _Capture()
    cap_final = _Capture()
    handles = []
    target_k = _hook_target(model, k)
    if hidden_override is None:
        handles.append(target_k.register_forward_pre_hook(cap_ak.hook, with_kwargs=True))
    else:
        rep = _Replace(hidden_override)
        handles.append(target_k.register_forward_pre_hook(rep.hook, with_kwargs=True))
        # also capture the (now-replaced) hidden into a_k so reports are consistent
        handles.append(target_k.register_forward_pre_hook(cap_ak.hook, with_kwargs=True))
    # capture server output = input to final norm
    handles.append(M.final_norm(model).register_forward_pre_hook(cap_final.hook, with_kwargs=True))
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()
    return {
        "a_k": cap_ak.hidden,
        "server_out": cap_final.hidden,
        "logits": out.logits,
    }
