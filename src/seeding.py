from __future__ import annotations

import os
import random


def set_seed(seed: int = 0, *, deterministic: bool = True) -> int:
    """Seed all RNGs and (optionally) pin deterministic GPU algorithms.
    """
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch
    except Exception:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        if os.environ.get("DEFEND_SC_STRICT_DETERMINISM", "") not in ("", "0", "false"):
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass

    return seed
