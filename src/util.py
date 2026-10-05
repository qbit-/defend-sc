from __future__ import annotations
import os, sys, json, time, hashlib, subprocess, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def model_slug(model_id: str) -> str:
    """Return a filesystem slug for a Hugging Face model id.

    Args:
        model_id: Model id such as ``Qwen/Qwen2.5-0.5B``.

    Returns:
        ``model_id`` with ``/`` replaced by ``_``.
    """
    return model_id.replace("/", "_")


def art_path(kind: str, model_id: str, task: str, *parts: str) -> Path:
    """Return an artifact path grouped by model and benchmark.

    Args:
        kind: Top-level artifact folder, such as ``plots``.
        model_id: Hugging Face model id.
        task: Benchmark name, such as ``sst2``.
        *parts: Extra path components under the task folder.

    Returns:
        ``artifacts/<kind>/<model>/<task>/...``.
    """
    path = ART / kind / model_slug(model_id) / task
    for part in parts:
        path = path / part
    return path


def run_id(tag: str) -> str:
    return f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}"


def gpu_info() -> dict:
    try:
        import torch
        return {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception:
        return {}


def git_commit() -> str:
    """Return short HEAD sha or 'n/a' if not in a git tree.

    Resolved against the file's own repo so callers do not need to manage cwd.
    """
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
        return out or "n/a"
    except Exception:
        return "n/a"


def base_meta(script_name: str, **extra) -> dict:
    return {
        "run_id": run_id(Path(script_name).stem),
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "script_name": script_name,
        "gpu": gpu_info(),
        **extra,
    }


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def append_log(name: str, msg: str):
    p = ART / "logs" / f"{name}.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}\n")


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a")
        self.stdout = sys.stdout
    def write(self, s):
        self.f.write(s); self.f.flush(); self.stdout.write(s)
    def flush(self):
        self.f.flush(); self.stdout.flush()
