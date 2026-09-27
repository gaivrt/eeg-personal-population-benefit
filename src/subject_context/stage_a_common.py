"""Small shared contracts for the real-data stage A commands."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[2]


def config():
    return yaml.safe_load((ROOT / "configs/stage_a.yaml").read_text(encoding="utf-8"))


def participants(dataset):
    d = config()["datasets"][dataset]
    return [s for s in range(1, d["subjects"] + 1) if s not in d["exclude"]]


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save_json(path, value):
    def numpy_value(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Unsupported JSON value: {type(obj).__name__}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                              default=numpy_value), encoding="utf-8")
    tmp.replace(path)


def scratch_root():
    path = Path(os.environ["STAGE_A_ROOT"]).resolve()
    if not path.is_relative_to(Path("/scratch").resolve()):
        raise RuntimeError("Stage A data and outputs must reside in /scratch")
    return path


def require_compute():
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("Run this command in a Slurm compute job, never on login")


def failure_stop(failed, cohort_size, threshold):
    if cohort_size <= 0 or not 0 <= failed <= cohort_size:
        raise ValueError("Invalid full-cohort failure count")
    return failed / cohort_size > threshold


def provenance():
    packages = ["numpy", "scipy", "pandas", "scikit-learn", "mne", "moabb", "torch", "pyriemann"]
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config_sha256": sha256(ROOT / "configs/stage_a.yaml"),
        "python": platform.python_version(), "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "packages": {p: importlib.metadata.version(p) for p in packages},
        "gpu": None,
    }
