from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict[str, Any], dotted_key: str, value: Any) -> None:
    cur = base
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def resolve_n_jobs(value: Any = "half") -> int:
    cores = os.cpu_count() or 2
    if value in (None, "half"):
        return max(1, cores // 2)
    if value == "all":
        return cores
    return max(1, min(int(value), cores))


def set_thread_limits(n_jobs: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(n_jobs))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n_jobs))
    os.environ.setdefault("MKL_NUM_THREADS", str(n_jobs))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(n_jobs))
    try:
        torch.set_num_threads(n_jobs)
        torch.set_num_interop_threads(max(1, min(2, n_jobs)))
    except RuntimeError:
        pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_dataset_dirs(processed_dir: str | Path) -> list[Path]:
    p = Path(processed_dir)
    if not p.exists():
        return []
    return sorted([x for x in p.iterdir() if x.is_dir() and (x / "arrays.npz").exists()])


def slugify(name: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")
