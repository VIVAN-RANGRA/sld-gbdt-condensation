from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from joblib import Parallel, delayed

from gaindistill.baselines import BASELINES
from gaindistill.data import load_processed
from gaindistill.landscape import load_landscape
from gaindistill.utils import ensure_dir, list_dataset_dirs, load_config, resolve_n_jobs, set_thread_limits, write_json


def run_method(ds_dir: Path, cfg: dict, method: str, budget: int, seed: int, force: bool = False) -> None:
    bundle = load_processed(ds_dir)
    save_dir = ensure_dir(Path(cfg["project"]["results_dir"]) / "distilled" / bundle.name / method / f"budget_{budget}" / f"seed_{seed}")
    if (save_dir / "distilled.npz").exists() and not force:
        return
    teacher_path = Path(cfg["project"]["results_dir"]) / "teachers" / bundle.name / f"seed_{seed}" / "teacher.joblib"
    teacher = joblib.load(teacher_path) if teacher_path.exists() else None
    landscape_path = Path(cfg["project"]["results_dir"]) / "landscapes" / bundle.name / f"seed_{seed}"
    landscape = load_landscape(landscape_path) if (landscape_path / "landscape.npz").exists() else None
    out = BASELINES[method](bundle.X_train, bundle.y_train, budget, seed, meta=bundle.meta, teacher=teacher, landscape=landscape)
    np.savez_compressed(save_dir / "distilled.npz", X=out["X"], y=out["y"], weights=out["weights"])
    write_json(save_dir / "metadata.json", {"rows": int(len(out["y"])), "dataset": bundle.name, "method": method})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--budget-per-class", type=int, default=None)
    ap.add_argument(
        "--methods",
        default="full_data,random,k_center,herding,gradient_sampling,distribution_matching,tree_region,gain_herding,gain_weighted_coreset,gaindistill_hybrid,gain_path_herding,gaindistill_path_hybrid",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    budget = int(args.budget_per_class or cfg["distill"]["budget_per_class"])
    n_jobs = resolve_n_jobs(args.n_jobs or cfg["project"]["n_jobs"])
    set_thread_limits(n_jobs)
    dirs = list_dataset_dirs(cfg["project"]["processed_dir"])
    if args.dataset:
        dirs = [Path(cfg["project"]["processed_dir"]) / args.dataset]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    tasks = [(d, m) for d in dirs for m in methods]
    Parallel(n_jobs=min(n_jobs, len(tasks) or 1))(delayed(run_method)(d, cfg, m, budget, args.seed, args.force) for d, m in tasks)
    print(f"Wrote {len(tasks)} baseline dataset(s) at budget {budget}/class.")


if __name__ == "__main__":
    main()
