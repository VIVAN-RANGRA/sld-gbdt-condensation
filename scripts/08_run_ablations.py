from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from joblib import Parallel, delayed

from gaindistill.data import load_processed
from gaindistill.distill import distill_root, save_distilled
from gaindistill.landscape import load_landscape
from gaindistill.utils import deep_update, list_dataset_dirs, load_config, resolve_n_jobs, set_thread_limits


def run_variant(ds_dir: Path, base_cfg: dict, ab_cfg: dict, variant: str, changes: dict, seed: int, budget: int, worker_threads: int) -> None:
    set_thread_limits(worker_threads)
    cfg = copy.deepcopy(base_cfg)
    for k, v in ab_cfg.get("fast", {}).items():
        deep_update(cfg, k, v)
    for k, v in changes.items():
        deep_update(cfg, k, v)
    bundle = load_processed(ds_dir)
    landscape = load_landscape(Path(cfg["project"]["results_dir"]) / "landscapes" / bundle.name / f"seed_{seed}")
    distilled = distill_root(bundle.X_train, bundle.y_train, bundle.meta, landscape, cfg, seed, budget)
    out = Path(cfg["project"]["results_dir"]) / "distilled" / bundle.name / f"ablation_{variant}" / f"budget_{budget}" / f"seed_{seed}"
    save_distilled(out, distilled, bundle.meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ablations.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--budget-per-class", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", default=None)
    args = ap.parse_args()
    ab_cfg = load_config(args.config)
    base_cfg = load_config(ab_cfg.get("base_config", "configs/default.yaml"))
    n_jobs = resolve_n_jobs(args.n_jobs or base_cfg["project"]["n_jobs"])
    set_thread_limits(n_jobs)
    dirs = list_dataset_dirs(base_cfg["project"]["processed_dir"])
    if args.dataset:
        dirs = [Path(base_cfg["project"]["processed_dir"]) / args.dataset]
    elif args.datasets and args.datasets.startswith("selected_"):
        dirs = dirs[: int(args.datasets.split("_")[1])]
    variants = ab_cfg["loss_components"]
    tasks = [(d, v, changes) for d in dirs for v, changes in variants.items()]
    worker_threads = max(1, n_jobs // max(1, min(len(tasks), n_jobs)))
    Parallel(n_jobs=min(n_jobs, len(tasks) or 1))(
        delayed(run_variant)(d, base_cfg, ab_cfg, v, changes or {}, args.seed, args.budget_per_class, worker_threads) for d, v, changes in tasks
    )
    print(f"Generated {len(tasks)} ablation distilled dataset(s). Run downstream and fidelity scripts next.")


if __name__ == "__main__":
    main()
