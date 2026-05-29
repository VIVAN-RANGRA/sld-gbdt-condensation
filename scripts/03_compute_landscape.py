from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from joblib import Parallel, delayed

from gaindistill.data import load_processed
from gaindistill.landscape import compute_landscape_from_arrays, load_teacher, save_landscape
from gaindistill.utils import list_dataset_dirs, load_config, resolve_n_jobs, set_thread_limits


def run_one(ds_dir: Path, cfg: dict, seed: int, force: bool) -> None:
    bundle = load_processed(ds_dir)
    out = Path(cfg["project"]["results_dir"]) / "landscapes" / bundle.name / f"seed_{seed}"
    if (out / "landscape.npz").exists() and not force:
        return
    teacher_path = Path(cfg["project"]["results_dir"]) / "teachers" / bundle.name / f"seed_{seed}" / "teacher.joblib"
    teacher = load_teacher(teacher_path)
    landscape = compute_landscape_from_arrays(
        bundle.X_train,
        bundle.y_train,
        bundle.meta,
        teacher,
        float(cfg["teacher"]["reg_lambda"]),
        float(cfg["teacher"]["gamma"]),
        cfg,
        seed,
    )
    save_landscape(out, landscape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--teacher", default="xgboost")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    n_jobs = resolve_n_jobs(args.n_jobs or cfg["project"]["n_jobs"])
    set_thread_limits(n_jobs)
    dirs = list_dataset_dirs(cfg["project"]["processed_dir"])
    if args.dataset:
        dirs = [Path(cfg["project"]["processed_dir"]) / args.dataset]
    Parallel(n_jobs=min(n_jobs, len(dirs) or 1))(delayed(run_one)(d, cfg, args.seed, args.force) for d in dirs)
    print(f"Computed landscapes for {len(dirs)} dataset(s).")


if __name__ == "__main__":
    main()
