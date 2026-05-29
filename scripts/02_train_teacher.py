from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from joblib import Parallel, delayed

from gaindistill.data import load_processed
from gaindistill.models import evaluate_binary, fit_teacher
from gaindistill.utils import ensure_dir, list_dataset_dirs, load_config, resolve_n_jobs, set_seed, set_thread_limits, write_json


def run_one(ds_dir: Path, cfg: dict, seed: int, n_jobs: int) -> None:
    set_seed(seed)
    bundle = load_processed(ds_dir)
    out = ensure_dir(Path(cfg["project"]["results_dir"]) / "teachers" / bundle.name / f"seed_{seed}")
    if (out / "metrics.json").exists() and (out / "teacher.joblib").exists():
        return
    fit = fit_teacher(bundle.X_train, bundle.y_train, cfg, n_jobs)
    joblib.dump(fit.model, out / "teacher.joblib")
    metrics = evaluate_binary(fit.model, bundle.X_test, bundle.y_test)
    metrics.update({"train_seconds": fit.train_seconds, "backend": fit.backend, "n_jobs": n_jobs})
    write_json(out / "metrics.json", metrics)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--teacher", default="xgboost")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["teacher"]["learner"] = args.teacher
    cfg["project"]["seed"] = args.seed
    n_jobs = resolve_n_jobs(args.n_jobs or cfg["project"]["n_jobs"])
    set_thread_limits(n_jobs)
    dirs = list_dataset_dirs(cfg["project"]["processed_dir"])
    if args.dataset:
        dirs = [Path(cfg["project"]["processed_dir"]) / args.dataset]
    Parallel(n_jobs=min(n_jobs, len(dirs) or 1))(delayed(run_one)(d, cfg, args.seed, max(1, n_jobs // max(1, len(dirs)))) for d in dirs)
    print(f"Trained teacher(s) for {len(dirs)} dataset(s) with global n_jobs={n_jobs}.")


if __name__ == "__main__":
    main()
