from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from joblib import Parallel, delayed

from gaindistill.data import load_processed
from gaindistill.distill import distill_multi_probe, save_distilled
from gaindistill.landscape import load_landscape, load_teacher
from gaindistill.utils import deep_update, load_config, resolve_n_jobs, set_seed, set_thread_limits


VARIANTS: dict[str, dict[str, object]] = {
    "math_full": {},
    "math_no_probes": {
        "distill.weight_gain": 0.0,
        "distill.weight_margin": 0.0,
        "distill.weight_newton": 0.0,
        "distill.weight_child": 0.0,
        "distill.weight_hist": 0.0,
        "distill.max_train_probes": 1,
    },
    "math_root_only": {
        "distill.max_train_probes": 1,
    },
    "math_no_soft_routing": {
        "distill.use_soft_routing": False,
        "distill.use_soft_teacher_routing": False,
    },
    "math_no_anchor": {
        "distill.weight_anchor": 0.0,
        "distill.weight_label_anchor": 0.0,
    },
    "math_no_newton": {
        "distill.weight_newton": 0.0,
    },
    "math_no_hist": {
        "distill.weight_hist": 0.0,
    },
    "math_tuned": {
        "distill.weight_newton": 0.0,
        "distill.decode_strategy": "anchored_class_median",
    },
    "math_tuned_no_hist": {
        "distill.weight_newton": 0.0,
        "distill.weight_hist": 0.0,
        "distill.decode_strategy": "anchored_class_median",
    },
    "math_tuned_no_anchor": {
        "distill.weight_newton": 0.0,
        "distill.weight_anchor": 0.0,
        "distill.weight_label_anchor": 0.0,
        "distill.decode_strategy": "anchored_class_median",
    },
    "math_tuned_more_probes": {
        "distill.weight_newton": 0.0,
        "distill.max_train_probes": 20,
        "distill.decode_strategy": "anchored_class_median",
    },
}


def run_one(dataset: str, variant: str, changes: dict[str, object], cfg: dict, budget: int, seed: int, force: bool, worker_threads: int) -> None:
    set_thread_limits(worker_threads)
    set_seed(seed)
    local_cfg = copy.deepcopy(cfg)
    for key, value in changes.items():
        deep_update(local_cfg, key, value)
    bundle = load_processed(Path(local_cfg["project"]["processed_dir"]) / dataset)
    out = Path(local_cfg["project"]["results_dir"]) / "distilled" / bundle.name / variant / f"budget_{budget}" / f"seed_{seed}"
    if (out / "distilled.npz").exists() and not force:
        return
    landscape = load_landscape(Path(local_cfg["project"]["results_dir"]) / "landscapes" / bundle.name / f"seed_{seed}")
    teacher = load_teacher(Path(local_cfg["project"]["results_dir"]) / "teachers" / bundle.name / f"seed_{seed}" / "teacher.joblib")
    distilled = distill_multi_probe(bundle.X_train, bundle.y_train, bundle.meta, landscape, local_cfg, seed, budget, teacher=teacher)
    save_distilled(out, distilled, bundle.meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/focused.yaml")
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--budgets", default="50")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--n-jobs", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    datasets = [x.strip() for x in (args.datasets or ",".join(cfg["data"]["focused_binary_datasets"])).split(",") if x.strip()]
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    n_jobs = resolve_n_jobs(args.n_jobs or cfg["project"]["n_jobs"])
    set_thread_limits(n_jobs)
    tasks = [(ds, v, VARIANTS[v], b, s) for ds in datasets for v in variants for b in budgets for s in seeds]
    worker_threads = max(1, n_jobs // max(1, min(n_jobs, len(tasks))))
    Parallel(n_jobs=min(n_jobs, len(tasks) or 1))(
        delayed(run_one)(ds, v, changes, cfg, b, s, args.force, worker_threads) for ds, v, changes, b, s in tasks
    )
    print(f"Generated {len(tasks)} math ablation artifact(s).")


if __name__ == "__main__":
    main()
