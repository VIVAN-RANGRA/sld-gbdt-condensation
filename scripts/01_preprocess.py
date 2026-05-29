from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from joblib import Parallel, delayed

from gaindistill.data import preprocess_raw_dataset
from gaindistill.utils import load_config, resolve_n_jobs, set_thread_limits


def safe_preprocess(raw_dir: Path, out: str, ds: str, cfg: dict, seed: int) -> str:
    try:
        preprocess_raw_dataset(raw_dir, out, ds, cfg, seed)
        return f"ok:{ds}"
    except Exception as exc:
        return f"skip:{ds}:{exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    n_jobs = resolve_n_jobs(args.n_jobs or cfg["project"]["n_jobs"])
    set_thread_limits(n_jobs)
    raw_dir = Path(args.raw_dir or cfg["project"]["raw_dir"])
    datasets = [args.dataset] if args.dataset else [p.name for p in raw_dir.iterdir() if p.is_dir()]
    if not datasets:
        raise SystemExit(f"No raw datasets found in {raw_dir}. Run scripts/00_download_datasets.py first.")
    out = cfg["project"]["processed_dir"]
    results = Parallel(n_jobs=n_jobs)(delayed(safe_preprocess)(raw_dir, out, ds, cfg, args.seed) for ds in datasets)
    ok = [r for r in results if r.startswith("ok:")]
    skipped = [r for r in results if r.startswith("skip:")]
    for item in skipped:
        print(item)
    print(f"Preprocessed {len(ok)} dataset(s), skipped {len(skipped)} using {n_jobs} worker(s).")


if __name__ == "__main__":
    main()
