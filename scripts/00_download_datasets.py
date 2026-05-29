from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaindistill.data import create_smoke_dataset, download_openml_curated, download_openml_suite, download_pmlb
from gaindistill.utils import ensure_dir, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--source", choices=["openml_cc18", "openml_curated", "pmlb", "smoke"], default="openml_curated")
    ap.add_argument("--suite-id", type=int, default=99)
    ap.add_argument("--max-datasets", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out or Path(cfg["project"]["raw_dir"]) / args.source)
    ensure_dir(out)
    saved: list[str] = []
    if args.source == "openml_cc18":
        saved = download_openml_suite(args.suite_id, out, args.max_datasets)
        if not saved:
            saved = download_openml_curated(cfg["data"]["curated_openml_names"], out, args.max_datasets)
    elif args.source == "openml_curated":
        saved = download_openml_curated(cfg["data"]["curated_openml_names"], out, args.max_datasets)
    elif args.source == "pmlb":
        saved = download_pmlb(out, args.max_datasets)
    else:
        saved = [create_smoke_dataset(out, seed=args.seed)]
    if not saved:
        saved = [create_smoke_dataset(out, seed=args.seed)]
        print("Remote downloads did not produce a dataset; created smoke dataset instead.")
    print(f"Saved {len(saved)} dataset(s) in {out}: {', '.join(saved)}")


if __name__ == "__main__":
    main()
