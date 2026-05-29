from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from gaindistill.data import load_processed
from gaindistill.utils import ensure_dir, load_config


def sample_configs(n_trials: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    configs: list[dict[str, Any]] = []
    for trial in range(n_trials):
        configs.append(
            {
                "trial": trial,
                "n_estimators": int(rng.integers(50, 151)),
                "max_depth": int(rng.integers(2, 6)),
                "learning_rate": float(np.exp(rng.uniform(np.log(0.025), np.log(0.18)))),
                "min_child_weight": float(np.exp(rng.uniform(np.log(0.5), np.log(8.0)))),
                "subsample": float(rng.uniform(0.65, 1.0)),
                "colsample_bytree": float(rng.uniform(0.65, 1.0)),
                "reg_lambda": float(np.exp(rng.uniform(np.log(0.05), np.log(8.0)))),
                "gamma": float(rng.uniform(0.0, 2.0)),
            }
        )
    return configs


def make_model(config: dict[str, Any], seed: int, n_jobs: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=int(config["n_estimators"]),
        max_depth=int(config["max_depth"]),
        learning_rate=float(config["learning_rate"]),
        min_child_weight=float(config["min_child_weight"]),
        subsample=float(config["subsample"]),
        colsample_bytree=float(config["colsample_bytree"]),
        reg_lambda=float(config["reg_lambda"]),
        gamma=float(config["gamma"]),
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=max(1, int(n_jobs)),
        random_state=int(seed),
        verbosity=0,
    )


def fit_score(
    config: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
    n_jobs: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, float]:
    model = make_model(config, seed, n_jobs)
    start = time.perf_counter()
    if sample_weight is None:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    seconds = time.perf_counter() - start
    proba = np.clip(model.predict_proba(X_eval)[:, 1], 1e-7, 1.0 - 1e-7)
    return float(roc_auc_score(y_eval, proba)), float(seconds)


def test_auc_for_config(bundle, config: dict[str, Any], seed: int, n_jobs: int) -> tuple[float, float]:
    X_train = np.vstack([bundle.X_train, bundle.X_val])
    y_train = np.concatenate([bundle.y_train, bundle.y_val])
    return fit_score(config, X_train, y_train, bundle.X_test, bundle.y_test, seed, n_jobs)


def topk_jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    return float(len(sa & sb) / max(1, len(sa | sb)))


def run_dataset(
    cfg: dict,
    dataset: str,
    method: str,
    budget: str,
    distilled_seed: str,
    n_trials: int,
    search_seed: int,
    xgb_threads: int,
    out_dir: Path,
) -> dict[str, Any]:
    bundle = load_processed(Path(cfg["project"]["processed_dir"]) / dataset)
    dist_path = Path(cfg["project"]["results_dir"]) / "distilled" / dataset / method / budget / distilled_seed / "distilled.npz"
    if not dist_path.exists():
        raise FileNotFoundError(f"Missing distilled data: {dist_path}")
    arr = np.load(dist_path)
    X_dist, y_dist = arr["X"], arr["y"]
    weights = arr["weights"] if "weights" in arr.files else None
    configs = sample_configs(n_trials, search_seed)

    rows = []
    for config in configs:
        full_auc, full_seconds = fit_score(
            config,
            bundle.X_train,
            bundle.y_train,
            bundle.X_val,
            bundle.y_val,
            search_seed + int(config["trial"]),
            xgb_threads,
        )
        dist_auc, dist_seconds = fit_score(
            config,
            X_dist,
            y_dist,
            bundle.X_val,
            bundle.y_val,
            search_seed + int(config["trial"]),
            xgb_threads,
            sample_weight=weights,
        )
        row = {"dataset": dataset, "method": method, "budget": budget, "seed": distilled_seed, **config}
        row.update(
            {
                "full_val_auc": full_auc,
                "distilled_val_auc": dist_auc,
                "full_seconds": full_seconds,
                "distilled_seconds": dist_seconds,
            }
        )
        rows.append(row)
    trials = pd.DataFrame(rows)
    ensure_dir(out_dir)
    trials.to_csv(out_dir / f"icdm_hpo_trials_{dataset}.csv", index=False)

    full_top = trials.sort_values("full_val_auc", ascending=False).head(3)["trial"].astype(int).tolist()
    dist_top = trials.sort_values("distilled_val_auc", ascending=False).head(3)["trial"].astype(int).tolist()
    full_best_trial = int(full_top[0])
    dist_best_trial = int(dist_top[0])
    full_best_config = configs[full_best_trial]
    dist_best_config = configs[dist_best_trial]
    full_best_test_auc, full_best_test_seconds = test_auc_for_config(bundle, full_best_config, search_seed + 1000, xgb_threads)
    dist_selected_test_auc, dist_selected_test_seconds = test_auc_for_config(bundle, dist_best_config, search_seed + 2000, xgb_threads)
    corr = spearmanr(trials["full_val_auc"], trials["distilled_val_auc"]).correlation
    return {
        "dataset": dataset,
        "method": method,
        "budget": budget,
        "seed": distilled_seed,
        "n_trials": n_trials,
        "top3_jaccard": topk_jaccard(full_top, dist_top),
        "top1_match": float(full_best_trial == dist_best_trial),
        "score_spearman": float(0.0 if pd.isna(corr) else corr),
        "full_hpo_seconds": float(trials["full_seconds"].sum()),
        "distilled_hpo_seconds": float(trials["distilled_seconds"].sum()),
        "hpo_speedup": float(trials["full_seconds"].sum() / max(1e-9, trials["distilled_seconds"].sum())),
        "full_best_trial": full_best_trial,
        "distilled_best_trial": dist_best_trial,
        "full_best_val_auc": float(trials.loc[trials["trial"] == full_best_trial, "full_val_auc"].iloc[0]),
        "distilled_best_val_auc": float(trials.loc[trials["trial"] == dist_best_trial, "distilled_val_auc"].iloc[0]),
        "full_best_test_auc": full_best_test_auc,
        "distilled_selected_test_auc": dist_selected_test_auc,
        "test_auc_gap": float(full_best_test_auc - dist_selected_test_auc),
        "full_best_test_seconds": full_best_test_seconds,
        "distilled_selected_test_seconds": dist_selected_test_seconds,
    }


def md_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.{float_digits}f}")
    cols = list(out.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def make_figure(summary: pd.DataFrame, out_figs: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.scatter(summary["hpo_speedup"], summary["top3_jaccard"], s=58)
    for _, row in summary.iterrows():
        ax.annotate(str(row["dataset"]), (row["hpo_speedup"], row["top3_jaccard"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axhline(0.8, color="black", alpha=0.35, linewidth=1.0)
    ax.axvline(10.0, color="black", alpha=0.35, linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Full-search seconds / distilled-search seconds")
    ax.set_ylabel("Top-3 config Jaccard")
    ax.set_title("HPO speedup versus agreement")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_figs / "icdm_hpo_speedup_agreement.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/focused.yaml")
    ap.add_argument("--paper-materials", default="paper_materials")
    ap.add_argument(
        "--datasets",
        default="adult,australian,bank_marketing,breast_w,credit_g,diabetes,electricity,kr_vs_kp,pc1,phoneme,qsar_biodeg,spambase",
    )
    ap.add_argument("--method", default="histdistill_refined")
    ap.add_argument("--budget", default="budget_50")
    ap.add_argument("--distilled-seed", default="seed_0")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--search-seed", type=int, default=2026)
    ap.add_argument("--xgb-threads", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config(args.config)
    materials = ensure_dir(args.paper_materials)
    out_tables = ensure_dir(materials / "tables")
    out_notes = ensure_dir(materials / "notes")
    out_figs = ensure_dir(materials / "figures")
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    rows = []
    for dataset in datasets:
        print(f"Running HPO diagnostic for {dataset}...", flush=True)
        rows.append(
            run_dataset(
                cfg,
                dataset,
                args.method,
                args.budget,
                args.distilled_seed,
                args.trials,
                args.search_seed,
                args.xgb_threads,
                out_tables,
            )
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_tables / "icdm_hpo_summary.csv", index=False)
    make_figure(summary, out_figs)

    overall = pd.DataFrame(
        [
            {
                "datasets": int(len(summary)),
                "mean_top3_jaccard": float(summary["top3_jaccard"].mean()),
                "mean_top1_match": float(summary["top1_match"].mean()),
                "mean_score_spearman": float(summary["score_spearman"].mean()),
                "median_hpo_speedup": float(summary["hpo_speedup"].median()),
                "mean_test_auc_gap": float(summary["test_auc_gap"].mean()),
            }
        ]
    )
    overall.to_csv(out_tables / "icdm_hpo_overall.csv", index=False)
    lines = [
        "# ICDM HPO Diagnostic",
        "",
        f"XGBoost random-search HPO on {len(summary)} datasets with {args.trials} shared configurations. "
        f"Distilled search uses `{args.method}` at `{args.budget}` / `{args.distilled_seed}`.",
        "",
        "## Overall",
        "",
        md_table(overall),
        "",
        "## Per Dataset",
        "",
        md_table(
            summary[
                [
                    "dataset",
                    "top3_jaccard",
                    "top1_match",
                    "score_spearman",
                    "hpo_speedup",
                    "full_best_test_auc",
                    "distilled_selected_test_auc",
                    "test_auc_gap",
                ]
            ]
        ),
        "",
        "## Files",
        "",
        "- `tables/icdm_hpo_summary.csv`",
        "- `tables/icdm_hpo_overall.csv`",
        "- `tables/icdm_hpo_trials_<dataset>.csv`",
        "- `figures/icdm_hpo_speedup_agreement.png`",
        "",
    ]
    (out_notes / "ICDM_HPO_DIAGNOSTIC.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote HPO diagnostics to paper_materials.")


if __name__ == "__main__":
    main()
