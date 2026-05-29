from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, roc_auc_score

from gaindistill.data import load_processed
from gaindistill.landscape import candidate_grid, extract_xgb_trees
from gaindistill.utils import ensure_dir, load_config, resolve_n_jobs, set_thread_limits


def import_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PHASE2 = import_script(Path(__file__).resolve().parent / "32_run_phase2_experiments.py", "phase2_partial")
BREADTH = import_script(Path(__file__).resolve().parent / "29_run_breadth_experiments.py", "breadth_phase2")


LABELS = {
    "histdistill": "HistDistill",
    "random": "Random",
    "herding": "Herding",
    "k_center": "K-center",
    "full_data": "Full data",
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def logit_mean(y: np.ndarray) -> float:
    p = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
    return math.log(p / (1.0 - p))


def threshold(meta: dict[str, Any], feature: int, bin_id: int) -> float:
    return float(meta["bins"][int(feature)]["thresholds"][int(bin_id)])


def safe_spearman(df: pd.DataFrame, x: str, y: str) -> float:
    cols = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(cols) < 3 or cols[x].nunique() < 2 or cols[y].nunique() < 2:
        return np.nan
    corr = spearmanr(cols[x], cols[y]).correlation
    return float(corr) if not pd.isna(corr) else np.nan


@dataclass
class DosedNode:
    indices: np.ndarray
    depth: int
    feature: int | None = None
    threshold: float | None = None
    left: "DosedNode | None" = None
    right: "DosedNode | None" = None
    value: float = 0.0


def split_gains_for_indices(
    X: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    idx: np.ndarray,
    candidates: np.ndarray,
    meta: dict[str, Any],
    reg_lambda: float,
    gamma: float,
    min_leaf: int,
) -> np.ndarray:
    G0 = float(g[idx].sum())
    H0 = float(h[idx].sum())
    parent_score = (G0 * G0) / (H0 + reg_lambda + 1e-8)
    gains = np.full(len(candidates), -np.inf, dtype=np.float64)
    for pos, (feature, bin_id) in enumerate(candidates):
        thr = threshold(meta, int(feature), int(bin_id))
        left_local = X[idx, int(feature)] <= thr
        n_left = int(left_local.sum())
        n_right = int(len(idx) - n_left)
        if n_left < min_leaf or n_right < min_leaf:
            continue
        left_idx = idx[left_local]
        GL = float(g[left_idx].sum())
        HL = float(h[left_idx].sum())
        GR = G0 - GL
        HR = H0 - HL
        gains[pos] = 0.5 * ((GL * GL) / (HL + reg_lambda + 1e-8) + (GR * GR) / (HR + reg_lambda + 1e-8) - parent_score) - gamma
    return gains


def build_dosed_tree(
    X: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    idx: np.ndarray,
    candidates: np.ndarray,
    meta: dict[str, Any],
    reg_lambda: float,
    gamma: float,
    max_depth: int,
    min_leaf: int,
    dose_ratio: float,
    rng: np.random.Generator,
    depth: int = 0,
    stats: list[dict[str, float]] | None = None,
) -> DosedNode:
    node = DosedNode(indices=idx, depth=depth)
    if stats is None:
        stats = []
    if depth >= max_depth or len(idx) < 2 * min_leaf:
        return node
    gains = split_gains_for_indices(X, g, h, idx, candidates, meta, reg_lambda, gamma, min_leaf)
    valid = np.isfinite(gains)
    if int(valid.sum()) < 1:
        return node
    order = np.argsort(-gains)
    best = int(order[0])
    second_gain = float(gains[order[1]]) if len(order) > 1 and np.isfinite(gains[order[1]]) else float(gains[best])
    margin = max(0.0, float(gains[best] - second_gain))
    eps = float(dose_ratio) * margin
    noisy = gains.copy()
    noise = np.zeros_like(gains)
    if eps > 0:
        noise[valid] = rng.uniform(-eps, eps, size=int(valid.sum()))
        noisy[valid] = gains[valid] + noise[valid]
    chosen = int(np.argmax(noisy))
    feature, bin_id = map(int, candidates[chosen])
    thr = threshold(meta, feature, bin_id)
    left_mask = X[idx, feature] <= thr
    left_idx = idx[left_mask]
    right_idx = idx[~left_mask]
    if len(left_idx) < min_leaf or len(right_idx) < min_leaf:
        return node
    best_gain = float(gains[best])
    chosen_gain = float(gains[chosen])
    regret = max(0.0, best_gain - chosen_gain)
    scale = max(1.0, abs(best_gain), float(np.nanmax(np.abs(gains[valid]))))
    stats.append(
        {
            "agreement": float(regret <= 1e-12),
            "split_regret_norm": regret / scale,
            "margin": margin,
            "injected_sld": float(np.max(np.abs(noise[valid]))) if eps > 0 else 0.0,
            "depth": float(depth),
        }
    )
    node.feature = feature
    node.threshold = thr
    node.left = build_dosed_tree(X, g, h, left_idx, candidates, meta, reg_lambda, gamma, max_depth, min_leaf, dose_ratio, rng, depth + 1, stats)
    node.right = build_dosed_tree(X, g, h, right_idx, candidates, meta, reg_lambda, gamma, max_depth, min_leaf, dose_ratio, rng, depth + 1, stats)
    return node


def assign_leaf_values(node: DosedNode, g: np.ndarray, h: np.ndarray, reg_lambda: float, learning_rate: float) -> None:
    if node.feature is None or node.left is None or node.right is None:
        idx = node.indices
        node.value = float(-g[idx].sum() / (h[idx].sum() + reg_lambda + 1e-8) * learning_rate)
        return
    assign_leaf_values(node.left, g, h, reg_lambda, learning_rate)
    assign_leaf_values(node.right, g, h, reg_lambda, learning_rate)


def predict_tree_values(node: DosedNode, X: np.ndarray) -> np.ndarray:
    out = np.empty(X.shape[0], dtype=np.float64)

    def fill(cur: DosedNode, rows: np.ndarray) -> None:
        if cur.feature is None or cur.left is None or cur.right is None:
            out[rows] = float(cur.value)
            return
        mask = X[rows, int(cur.feature)] <= float(cur.threshold)
        fill(cur.left, rows[mask])
        fill(cur.right, rows[~mask])

    fill(node, np.arange(X.shape[0], dtype=int))
    return out


def run_e18_downstream_cell(
    cfg: dict[str, Any],
    dataset: str,
    seed: int,
    dose_ratio: float,
    repeat: int,
    n_estimators: int,
    max_candidates: int,
) -> dict[str, Any]:
    set_thread_limits(1)
    bundle = load_processed(Path(cfg["project"]["processed_dir"]) / dataset)
    X_train = np.asarray(bundle.X_train, dtype=np.float32)
    y_train = np.asarray(bundle.y_train, dtype=np.float64)
    X_test = np.asarray(bundle.X_test, dtype=np.float32)
    y_test = np.asarray(bundle.y_test, dtype=np.int64)
    meta = bundle.meta
    candidates = candidate_grid(meta)
    if len(candidates) > max_candidates:
        # Deterministic spread over the full grid keeps this controlled while avoiding
        # a very expensive Python CART implementation on wide tables.
        keep = np.linspace(0, len(candidates) - 1, max_candidates).round().astype(int)
        candidates = candidates[np.unique(keep)]
    rng = np.random.default_rng(10_000 * seed + 997 * repeat + int(round(dose_ratio * 1000)))
    learning_rate = float(cfg["teacher"]["learning_rate"])
    reg_lambda = float(cfg["teacher"]["reg_lambda"])
    gamma = float(cfg["teacher"]["gamma"])
    max_depth = int(cfg["teacher"]["max_depth"])
    min_leaf = max(5, min(32, int(0.02 * len(y_train))))
    margin_train = np.full(len(y_train), logit_mean(y_train), dtype=np.float64)
    margin_test = np.full(len(y_test), logit_mean(y_train), dtype=np.float64)
    node_stats: list[dict[str, float]] = []
    for _ in range(int(n_estimators)):
        p = np.clip(sigmoid(margin_train), 1e-6, 1.0 - 1e-6)
        g = p - y_train
        h = np.maximum(p * (1.0 - p), 1e-6)
        tree_stats: list[dict[str, float]] = []
        tree = build_dosed_tree(
            X_train,
            g,
            h,
            np.arange(len(y_train), dtype=int),
            candidates,
            meta,
            reg_lambda,
            gamma,
            max_depth,
            min_leaf,
            float(dose_ratio),
            rng,
            stats=tree_stats,
        )
        assign_leaf_values(tree, g, h, reg_lambda, learning_rate)
        margin_train += predict_tree_values(tree, X_train)
        margin_test += predict_tree_values(tree, X_test)
        node_stats.extend(tree_stats)
    proba = np.clip(sigmoid(margin_test), 1e-7, 1.0 - 1e-7)
    if node_stats:
        stats_df = pd.DataFrame(node_stats)
        agreement = float(stats_df["agreement"].mean())
        root_agreement = float(stats_df.loc[stats_df["depth"].eq(0.0), "agreement"].mean())
        regret = float(stats_df["split_regret_norm"].mean())
        injected = float(stats_df["injected_sld"].mean())
        node_count = int(len(stats_df))
    else:
        agreement = root_agreement = regret = injected = np.nan
        node_count = 0
    return {
        "dataset": dataset,
        "seed": f"seed_{seed}",
        "repeat": int(repeat),
        "dose_ratio": float(dose_ratio),
        "auroc": float(roc_auc_score(y_test, proba)),
        "agreement": agreement,
        "root_agreement": root_agreement,
        "split_regret_norm": regret,
        "injected_sld": injected,
        "node_count": node_count,
        "n_estimators": int(n_estimators),
        "max_candidates": int(max_candidates),
        "train_rows": int(len(y_train)),
    }


def refit_xgb_regression_leaves(model: Any, X_full: np.ndarray, y_full: np.ndarray, X_test: np.ndarray, X_sub: np.ndarray) -> dict[str, Any]:
    trees = extract_xgb_trees(model)
    if not trees:
        raise RuntimeError("xgboost regressor has no JSON trees")
    params = model.get_xgb_params() if hasattr(model, "get_xgb_params") else {}
    eta = float(params.get("learning_rate") or 0.06)
    reg_lambda = float(params.get("reg_lambda") or 1.0)
    margin_full = np.full(len(y_full), float(np.mean(y_full)), dtype=np.float64)
    margin_test = np.full(X_test.shape[0], float(np.mean(y_full)), dtype=np.float64)
    l1s: list[float] = []
    for tree in trees:
        full_ids = PHASE2.route_xgb_leaf_ids(tree, X_full)
        test_ids = PHASE2.route_xgb_leaf_ids(tree, X_test)
        sub_ids = PHASE2.route_xgb_leaf_ids(tree, X_sub)
        l1, _ = PHASE2.leaf_population_distance(full_ids, sub_ids, None)
        l1s.append(l1)
        g = margin_full - y_full
        h = np.ones_like(g, dtype=np.float64)
        values: dict[int, float] = {}
        for leaf in np.unique(full_ids):
            mask = full_ids == leaf
            values[int(leaf)] = float(-g[mask].sum() / (h[mask].sum() + reg_lambda + 1e-8) * eta)
        margin_full += np.asarray([values.get(int(v), 0.0) for v in full_ids], dtype=np.float64)
        margin_test += np.asarray([values.get(int(v), 0.0) for v in test_ids], dtype=np.float64)
    return {"pred_test": margin_test, "leaf_population_l1": float(np.mean(l1s)) if l1s else np.nan}


def refit_lgbm_regression_leaves(model: Any, X_full: np.ndarray, y_full: np.ndarray, X_test: np.ndarray, X_sub: np.ndarray) -> dict[str, Any]:
    if not hasattr(model, "booster_"):
        raise RuntimeError("not a fitted lightgbm regressor")
    info = model.booster_.dump_model()
    trees = [t["tree_structure"] for t in info.get("tree_info", [])]
    if not trees:
        raise RuntimeError("lightgbm regressor has no dumped trees")
    params = model.get_params()
    eta = float(params.get("learning_rate", 0.05))
    reg_lambda = float(params.get("reg_lambda", params.get("lambda_l2", 0.0)) or 0.0)
    margin_full = np.full(len(y_full), float(np.mean(y_full)), dtype=np.float64)
    margin_test = np.full(X_test.shape[0], float(np.mean(y_full)), dtype=np.float64)
    l1s: list[float] = []
    for tree in trees:
        full_ids = PHASE2.route_lgb_leaf_ids(tree, X_full)
        test_ids = PHASE2.route_lgb_leaf_ids(tree, X_test)
        sub_ids = PHASE2.route_lgb_leaf_ids(tree, X_sub)
        l1, _ = PHASE2.leaf_population_distance(full_ids, sub_ids, None)
        l1s.append(l1)
        g = margin_full - y_full
        h = np.ones_like(g, dtype=np.float64)
        values: dict[int, float] = {}
        for leaf in np.unique(full_ids):
            mask = full_ids == leaf
            values[int(leaf)] = float(-g[mask].sum() / (h[mask].sum() + reg_lambda + 1e-8) * eta)
        margin_full += np.asarray([values.get(int(v), 0.0) for v in full_ids], dtype=np.float64)
        margin_test += np.asarray([values.get(int(v), 0.0) for v in test_ids], dtype=np.float64)
    return {"pred_test": margin_test, "leaf_population_l1": float(np.mean(l1s)) if l1s else np.nan}


def run_e21_regression_cell(bundle: Any, method: str, budget: int, seed: int, learner: str, worker_threads: int) -> dict[str, Any]:
    set_thread_limits(worker_threads)
    try:
        idx = BREADTH.select_indices(bundle, method, budget, seed, worker_threads)
        X_sub = bundle.X_train[idx]
        y_sub = bundle.y_train[idx]
        model = BREADTH.fit_regression_downstream(learner, X_sub, y_sub, seed, worker_threads)
        deployed_pred = np.asarray(model.predict(bundle.X_test), dtype=np.float64)
        deployed_r2 = float(r2_score(bundle.y_test, deployed_pred))
        if learner == "xgboost":
            refit = refit_xgb_regression_leaves(model, bundle.X_train, bundle.y_train, bundle.X_test, X_sub)
        elif learner == "lightgbm":
            refit = refit_lgbm_regression_leaves(model, bundle.X_train, bundle.y_train, bundle.X_test, X_sub)
        else:
            return {}
        leaf_refit_r2 = float(r2_score(bundle.y_test, refit["pred_test"]))
        return {
            "dataset": bundle.name,
            "method": method,
            "method_label": LABELS.get(method, method),
            "budget": f"budget_{budget}",
            "seed": f"seed_{seed}",
            "learner": learner,
            "deployed_r2": deployed_r2,
            "leaf_refit_r2": leaf_refit_r2,
            "leaf_estimate_recovery": float(leaf_refit_r2 - deployed_r2),
            "leaf_population_l1": float(refit["leaf_population_l1"]),
            "train_rows": int(len(idx)),
            "status": "ok",
        }
    except Exception as exc:
        return {
            "dataset": getattr(bundle, "name", "unknown"),
            "method": method,
            "method_label": LABELS.get(method, method),
            "budget": f"budget_{budget}",
            "seed": f"seed_{seed}",
            "learner": learner,
            "status": "error",
            "error": str(exc)[:300],
        }


def run_e18_downstream(cfg: dict[str, Any], args: argparse.Namespace, out_tables: Path, out_figs: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    datasets = [x.strip() for x in args.e18_datasets.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    doses = [float(x.strip()) for x in args.e18_doses.split(",") if x.strip()]
    repeats = list(range(int(args.e18_repeats)))
    tasks = [(d, s, dose, r) for d in datasets for s in seeds for dose in doses for r in repeats]
    rows = Parallel(n_jobs=min(args.n_jobs_int, len(tasks) or 1))(
        delayed(run_e18_downstream_cell)(cfg, d, s, dose, r, int(args.e18_estimators), int(args.e18_max_candidates))
        for d, s, dose, r in tasks
    )
    cells = pd.DataFrame(rows)
    if not cells.empty:
        ref = cells[cells["dose_ratio"].eq(0.0)][["dataset", "seed", "repeat", "auroc"]].rename(columns={"auroc": "dose0_auroc"})
        cells = cells.merge(ref, on=["dataset", "seed", "repeat"], how="left")
        cells["delta_auc_vs_dose0"] = cells["auroc"] - cells["dose0_auroc"]
        cells.to_csv(out_tables / "phase2_e18_downstream_dose_response_cells.csv", index=False)
    summary = pd.DataFrame()
    if not cells.empty:
        summary = (
            cells.groupby("dose_ratio")
            .agg(
                auroc=("auroc", "mean"),
                delta_auc_vs_dose0=("delta_auc_vs_dose0", "mean"),
                agreement=("agreement", "mean"),
                root_agreement=("root_agreement", "mean"),
                split_regret_norm=("split_regret_norm", "mean"),
                n_cells=("auroc", "count"),
            )
            .reset_index()
            .sort_values("dose_ratio")
        )
        summary.to_csv(out_tables / "phase2_e18_downstream_dose_response_summary.csv", index=False)
        fig, ax1 = plt.subplots(figsize=(7.2, 4.5))
        ax1.plot(summary["dose_ratio"], summary["auroc"], marker="o", label="AUROC")
        ax1.axvline(0.5, color="black", alpha=0.35, linewidth=1)
        ax1.set_xlabel("Injected SLD / node margin")
        ax1.set_ylabel("Custom full-leaf GBDT AUROC")
        ax1.grid(alpha=0.2)
        ax2 = ax1.twinx()
        ax2.plot(summary["dose_ratio"], summary["agreement"], marker="s", color="#b35c32", label="split agreement")
        ax2.set_ylabel("Split agreement")
        ax2.set_ylim(-0.02, 1.02)
        fig.tight_layout()
        fig.savefig(out_figs / "phase2_e18_downstream_dose_response.png", dpi=220)
        plt.close(fig)
    return cells, summary


def run_e21_regression_decomposition(cfg: dict[str, Any], args: argparse.Namespace, out_tables: Path, out_figs: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    datasets = [x.strip() for x in args.regression_datasets.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    budgets = [int(x.strip()) for x in args.e21_budgets.split(",") if x.strip()]
    methods = [x.strip() for x in args.e21_methods.split(",") if x.strip()]
    learners = [x.strip() for x in args.e21_learners.split(",") if x.strip()]
    all_rows: list[dict[str, Any]] = []
    for seed in seeds:
        bundles = BREADTH.load_bundles("regression", cfg, seed, int(args.max_rows), datasets)
        tasks = [(bundle, method, budget, seed, learner) for bundle in bundles for method in methods for budget in budgets for learner in learners]
        worker_threads = max(1, args.n_jobs_int // max(1, min(args.n_jobs_int, len(tasks))))
        rows = Parallel(n_jobs=min(args.n_jobs_int, len(tasks) or 1), prefer="threads")(
            delayed(run_e21_regression_cell)(bundle, method, budget, seed, learner, worker_threads)
            for bundle, method, budget, seed, learner in tasks
        )
        all_rows.extend([r for r in rows if r])
    cells = pd.DataFrame(all_rows)
    if cells.empty:
        return cells, pd.DataFrame(), pd.DataFrame()
    ok = cells[cells["status"].eq("ok")].copy()
    ref = ok[ok["method"].eq("full_data")][["dataset", "budget", "seed", "learner", "deployed_r2", "leaf_refit_r2"]].rename(
        columns={"deployed_r2": "full_deployed_r2", "leaf_refit_r2": "structure_ref_r2"}
    )
    cells = cells.drop(columns=["full_deployed_r2", "structure_ref_r2"], errors="ignore").merge(ref, on=["dataset", "budget", "seed", "learner"], how="left")
    mask = cells["status"].eq("ok") & cells["structure_ref_r2"].notna()
    cells.loc[mask, "structure_error"] = cells.loc[mask, "structure_ref_r2"] - cells.loc[mask, "leaf_refit_r2"]
    cells.loc[mask, "total_gap"] = cells.loc[mask, "structure_ref_r2"] - cells.loc[mask, "deployed_r2"]
    cells.to_csv(out_tables / "phase2_e21_regression_decomposition_cells.csv", index=False)
    summary = (
        cells[cells["status"].eq("ok") & ~cells["method"].eq("full_data")]
        .groupby(["method", "method_label"])
        .agg(
            deployed_r2=("deployed_r2", "mean"),
            leaf_refit_r2=("leaf_refit_r2", "mean"),
            structure_ref_r2=("structure_ref_r2", "mean"),
            structure_error=("structure_error", "mean"),
            leaf_estimate_recovery=("leaf_estimate_recovery", "mean"),
            leaf_population_l1=("leaf_population_l1", "mean"),
            n_cells=("deployed_r2", "count"),
        )
        .reset_index()
        .sort_values("deployed_r2", ascending=False)
    )
    summary.to_csv(out_tables / "phase2_e21_regression_decomposition_summary.csv", index=False)
    corr = pd.DataFrame(
        [
            {
                "x": "leaf_population_l1",
                "y": "leaf_estimate_recovery",
                "pooled_spearman": safe_spearman(cells[cells["status"].eq("ok")], "leaf_population_l1", "leaf_estimate_recovery"),
                "n_cells": int(cells[["leaf_population_l1", "leaf_estimate_recovery"]].dropna().shape[0]),
            },
            {
                "x": "leaf_population_l1",
                "y": "structure_error",
                "pooled_spearman": safe_spearman(cells[cells["status"].eq("ok")], "leaf_population_l1", "structure_error"),
                "n_cells": int(cells[["leaf_population_l1", "structure_error"]].dropna().shape[0]),
            },
        ]
    )
    corr.to_csv(out_tables / "phase2_e21_regression_decomposition_correlations.csv", index=False)
    if not summary.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
        plot = cells[cells["status"].eq("ok") & ~cells["method"].eq("full_data")].copy()
        for method, sub in plot.groupby("method"):
            axes[0].scatter(sub["leaf_population_l1"], sub["leaf_estimate_recovery"], s=18, alpha=0.45, label=LABELS.get(method, method))
            axes[1].scatter(sub["deployed_r2"], sub["leaf_refit_r2"], s=18, alpha=0.45, label=LABELS.get(method, method))
        axes[0].axhline(0, color="black", alpha=0.45, linewidth=1)
        axes[0].set_xlabel("Leaf-population mismatch")
        axes[0].set_ylabel("Leaf-estimate gap")
        axes[0].set_title("Regression leaf-estimate gap")
        axes[1].axline((0, 0), slope=1, color="black", alpha=0.35, linewidth=1)
        axes[1].set_xlabel("Deployed R2")
        axes[1].set_ylabel("Leaf-refit R2")
        axes[1].set_title("Regression refit effect")
        axes[0].grid(alpha=0.2)
        axes[1].grid(alpha=0.2)
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_figs / "phase2_e21_regression_decomposition.png", dpi=220)
        plt.close(fig)
    return cells, summary, corr


def write_note(
    out_notes: Path,
    e18_summary: pd.DataFrame,
    e21_summary: pd.DataFrame,
    e21_corr: pd.DataFrame,
) -> None:
    lines = [
        "# Phase 2 Full E18/E21 Addendum",
        "",
        "This addendum completes the two Phase 2 items that were previously proxy-only: E18 now includes downstream AUROC under controlled split-noise dosing, and E21 now includes a regression structure-vs-leaf decomposition.",
        "",
        "## E18 Downstream Dose-Response",
        "",
        md_table(e18_summary if not e18_summary.empty else pd.DataFrame()),
        "",
        "## E21 Regression Decomposition",
        "",
        md_table(e21_summary if not e21_summary.empty else pd.DataFrame()),
        "",
        "### E21 Correlations",
        "",
        md_table(e21_corr if not e21_corr.empty else pd.DataFrame()),
        "",
        "## Files",
        "",
        "- `tables/phase2_e18_downstream_dose_response_cells.csv`",
        "- `tables/phase2_e18_downstream_dose_response_summary.csv`",
        "- `tables/phase2_e21_regression_decomposition_cells.csv`",
        "- `tables/phase2_e21_regression_decomposition_summary.csv`",
        "- `tables/phase2_e21_regression_decomposition_correlations.csv`",
        "- `figures/phase2_e18_downstream_dose_response.png`",
        "- `figures/phase2_e21_regression_decomposition.png`",
        "",
    ]
    (out_notes / "PHASE2_E18_E21_FULL.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/focused.yaml")
    ap.add_argument("--paper-materials", default="paper_materials")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-jobs", default="6")
    ap.add_argument("--max-rows", type=int, default=12000)
    ap.add_argument("--skip-e18", action="store_true")
    ap.add_argument("--skip-e21", action="store_true")
    ap.add_argument("--e18-datasets", default="australian,breast_w,credit_g,diabetes,pc1,spambase")
    ap.add_argument("--e18-doses", default="0,0.25,0.49,0.5,0.75,1.0,1.5,2.0")
    ap.add_argument("--e18-repeats", type=int, default=3)
    ap.add_argument("--e18-estimators", type=int, default=80)
    ap.add_argument("--e18-max-candidates", type=int, default=192)
    ap.add_argument("--regression-datasets", default=",".join(BREADTH.REGRESSION_DATASETS))
    ap.add_argument("--e21-budgets", default="25,50")
    ap.add_argument("--e21-methods", default="histdistill,random,herding,k_center,full_data")
    ap.add_argument("--e21-learners", default="xgboost,lightgbm")
    args = ap.parse_args()
    args.n_jobs_int = resolve_n_jobs(args.n_jobs)
    set_thread_limits(args.n_jobs_int)
    cfg = load_config(args.config)
    materials = ensure_dir(args.paper_materials)
    out_tables = ensure_dir(materials / "tables")
    out_figs = ensure_dir(materials / "figures")
    out_notes = ensure_dir(materials / "notes")

    e18_summary = pd.DataFrame()
    if not args.skip_e18:
        _, e18_summary = run_e18_downstream(cfg, args, out_tables, out_figs)
    else:
        cached = out_tables / "phase2_e18_downstream_dose_response_summary.csv"
        if cached.exists():
            e18_summary = pd.read_csv(cached)
    e21_summary = pd.DataFrame()
    e21_corr = pd.DataFrame()
    if not args.skip_e21:
        _, e21_summary, e21_corr = run_e21_regression_decomposition(cfg, args, out_tables, out_figs)
    else:
        cached_summary = out_tables / "phase2_e21_regression_decomposition_summary.csv"
        cached_corr = out_tables / "phase2_e21_regression_decomposition_correlations.csv"
        if cached_summary.exists():
            e21_summary = pd.read_csv(cached_summary)
        if cached_corr.exists():
            e21_corr = pd.read_csv(cached_corr)
    write_note(out_notes, e18_summary, e21_summary, e21_corr)
    print("Wrote full E18/E21 Phase 2 addendum.")


if __name__ == "__main__":
    main()
