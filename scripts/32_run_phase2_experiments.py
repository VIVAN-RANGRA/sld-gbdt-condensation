from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from gaindistill.data import load_processed
from gaindistill.landscape import extract_xgb_trees, load_landscape, load_teacher, split_regret_metrics
from gaindistill.models import fit_downstream, predict_proba_positive
from gaindistill.utils import ensure_dir, list_dataset_dirs, load_config, resolve_n_jobs, set_thread_limits


LABELS = {
    "histdistill_refined": "HistDistill-Refined",
    "histdistill_density": "HistDistill-Density",
    "histdistill_greedy": "HistDistill-Greedy",
    "histdistill_importance": "ImportanceSample",
    "gain_path_refined": "Legacy gain-path refined",
    "herding": "Herding",
    "random": "Random",
    "goss_coreset": "GOSS coreset",
    "mvs_coreset": "MVS coreset",
    "craig_coreset": "CRAIG-style",
    "gradmatch_coreset": "GradMatch-style",
    "grand_coreset": "GraNd",
    "el2n_coreset": "EL2N",
    "distribution_matching": "DistributionMatching",
    "k_center": "K-center",
    "gradient_sampling": "Gradient sampling",
    "gain_herding": "Gain sketch",
    "gain_path_herding": "Gain-path sketch",
}


def label(method: str) -> str:
    return LABELS.get(method, method.replace("_", " "))


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


def parse_set(text: str, prefix: str | None = None) -> list[str]:
    out = []
    for raw in text.split(","):
        x = raw.strip()
        if not x:
            continue
        if prefix and not x.startswith(prefix):
            x = f"{prefix}{x}"
        out.append(x)
    return out


def logit_mean(y: np.ndarray, weight: np.ndarray | None = None) -> float:
    yy = np.asarray(y, dtype=np.float64)
    if weight is None:
        p = float(np.mean(yy))
    else:
        w = np.asarray(weight, dtype=np.float64)
        p = float(np.sum(w * yy) / max(1e-12, float(w.sum())))
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return math.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def child_by_id(children: list[dict[str, Any]], node_id: int) -> dict[str, Any]:
    for child in children:
        if int(child.get("nodeid", -1)) == int(node_id):
            return child
    return children[0]


def route_xgb_leaf_ids(tree: dict[str, Any], X: np.ndarray) -> np.ndarray:
    out = np.zeros(X.shape[0], dtype=np.int64)
    for i, row in enumerate(X):
        node = tree
        while "leaf" not in node:
            feat = int(str(node["split"]).lstrip("f"))
            thr = float(node["split_condition"])
            yes = int(node.get("yes", node.get("children", [{}])[0].get("nodeid", 0)))
            no = int(node.get("no", node.get("children", [{}, {}])[-1].get("nodeid", yes)))
            node = child_by_id(node.get("children", []), yes if row[feat] <= thr else no)
        out[i] = int(node.get("nodeid", 0))
    return out


def route_lgb_leaf_ids(node: dict[str, Any], X: np.ndarray) -> np.ndarray:
    out = np.zeros(X.shape[0], dtype=np.int64)
    for i, row in enumerate(X):
        cur = node
        while "leaf_value" not in cur:
            feat = int(cur["split_feature"])
            thr = float(cur["threshold"])
            cur = cur["left_child"] if row[feat] <= thr else cur["right_child"]
        out[i] = int(cur.get("leaf_index", 0))
    return out


def leaf_distribution(ids: np.ndarray, weight: np.ndarray | None = None) -> dict[int, float]:
    ids = np.asarray(ids, dtype=np.int64)
    w = np.ones(len(ids), dtype=np.float64) if weight is None else np.asarray(weight, dtype=np.float64)
    total = max(1e-12, float(w.sum()))
    out: dict[int, float] = {}
    for leaf in np.unique(ids):
        out[int(leaf)] = float(w[ids == leaf].sum() / total)
    return out


def leaf_population_distance(full_ids: np.ndarray, sub_ids: np.ndarray, sub_weight: np.ndarray | None = None) -> tuple[float, float]:
    pf = leaf_distribution(full_ids)
    ps = leaf_distribution(sub_ids, sub_weight)
    keys = sorted(set(pf) | set(ps))
    f = np.asarray([pf.get(k, 0.0) for k in keys], dtype=np.float64)
    s = np.asarray([ps.get(k, 0.0) for k in keys], dtype=np.float64)
    l1 = 0.5 * float(np.sum(np.abs(f - s)))
    eps = 1e-12
    kl = float(np.sum(f * np.log((f + eps) / (s + eps))))
    return l1, kl


def refit_xgb_leaves(
    model: Any,
    X_full: np.ndarray,
    y_full: np.ndarray,
    X_test: np.ndarray,
    X_sub: np.ndarray,
    sub_weight: np.ndarray | None,
    cfg: dict[str, Any],
) -> dict[str, float]:
    trees = extract_xgb_trees(model)
    if not trees:
        raise RuntimeError("xgboost model has no JSON trees")
    params = model.get_xgb_params() if hasattr(model, "get_xgb_params") else {}
    eta = float(params.get("learning_rate") or cfg["teacher"]["learning_rate"])
    reg_lambda = float(params.get("reg_lambda") or cfg["teacher"]["reg_lambda"])
    margin_full = np.full(len(y_full), logit_mean(y_full), dtype=np.float64)
    margin_test = np.full(X_test.shape[0], logit_mean(y_full), dtype=np.float64)
    l1s: list[float] = []
    kls: list[float] = []
    for tree in trees:
        full_ids = route_xgb_leaf_ids(tree, X_full)
        test_ids = route_xgb_leaf_ids(tree, X_test)
        sub_ids = route_xgb_leaf_ids(tree, X_sub)
        l1, kl = leaf_population_distance(full_ids, sub_ids, sub_weight)
        l1s.append(l1)
        kls.append(kl)
        p = np.clip(sigmoid(margin_full), 1e-6, 1.0 - 1e-6)
        g = p - y_full
        h = np.maximum(p * (1.0 - p), 1e-6)
        values: dict[int, float] = {}
        for leaf in np.unique(full_ids):
            mask = full_ids == leaf
            values[int(leaf)] = float(-g[mask].sum() / (h[mask].sum() + reg_lambda + 1e-8) * eta)
        margin_full += np.asarray([values.get(int(v), 0.0) for v in full_ids], dtype=np.float64)
        margin_test += np.asarray([values.get(int(v), 0.0) for v in test_ids], dtype=np.float64)
    p_test = np.clip(sigmoid(margin_test), 1e-7, 1.0 - 1e-7)
    return {
        "leaf_population_l1": float(np.mean(l1s)) if l1s else np.nan,
        "leaf_population_kl": float(np.mean(kls)) if kls else np.nan,
        "p_test_mean": float(np.mean(p_test)),
        "p_test_std": float(np.std(p_test)),
        "_p_test": p_test,
    }


def refit_lgbm_leaves(
    model: Any,
    X_full: np.ndarray,
    y_full: np.ndarray,
    X_test: np.ndarray,
    X_sub: np.ndarray,
    sub_weight: np.ndarray | None,
) -> dict[str, Any]:
    if not hasattr(model, "booster_"):
        raise RuntimeError("not a fitted lightgbm sklearn model")
    info = model.booster_.dump_model()
    trees = [t["tree_structure"] for t in info.get("tree_info", [])]
    if not trees:
        raise RuntimeError("lightgbm model has no dumped trees")
    params = model.get_params()
    eta = float(params.get("learning_rate", 0.06))
    reg_lambda = float(params.get("reg_lambda", params.get("lambda_l2", 0.0)) or 0.0)
    margin_full = np.full(len(y_full), logit_mean(y_full), dtype=np.float64)
    margin_test = np.full(X_test.shape[0], logit_mean(y_full), dtype=np.float64)
    l1s: list[float] = []
    kls: list[float] = []
    for tree in trees:
        full_ids = route_lgb_leaf_ids(tree, X_full)
        test_ids = route_lgb_leaf_ids(tree, X_test)
        sub_ids = route_lgb_leaf_ids(tree, X_sub)
        l1, kl = leaf_population_distance(full_ids, sub_ids, sub_weight)
        l1s.append(l1)
        kls.append(kl)
        p = np.clip(sigmoid(margin_full), 1e-6, 1.0 - 1e-6)
        g = p - y_full
        h = np.maximum(p * (1.0 - p), 1e-6)
        values: dict[int, float] = {}
        for leaf in np.unique(full_ids):
            mask = full_ids == leaf
            values[int(leaf)] = float(-g[mask].sum() / (h[mask].sum() + reg_lambda + 1e-8) * eta)
        margin_full += np.asarray([values.get(int(v), 0.0) for v in full_ids], dtype=np.float64)
        margin_test += np.asarray([values.get(int(v), 0.0) for v in test_ids], dtype=np.float64)
    p_test = np.clip(sigmoid(margin_test), 1e-7, 1.0 - 1e-7)
    return {
        "leaf_population_l1": float(np.mean(l1s)) if l1s else np.nan,
        "leaf_population_kl": float(np.mean(kls)) if kls else np.nan,
        "p_test_mean": float(np.mean(p_test)),
        "p_test_std": float(np.std(p_test)),
        "_p_test": p_test,
    }


def phase2_methods_available(results: Path, methods: list[str], budgets: list[str], seeds: list[str]) -> list[str]:
    available = []
    for method in methods:
        if any((results / "distilled" / ds / method / b / s / "distilled.npz").exists() for ds in [] for b in budgets for s in seeds):
            available.append(method)
    return available


def run_e16_cell(
    cfg: dict[str, Any],
    dataset: str,
    method: str,
    budget: str,
    seed: str,
    learner: str,
    full_lookup: dict[tuple[str, str, str, str], float],
    worker_threads: int,
) -> dict[str, Any] | None:
    set_thread_limits(worker_threads)
    results = Path(cfg["project"]["results_dir"])
    dist_path = results / "distilled" / dataset / method / budget / seed / "distilled.npz"
    if not dist_path.exists():
        return None
    bundle = load_processed(Path(cfg["project"]["processed_dir"]) / dataset)
    arr = np.load(dist_path)
    X_sub, y_sub = arr["X"], arr["y"]
    weights = arr["weights"] if "weights" in arr.files else None
    seed_n = int(seed.replace("seed_", ""))
    try:
        fit = fit_downstream(learner, X_sub, y_sub, cfg, worker_threads, seed_n, sample_weight=weights)
        deployed_p = np.clip(predict_proba_positive(fit.model, bundle.X_test), 1e-7, 1.0 - 1e-7)
        deployed_auc = float(roc_auc_score(bundle.y_test, deployed_p))
        if learner == "xgboost":
            refit = refit_xgb_leaves(fit.model, bundle.X_train, bundle.y_train, bundle.X_test, X_sub, weights, cfg)
        elif learner == "lightgbm":
            refit = refit_lgbm_leaves(fit.model, bundle.X_train, bundle.y_train, bundle.X_test, X_sub, weights)
        else:
            return None
        leaf_refit_auc = float(roc_auc_score(bundle.y_test, refit["_p_test"]))
        full_auc = full_lookup.get((dataset, budget, seed, learner))
        if full_auc is None or pd.isna(full_auc):
            full_fit = fit_downstream(learner, bundle.X_train, bundle.y_train, cfg, worker_threads, seed_n)
            full_auc = float(roc_auc_score(bundle.y_test, predict_proba_positive(full_fit.model, bundle.X_test)))
        structure_error = float(full_auc - leaf_refit_auc)
        leaf_estimate_error = float(leaf_refit_auc - deployed_auc)
        total_gap = float(full_auc - deployed_auc)
        return {
            "dataset": dataset,
            "method": method,
            "method_label": label(method),
            "budget": budget,
            "seed": seed,
            "learner": learner,
            "full_auc": float(full_auc),
            "deployed_auc": deployed_auc,
            "leaf_refit_auc": leaf_refit_auc,
            "total_gap": total_gap,
            "structure_error": structure_error,
            "leaf_estimate_recovery": leaf_estimate_error,
            "leaf_population_l1": float(refit["leaf_population_l1"]),
            "leaf_population_kl": float(refit["leaf_population_kl"]),
            "train_rows": int(len(y_sub)),
            "status": "ok",
        }
    except Exception as exc:
        return {
            "dataset": dataset,
            "method": method,
            "method_label": label(method),
            "budget": budget,
            "seed": seed,
            "learner": learner,
            "status": "error",
            "error": str(exc)[:300],
        }


def run_e17_cell(cfg: dict[str, Any], dataset: str, method: str, budget: str, seed: str) -> dict[str, Any] | None:
    results = Path(cfg["project"]["results_dir"])
    dist_path = results / "distilled" / dataset / method / budget / seed / "distilled.npz"
    landscape_path = results / "landscapes" / dataset / seed
    teacher_path = results / "teachers" / dataset / seed / "teacher.joblib"
    if not dist_path.exists() or not (landscape_path / "landscape.npz").exists() or not teacher_path.exists():
        return None
    bundle = load_processed(Path(cfg["project"]["processed_dir"]) / dataset)
    arr = np.load(dist_path)
    weights = arr["weights"] if "weights" in arr.files else None
    metrics = split_regret_metrics(
        load_landscape(landscape_path),
        arr["X"],
        arr["y"],
        bundle.meta,
        load_teacher(teacher_path),
        float(cfg["teacher"]["reg_lambda"]),
        float(cfg["teacher"]["gamma"]),
        sample_weight=weights,
    )
    return {
        "dataset": dataset,
        "method": method,
        "method_label": label(method),
        "budget": budget,
        "seed": seed,
        **metrics,
    }


def run_e18(cfg: dict[str, Any], datasets: list[str], seeds: list[str], out_tables: Path, out_figs: Path) -> pd.DataFrame:
    ratios = np.asarray([0.0, 0.25, 0.49, 0.50, 0.75, 1.0, 1.5, 2.0], dtype=np.float64)
    rows = []
    rng = np.random.default_rng(2026)
    for dataset in datasets[: min(8, len(datasets))]:
        for seed in seeds:
            landscape_path = Path(cfg["project"]["results_dir"]) / "landscapes" / dataset / seed
            if not (landscape_path / "landscape.npz").exists():
                continue
            landscape = load_landscape(landscape_path)
            states = [("root", 0, np.asarray(landscape.gains, dtype=np.float64))]
            for probe in landscape.probes:
                states.append((f"probe_depth_{probe.depth}", int(probe.depth), np.asarray(probe.gains, dtype=np.float64)))
            for state, depth, gains in states:
                if len(gains) < 2:
                    continue
                order = np.argsort(-gains)
                margin = float(gains[order[0]] - gains[order[1]])
                if margin <= 0:
                    continue
                scale = max(1.0, float(np.max(np.abs(gains))))
                for ratio in ratios:
                    eps = float(ratio * margin)
                    agreements = []
                    regrets = []
                    for _ in range(128):
                        noise = rng.uniform(-eps, eps, size=len(gains))
                        other = gains + noise
                        chosen = int(np.argmax(other))
                        agreements.append(float(chosen == int(order[0])))
                        regrets.append(max(0.0, float(gains[order[0]] - gains[chosen])) / scale)
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "state": state,
                            "depth": depth,
                            "dose_ratio": float(ratio),
                            "epsilon": eps,
                            "margin": margin,
                            "agreement": float(np.mean(agreements)),
                            "split_regret_norm": float(np.mean(regrets)),
                        }
                    )
    cells = pd.DataFrame(rows)
    cells.to_csv(out_tables / "phase2_e18_dose_response_cells.csv", index=False)
    if not cells.empty:
        summary = cells.groupby("dose_ratio").agg(agreement=("agreement", "mean"), split_regret_norm=("split_regret_norm", "mean")).reset_index()
        fig, ax1 = plt.subplots(figsize=(7.0, 4.4))
        ax1.plot(summary["dose_ratio"], summary["agreement"], marker="o", label="split agreement")
        ax1.axvline(0.5, color="black", alpha=0.4, linewidth=1.0)
        ax1.set_xlabel("Injected SLD / margin")
        ax1.set_ylabel("Agreement")
        ax1.set_ylim(-0.02, 1.02)
        ax2 = ax1.twinx()
        ax2.plot(summary["dose_ratio"], summary["split_regret_norm"], marker="s", color="#b35c32", label="split regret")
        ax2.set_ylabel("Normalized split regret")
        fig.tight_layout()
        fig.savefig(out_figs / "phase2_e18_dose_response.png", dpi=220)
        plt.close(fig)
    return cells


def save_figures(e16: pd.DataFrame, e17_merged: pd.DataFrame, universal: pd.DataFrame, selection: pd.DataFrame, out_figs: Path) -> None:
    if not e16.empty and {"sld_inf_norm", "structure_error", "leaf_population_l1", "leaf_estimate_recovery"}.issubset(e16.columns):
        ok = e16[e16["status"].eq("ok")].copy()
        if not ok.empty:
            fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
            for method, sub in ok.groupby("method"):
                axes[0].scatter(sub["sld_inf_norm"], sub["structure_error"], s=20, alpha=0.5, label=label(method))
                axes[1].scatter(sub["leaf_population_l1"], sub["leaf_estimate_recovery"], s=20, alpha=0.5, label=label(method))
            axes[0].set_xscale("log")
            axes[0].axhline(0, color="black", alpha=0.4, linewidth=1)
            axes[0].set_xlabel("SLD infinity")
            axes[0].set_ylabel("Full AUROC - leaf-refit AUROC")
            axes[0].set_title("Structure error vs SLD")
            axes[1].axhline(0, color="black", alpha=0.4, linewidth=1)
            axes[1].set_xlabel("Mean per-tree leaf mass TV")
            axes[1].set_ylabel("Leaf-refit AUROC - deployed AUROC")
            axes[1].set_title("Leaf-estimate recovery")
            axes[0].grid(alpha=0.2)
            axes[1].grid(alpha=0.2)
            axes[1].legend(fontsize=6, ncol=2)
            fig.tight_layout()
            fig.savefig(out_figs / "phase2_e16_error_decomposition.png", dpi=220)
            plt.close(fig)

    if not e17_merged.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        for method, sub in e17_merged.groupby("method"):
            ax.scatter(sub["split_regret_norm"], sub["auroc"], s=22, alpha=0.5, label=label(method))
        ax.set_xscale("log")
        ax.set_xlabel("Normalized split regret")
        ax.set_ylabel("Mean AUROC across learners")
        ax.set_title("Split-regret vs downstream AUROC")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(out_figs / "phase2_e17_split_regret_law.png", dpi=220)
        plt.close(fig)

    if not universal.empty:
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        ax.scatter(universal["split_regret_norm"], universal["auroc"], s=60)
        for _, row in universal.iterrows():
            ax.annotate(str(row["method_label"]), (row["split_regret_norm"], row["auroc"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("Mean normalized split regret")
        ax.set_ylabel("Mean AUROC")
        ax.set_title("Universal split-regret map")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_figs / "phase2_e19_universal_sld_map.png", dpi=220)
        plt.close(fig)

    if not selection.empty:
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        plot = selection.groupby("selector").agg(top1=("top1", "mean"), top3=("top3", "mean")).reset_index()
        x = np.arange(len(plot))
        ax.plot(x, plot["top1"], marker="o", label="top-1")
        ax.plot(x, plot["top3"], marker="s", label="top-3")
        ax.set_xticks(x)
        ax.set_xticklabels(plot["selector"], rotation=20, ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Agreement with AUROC ranking")
        ax.set_title("Training-free selection")
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_figs / "phase2_e20_training_free_selection.png", dpi=220)
        plt.close(fig)


def within_spearman(df: pd.DataFrame, x: str, y: str) -> float:
    vals = []
    for _, group in df.groupby(["dataset", "seed"]):
        if len(group) < 3 or group[x].nunique() < 2 or group[y].nunique() < 2:
            continue
        corr = spearmanr(group[x], group[y]).correlation
        if not pd.isna(corr):
            vals.append(float(corr))
    return float(np.mean(vals)) if vals else np.nan


def safe_spearman(df: pd.DataFrame, x: str, y: str) -> float:
    cols = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(cols) < 3 or cols[x].nunique() < 2 or cols[y].nunique() < 2:
        return np.nan
    corr = spearmanr(cols[x], cols[y]).correlation
    return float(corr) if not pd.isna(corr) else np.nan


def tree_only_scope_table(path: Path, metric_name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "learner" not in df.columns or metric_name not in df.columns:
        return pd.DataFrame()
    tree = df[df["learner"].isin(["xgboost", "lightgbm"])].copy()
    if tree.empty:
        return pd.DataFrame()
    return (
        tree.groupby(["method", "method_label"])
        .agg(metric_mean=(metric_name, "mean"), metric_std=(metric_name, "std"), n_cells=(metric_name, "count"))
        .reset_index()
        .sort_values("metric_mean", ascending=False)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/focused.yaml")
    ap.add_argument("--paper-tables", default="paper_tables")
    ap.add_argument("--paper-materials", default="paper_materials")
    ap.add_argument("--budgets", default="25,50")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument(
        "--methods",
        default="histdistill_greedy,histdistill_refined,histdistill_density,gain_path_refined,herding,random,goss_coreset,mvs_coreset,craig_coreset,gradmatch_coreset,grand_coreset,el2n_coreset,distribution_matching,k_center,gradient_sampling,gain_herding,gain_path_herding",
    )
    ap.add_argument("--e16-learners", default="xgboost,lightgbm")
    ap.add_argument("--n-jobs", default="6")
    ap.add_argument("--skip-e16", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_jobs = resolve_n_jobs(args.n_jobs)
    set_thread_limits(n_jobs)
    materials = ensure_dir(args.paper_materials)
    out_tables = ensure_dir(materials / "tables")
    out_figs = ensure_dir(materials / "figures")
    out_notes = ensure_dir(materials / "notes")
    budgets = parse_set(args.budgets, "budget_")
    seeds = parse_set(args.seeds, "seed_")
    methods = parse_set(args.methods)
    e16_learners = parse_set(args.e16_learners)
    datasets = [p.name for p in list_dataset_dirs(cfg["project"]["processed_dir"]) if p.name != "smoke_binary"]
    results = Path(cfg["project"]["results_dir"])
    downstream = pd.read_csv(Path(args.paper_tables) / "all_downstream_metrics.csv")
    fidelity = pd.read_csv(Path(args.paper_tables) / "all_split_fidelity_metrics.csv")
    full_lookup = {
        (r.dataset, r.budget, r.seed, r.learner): float(r.auroc)
        for r in downstream[(downstream["method"] == "full_data") & (downstream["dataset"] != "smoke_binary")].itertuples()
    }

    available_methods = []
    for method in methods:
        if any((results / "distilled" / ds / method / b / s / "distilled.npz").exists() for ds in datasets for b in budgets for s in seeds):
            available_methods.append(method)

    e16 = pd.DataFrame()
    if not args.skip_e16:
        e16_path = out_tables / "phase2_e16_error_decomposition_cells.csv"
        existing = pd.read_csv(e16_path) if e16_path.exists() else pd.DataFrame()
        existing_keys = set()
        if not existing.empty:
            existing_keys = {
                (str(r.dataset), str(r.method), str(r.budget), str(r.seed), str(r.learner))
                for r in existing.itertuples()
            }
        e16_methods = list(dict.fromkeys(["full_data", *available_methods]))
        tasks = [
            (dataset, method, budget, seed, learner)
            for dataset in datasets
            for method in e16_methods
            for budget in budgets
            for seed in seeds
            for learner in e16_learners
            if (results / "distilled" / dataset / method / budget / seed / "distilled.npz").exists()
            and (dataset, method, budget, seed, learner) not in existing_keys
        ]
        worker_threads = max(1, n_jobs // max(1, min(n_jobs, len(tasks))))
        rows = Parallel(n_jobs=min(n_jobs, len(tasks) or 1))(
            delayed(run_e16_cell)(cfg, dataset, method, budget, seed, learner, full_lookup, worker_threads)
            for dataset, method, budget, seed, learner in tasks
        )
        new_e16 = pd.DataFrame([r for r in rows if r is not None])
        e16 = pd.concat([existing, new_e16], ignore_index=True) if not existing.empty else new_e16
        if not e16.empty:
            fid_e16 = fidelity[["dataset", "method", "budget", "seed", "sld_inf_norm", "sld_root_agreement_direct"]].drop_duplicates()
            drop_cols = [c for c in ["sld_inf_norm", "sld_root_agreement_direct"] if c in e16.columns]
            e16 = e16.drop(columns=drop_cols).merge(fid_e16, on=["dataset", "method", "budget", "seed"], how="left")
            ok = e16[e16["status"].eq("ok")].copy()
            ref = ok[ok["method"].eq("full_data")][["dataset", "budget", "seed", "learner", "deployed_auc", "leaf_refit_auc"]].rename(
                columns={"deployed_auc": "full_deployed_auc", "leaf_refit_auc": "structure_ref_auc"}
            )
            if not ref.empty:
                e16 = e16.drop(columns=["full_deployed_auc", "structure_ref_auc"], errors="ignore")
                e16 = e16.merge(ref, on=["dataset", "budget", "seed", "learner"], how="left")
                ref_mask = e16["status"].eq("ok") & e16["structure_ref_auc"].notna()
                e16.loc[ref_mask, "structure_error"] = e16.loc[ref_mask, "structure_ref_auc"] - e16.loc[ref_mask, "leaf_refit_auc"]
                e16.loc[ref_mask, "total_gap"] = e16.loc[ref_mask, "structure_ref_auc"] - e16.loc[ref_mask, "deployed_auc"]
            e16.to_csv(out_tables / "phase2_e16_error_decomposition_cells.csv", index=False)

    regret_tasks = [
        (dataset, method, budget, seed)
        for dataset in datasets
        for method in available_methods
        for budget in budgets
        for seed in seeds
        if (results / "distilled" / dataset / method / budget / seed / "distilled.npz").exists()
    ]
    rows = Parallel(n_jobs=min(n_jobs, len(regret_tasks) or 1))(
        delayed(run_e17_cell)(cfg, dataset, method, budget, seed) for dataset, method, budget, seed in regret_tasks
    )
    e17 = pd.DataFrame([r for r in rows if r is not None])
    e17.to_csv(out_tables / "phase2_e17_split_regret_cells.csv", index=False)

    perf = (
        downstream[
            (downstream["dataset"].isin(datasets))
            & (downstream["method"].isin(available_methods))
            & (downstream["budget"].isin(budgets))
            & (downstream["seed"].isin(seeds))
        ]
        .groupby(["dataset", "method", "budget", "seed"])["auroc"]
        .mean()
        .reset_index()
    )
    e17_merged = perf.merge(e17, on=["dataset", "method", "budget", "seed"], how="inner")
    if not e17_merged.empty:
        e17_merged.to_csv(out_tables / "phase2_e17_split_regret_vs_auroc.csv", index=False)
    e18 = run_e18(cfg, datasets, seeds, out_tables, out_figs)
    e18_summary = pd.DataFrame()
    if not e18.empty:
        e18_summary = (
            e18.groupby("dose_ratio")
            .agg(
                agreement=("agreement", "mean"),
                split_regret_norm=("split_regret_norm", "mean"),
                n_states=("agreement", "count"),
            )
            .reset_index()
            .sort_values("dose_ratio")
        )
        e18_summary.to_csv(out_tables / "phase2_e18_dose_response_summary.csv", index=False)

    universal = pd.DataFrame()
    e19_residual_summary = pd.DataFrame()
    if not e17_merged.empty:
        universal = (
            e17_merged.groupby(["method", "method_label"])
            .agg(auroc=("auroc", "mean"), split_regret_norm=("split_regret_norm", "mean"), split_regret_agreement=("split_regret_agreement", "mean"), n_cells=("auroc", "count"))
            .reset_index()
            .sort_values(["split_regret_norm", "auroc"], ascending=[True, False])
        )
        if not e16.empty and "leaf_estimate_recovery" in e16.columns:
            leaf = e16[e16["status"].eq("ok")].groupby("method").agg(
                leaf_estimate_recovery=("leaf_estimate_recovery", "mean"),
                structure_error=("structure_error", "mean"),
                leaf_population_l1=("leaf_population_l1", "mean"),
            )
            universal = universal.join(leaf, on="method")
        if len(universal) >= 3 and universal["split_regret_norm"].nunique() > 1:
            fit_rows = universal[["split_regret_norm", "auroc"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(fit_rows) >= 3:
                slope, intercept = np.polyfit(fit_rows["split_regret_norm"], fit_rows["auroc"], 1)
                universal["line_pred_auc"] = intercept + slope * universal["split_regret_norm"]
                universal["line_residual_auc"] = universal["auroc"] - universal["line_pred_auc"]
                summary_rows = [
                    {
                        "quantity": "method_level_split_regret_to_auroc_slope",
                        "value": float(slope),
                        "n_methods": int(len(fit_rows)),
                    },
                    {
                        "quantity": "method_level_split_regret_to_auroc_spearman",
                        "value": safe_spearman(universal, "split_regret_norm", "auroc"),
                        "n_methods": int(len(fit_rows)),
                    },
                ]
                for col in ["leaf_estimate_recovery", "leaf_population_l1", "structure_error"]:
                    if col in universal.columns:
                        summary_rows.append(
                            {
                                "quantity": f"line_residual_vs_{col}_spearman",
                                "value": safe_spearman(universal, "line_residual_auc", col),
                                "n_methods": int(universal[["line_residual_auc", col]].dropna().shape[0]),
                            }
                        )
                e19_residual_summary = pd.DataFrame(summary_rows)
                e19_residual_summary.to_csv(out_tables / "phase2_e19_residual_summary.csv", index=False)
        universal.to_csv(out_tables / "phase2_e19_universal_sld_map.csv", index=False)

    selection_rows = []
    if not e17_merged.empty:
        for selector, col, ascending in [("split_regret", "split_regret_norm", True), ("sld", "sld_inf_norm", True)]:
            if col == "sld_inf_norm":
                base = e17_merged.merge(
                    fidelity[["dataset", "method", "budget", "seed", "sld_inf_norm"]].drop_duplicates(),
                    on=["dataset", "method", "budget", "seed"],
                    how="left",
                )
            else:
                base = e17_merged.copy()
            base = base.dropna(subset=[col, "auroc"])
            for key, group in base.groupby(["dataset", "budget", "seed"]):
                method_scores = group.groupby("method").agg(score=(col, "mean"), auroc=("auroc", "mean")).reset_index()
                if len(method_scores) < 3:
                    continue
                chosen = str(method_scores.sort_values("score", ascending=ascending).iloc[0]["method"])
                auroc_ranked = method_scores.sort_values("auroc", ascending=False)["method"].astype(str).tolist()
                corr = spearmanr(method_scores["score"], method_scores["auroc"]).correlation
                selection_rows.append(
                    {
                        "selector": selector,
                        "dataset": key[0],
                        "budget": key[1],
                        "seed": key[2],
                        "chosen_method": chosen,
                        "best_method": auroc_ranked[0],
                        "top1": float(chosen == auroc_ranked[0]),
                        "top3": float(chosen in set(auroc_ranked[:3])),
                        "rank_spearman": float(0.0 if pd.isna(corr) else corr),
                        "candidate_methods": int(len(method_scores)),
                    }
                )
    selection = pd.DataFrame(selection_rows)
    selection_summary = pd.DataFrame()
    if not selection.empty:
        selection.to_csv(out_tables / "phase2_e20_training_free_selection.csv", index=False)
        selection_summary = (
            selection.groupby("selector")
            .agg(
                top1=("top1", "mean"),
                top3=("top3", "mean"),
                rank_spearman=("rank_spearman", "mean"),
                cells=("top1", "count"),
            )
            .reset_index()
        )
        selection_summary.to_csv(out_tables / "phase2_e20_training_free_selection_summary.csv", index=False)

    save_figures(e16, e17_merged, universal, selection, out_figs)

    e16_summary = pd.DataFrame()
    e16_corr = pd.DataFrame()
    if not e16.empty:
        ok = e16[e16["status"].eq("ok")].copy()
        if not ok.empty:
            e16_for_summary = ok[~ok["method"].eq("full_data")].copy()
            e16_summary = (
                e16_for_summary.groupby(["method", "method_label"])
                .agg(
                    deployed_auc=("deployed_auc", "mean"),
                    leaf_refit_auc=("leaf_refit_auc", "mean"),
                    full_auc=("full_deployed_auc", "mean"),
                    structure_ref_auc=("structure_ref_auc", "mean"),
                    structure_error=("structure_error", "mean"),
                    leaf_estimate_recovery=("leaf_estimate_recovery", "mean"),
                    leaf_population_l1=("leaf_population_l1", "mean"),
                    sld_inf_norm=("sld_inf_norm", "mean"),
                    n_cells=("deployed_auc", "count"),
                )
                .reset_index()
                .sort_values("leaf_estimate_recovery", ascending=False)
            )
            e16_summary.to_csv(out_tables / "phase2_e16_error_decomposition_summary.csv", index=False)
            corr_source = e16_for_summary.copy()
            if not e17.empty:
                corr_source = corr_source.merge(
                    e17[["dataset", "method", "budget", "seed", "split_regret_norm", "split_regret_root_norm"]].drop_duplicates(),
                    on=["dataset", "method", "budget", "seed"],
                    how="left",
                )
            corr_rows = []
            for x, y in [
                ("sld_inf_norm", "structure_error"),
                ("split_regret_norm", "structure_error"),
                ("leaf_population_l1", "leaf_estimate_recovery"),
                ("leaf_population_kl", "leaf_estimate_recovery"),
                ("sld_inf_norm", "leaf_estimate_recovery"),
            ]:
                if x not in corr_source.columns or y not in corr_source.columns:
                    continue
                corr_rows.append(
                    {
                        "x": x,
                        "y": y,
                        "pooled_spearman": safe_spearman(corr_source, x, y),
                        "within_dataset_spearman": within_spearman(corr_source.dropna(subset=[x, y]), x, y),
                        "n_cells": int(corr_source[[x, y]].dropna().shape[0]),
                    }
                )
            e16_corr = pd.DataFrame(corr_rows)
            e16_corr.to_csv(out_tables / "phase2_e16_decomposition_correlations.csv", index=False)

    e17_summary = pd.DataFrame()
    if not e17_merged.empty:
        rows2 = []
        for metric in ["split_regret_norm", "split_regret_root_norm"]:
            pooled = spearmanr(e17_merged[metric], e17_merged["auroc"]).correlation if e17_merged[metric].nunique() > 1 else np.nan
            rows2.append(
                {
                    "metric": metric,
                    "pooled_spearman_vs_auroc": float(0.0 if pd.isna(pooled) else pooled),
                    "within_dataset_spearman_vs_auroc": within_spearman(e17_merged, metric, "auroc"),
                    "n_cells": int(len(e17_merged)),
                }
            )
        if "sld_inf_norm" in fidelity.columns:
            sld_merge = perf.merge(
                fidelity[["dataset", "method", "budget", "seed", "sld_inf_norm"]].drop_duplicates(),
                on=["dataset", "method", "budget", "seed"],
                how="inner",
            ).dropna()
            pooled = spearmanr(sld_merge["sld_inf_norm"], sld_merge["auroc"]).correlation if sld_merge["sld_inf_norm"].nunique() > 1 else np.nan
            rows2.append(
                {
                    "metric": "sld_inf_norm",
                    "pooled_spearman_vs_auroc": float(0.0 if pd.isna(pooled) else pooled),
                    "within_dataset_spearman_vs_auroc": within_spearman(sld_merge, "sld_inf_norm", "auroc"),
                    "n_cells": int(len(sld_merge)),
                }
            )
        e17_summary = pd.DataFrame(rows2)
        e17_summary.to_csv(out_tables / "phase2_e17_metric_law_summary.csv", index=False)

    generality_rows = []
    multiclass_perf = out_tables / "icdm_multiclass_ovr_performance.csv"
    if multiclass_perf.exists():
        df = pd.read_csv(multiclass_perf)
        full = float(df.loc[df["method"].eq("full_data"), "metric_mean"].mean())
        hist = float(df.loc[df["method"].eq("histdistill"), "metric_mean"].mean())
        random_mean = float(df.loc[df["method"].eq("random"), "metric_mean"].mean())
        tree_scope = tree_only_scope_table(out_tables / "icdm_multiclass_ovr_cells.csv", "primary_metric")
        if not tree_scope.empty:
            tree_scope.to_csv(out_tables / "phase2_e21_multiclass_tree_scope.csv", index=False)
        generality_rows.append(
            {
                "artifact": "multiclass_ovr",
                "rows": int(len(df)),
                "primary_metric": "macro_auroc",
                "histdistill": hist,
                "random": random_mean,
                "full_data": full,
                "note": "supportive: HistDistill beats random on mean macro-AUROC but remains below full-data",
            }
        )
    else:
        generality_rows.append({"artifact": "multiclass_ovr", "rows": 0, "primary_metric": "macro_auroc", "note": "missing"})

    regression_perf = out_tables / "icdm_regression_performance.csv"
    if regression_perf.exists():
        df = pd.read_csv(regression_perf)
        full = float(df.loc[df["method"].eq("full_data"), "metric_mean"].mean())
        hist = float(df.loc[df["method"].eq("histdistill"), "metric_mean"].mean())
        random_mean = float(df.loc[df["method"].eq("random"), "metric_mean"].mean())
        tree_scope = tree_only_scope_table(out_tables / "icdm_regression_cells.csv", "primary_metric")
        tree_hist = np.nan
        tree_random = np.nan
        if not tree_scope.empty:
            tree_scope.to_csv(out_tables / "phase2_e21_regression_tree_scope.csv", index=False)
            tree_hist = float(tree_scope.loc[tree_scope["method"].eq("histdistill"), "metric_mean"].mean())
            tree_random = float(tree_scope.loc[tree_scope["method"].eq("random"), "metric_mean"].mean())
        generality_rows.append(
            {
                "artifact": "regression",
                "rows": int(len(df)),
                "primary_metric": "r2",
                "histdistill": hist,
                "random": random_mean,
                "full_data": full,
                "tree_histdistill": tree_hist,
                "tree_random": tree_random,
                "note": "limitation: all-learner mean is dominated by severe MLP failures; tree-only slice is milder",
            }
        )
    else:
        generality_rows.append({"artifact": "regression", "rows": 0, "primary_metric": "r2", "note": "missing"})

    hpo_overall = out_tables / "icdm_hpo_overall.csv"
    if hpo_overall.exists():
        df = pd.read_csv(hpo_overall)
        row = df.iloc[0].to_dict() if not df.empty else {}
        generality_rows.append(
            {
                "artifact": "hpo",
                "rows": int(len(df)),
                "primary_metric": "top3_jaccard",
                "histdistill": float(row.get("mean_top3_jaccard", np.nan)),
                "random": np.nan,
                "full_data": np.nan,
                "note": "negative: condensed-set HPO rankings do not recover full-data top configurations",
            }
        )
    else:
        generality_rows.append({"artifact": "hpo", "rows": 0, "primary_metric": "top3_jaccard", "note": "missing"})
    e21 = pd.DataFrame(generality_rows)
    e21.to_csv(out_tables / "phase2_e21_generality_scope.csv", index=False)

    lines = [
        "# Phase 2 Results",
        "",
        "Phase 2 re-centers the paper on mechanism diagnostics: structure-vs-leaf error, continuous split regret, universal SLD mapping, and training-free selection.",
        "",
        "## E16 Structure vs Leaf-Estimate Error",
        "",
        md_table(
            e16_summary[["method_label", "deployed_auc", "leaf_refit_auc", "structure_ref_auc", "full_auc", "structure_error", "leaf_estimate_recovery", "leaf_population_l1", "sld_inf_norm", "n_cells"]]
            if not e16_summary.empty
            else pd.DataFrame()
        ),
        "",
        "### E16 Correlation Checks",
        "",
        md_table(e16_corr if not e16_corr.empty else pd.DataFrame()),
        "",
        "## E17 Split-Regret Law",
        "",
        md_table(e17_summary if not e17_summary.empty else pd.DataFrame()),
        "",
        "## E18 Controlled Dose-Response",
        "",
        "This is a landscape-only controlled perturbation test: it injects calibrated split-gain noise and measures agreement/regret. It does not claim downstream AUROC causality without a full perturbed-tree retraining operator.",
        "",
        md_table(e18_summary if not e18_summary.empty else pd.DataFrame()),
        "",
        "## E19 Universal Diagnostic Map",
        "",
        md_table(
            universal[["method_label", "auroc", "split_regret_norm", "split_regret_agreement", "line_residual_auc", "leaf_estimate_recovery", "structure_error", "n_cells"]]
            if not universal.empty and "leaf_estimate_recovery" in universal.columns
            else universal
        ),
        "",
        "### E19 Residual Checks",
        "",
        md_table(e19_residual_summary if not e19_residual_summary.empty else pd.DataFrame()),
        "",
        "## E20 Training-Free Selection",
        "",
        md_table(selection_summary if not selection_summary.empty else pd.DataFrame()),
        "",
        "## E21 Generality Scope",
        "",
        md_table(e21),
        "",
        "## Files",
        "",
        "- `tables/phase2_e16_error_decomposition_cells.csv`",
        "- `tables/phase2_e16_error_decomposition_summary.csv`",
        "- `tables/phase2_e16_decomposition_correlations.csv`",
        "- `tables/phase2_e17_split_regret_cells.csv`",
        "- `tables/phase2_e17_metric_law_summary.csv`",
        "- `tables/phase2_e18_dose_response_cells.csv`",
        "- `tables/phase2_e18_dose_response_summary.csv`",
        "- `tables/phase2_e19_universal_sld_map.csv`",
        "- `tables/phase2_e19_residual_summary.csv`",
        "- `tables/phase2_e20_training_free_selection.csv`",
        "- `tables/phase2_e20_training_free_selection_summary.csv`",
        "- `tables/phase2_e21_generality_scope.csv`",
        "- `tables/phase2_e21_multiclass_tree_scope.csv`",
        "- `tables/phase2_e21_regression_tree_scope.csv`",
        "- `figures/phase2_e16_error_decomposition.png`",
        "- `figures/phase2_e17_split_regret_law.png`",
        "- `figures/phase2_e18_dose_response.png`",
        "- `figures/phase2_e19_universal_sld_map.png`",
        "- `figures/phase2_e20_training_free_selection.png`",
        "",
    ]
    (out_notes / "PHASE2_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote Phase 2 results.")


if __name__ == "__main__":
    main()
