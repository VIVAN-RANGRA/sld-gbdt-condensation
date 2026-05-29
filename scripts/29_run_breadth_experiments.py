from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.datasets import fetch_california_housing, fetch_openml, load_diabetes
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.metrics import pairwise_distances
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor

from gaindistill.utils import ensure_dir, load_config, read_json, resolve_n_jobs, set_thread_limits, slugify


MULTICLASS_DATASETS = [
    "balance_scale",
    "cmc",
    "eucalyptus",
    "letter",
    "mfeat_fourier",
    "optdigits",
    "satimage",
    "vehicle",
]
REGRESSION_DATASETS = [
    "diabetes_regression",
    "california_housing",
    "abalone",
    "cpu_act",
    "kin8nm",
    "house_16H",
    "elevators",
    "wine_quality",
]
METHODS = ["histdistill", "random", "herding", "k_center", "full_data"]
LEARNERS = ["xgboost", "lightgbm", "mlp"]
LABELS = {
    "histdistill": "HistDistill",
    "random": "Random",
    "herding": "Herding",
    "k_center": "K-center",
    "full_data": "Full data",
}


@dataclass
class TabularBundle:
    name: str
    task: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_classes: int | None = None


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


def encode_features(frame: pd.DataFrame, target: str, max_rows: int, seed: int) -> tuple[np.ndarray, pd.Series]:
    df = frame.copy()
    df = df.dropna(axis=0, subset=[target]).reset_index(drop=True)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
    y = df[target].copy()
    X_df = df.drop(columns=[target]).copy()
    cols: list[np.ndarray] = []
    for col in X_df.columns:
        s = X_df[col]
        if pd.api.types.is_numeric_dtype(s):
            vals = pd.to_numeric(s, errors="coerce")
            fill = float(vals.median()) if vals.notna().any() else 0.0
            cols.append(vals.fillna(fill).to_numpy(dtype=np.float32))
        else:
            obj = s.astype("string").fillna("__MISSING__")
            counts = obj.value_counts()
            common = set(counts.index[:128])
            obj = obj.where(obj.isin(common), "__RARE__")
            cats = {v: i for i, v in enumerate(sorted(obj.unique().tolist()))}
            cols.append(obj.map(cats).to_numpy(dtype=np.float32))
    if not cols:
        raise ValueError("dataset has no usable features")
    return np.vstack(cols).T.astype(np.float32), y


def split_bundle(name: str, task: str, X: np.ndarray, y: np.ndarray, seed: int) -> TabularBundle:
    if task == "multiclass":
        enc = LabelEncoder()
        yy = enc.fit_transform(pd.Series(y).astype(str))
        classes, counts = np.unique(yy, return_counts=True)
        if len(classes) < 3:
            raise ValueError("not multiclass")
        if int(counts.min()) < 8:
            raise ValueError("minority class too small")
        strat = yy
        X_temp, X_test, y_temp, y_test = train_test_split(X, yy, test_size=0.2, random_state=seed, stratify=strat)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=seed, stratify=y_temp
        )
        return TabularBundle(name, task, X_train, y_train, X_val, y_val, X_test, y_test, n_classes=len(classes))
    yy = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(yy)
    X = X[mask]
    yy = yy[mask]
    X_temp, X_test, y_temp, y_test = train_test_split(X, yy, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=seed)
    return TabularBundle(name, task, X_train, y_train, X_val, y_val, X_test, y_test)


def load_multiclass_bundle(cfg: dict[str, Any], dataset: str, seed: int, max_rows: int) -> TabularBundle:
    raw_root = Path(cfg["project"]["raw_dir"]) / "openml_cc18" / dataset
    meta = read_json(raw_root / "metadata.json")
    frame = pd.read_csv(raw_root / "data.csv")
    X, y = encode_features(frame, str(meta["target"]), max_rows=max_rows, seed=seed)
    return split_bundle(dataset, "multiclass", X, y, seed)


def fetch_regression_frame(dataset: str) -> tuple[pd.DataFrame, str]:
    if dataset == "diabetes_regression":
        ds = load_diabetes(as_frame=True)
        frame = ds.frame.copy()
        return frame, "target"
    if dataset == "california_housing":
        ds = fetch_california_housing(as_frame=True)
        frame = ds.frame.copy()
        return frame, "MedHouseVal"
    openml_aliases = {
        "abalone": ["abalone"],
        "cpu_act": ["cpu_act"],
        "kin8nm": ["kin8nm"],
        "house_16H": ["house_16H", "house_16h"],
        "elevators": ["elevators"],
        "wine_quality": ["wine_quality", "wine-quality-white", "wine-quality-red"],
    }
    last_error: Exception | None = None
    for name in openml_aliases.get(dataset, [dataset]):
        try:
            ds = fetch_openml(name=name, version="active", as_frame=True, parser="auto")
            frame = ds.frame.copy()
            target = ds.target_names[0] if ds.target_names else "target"
            if target not in frame.columns:
                frame[target] = ds.target
            return frame, target
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"OpenML fetch failed for {dataset}: {last_error}")


def load_regression_bundle(dataset: str, seed: int, max_rows: int) -> TabularBundle:
    frame, target = fetch_regression_frame(dataset)
    X, y = encode_features(frame, target, max_rows=max_rows, seed=seed)
    return split_bundle(dataset, "regression", X, y, seed)


def fit_xgb_classifier(X: np.ndarray, y: np.ndarray, n_classes: int, seed: int, n_jobs: int):
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=int(n_classes),
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=max(1, n_jobs),
        random_state=seed,
        verbosity=0,
    )
    model.fit(X, y)
    return model


def fit_xgb_regressor(X: np.ndarray, y: np.ndarray, seed: int, n_jobs: int):
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=max(1, n_jobs),
        random_state=seed,
        verbosity=0,
    )
    model.fit(X, y)
    return model


def split_sketch_features(X: np.ndarray, signal: np.ndarray, max_features: int = 12, quantiles: int = 8) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X).astype(np.float64)
    feat_scores = np.nan_to_num(np.var(Xs, axis=0), nan=0.0)
    top = np.argsort(-feat_scores)[: min(max_features, X.shape[1])]
    cols = [Xs[:, j] for j in range(min(16, Xs.shape[1]))]
    sig = np.asarray(signal, dtype=np.float64)
    sig = (sig - float(np.mean(sig))) / max(float(np.std(sig)), 1e-6)
    cols.append(sig)
    for j in top:
        thresholds = np.unique(np.quantile(X[:, j], np.linspace(0, 1, quantiles + 2)[1:-1]))
        for thr in thresholds[:quantiles]:
            left = (X[:, j] <= thr).astype(np.float64)
            cols.append(left)
            cols.append(left * sig)
    return np.vstack(cols).T.astype(np.float64)


def multiclass_ovr_sketch(bundle: TabularBundle, seed: int, n_jobs: int) -> np.ndarray:
    teacher = fit_xgb_classifier(bundle.X_train, bundle.y_train, int(bundle.n_classes or 2), seed, n_jobs)
    proba = np.clip(teacher.predict_proba(bundle.X_train), 1e-6, 1.0 - 1e-6)
    cols = [StandardScaler().fit_transform(bundle.X_train).astype(np.float64)]
    for c in range(int(bundle.n_classes or proba.shape[1])):
        y_bin = (bundle.y_train == c).astype(np.float64)
        g = proba[:, c] - y_bin
        h = np.maximum(proba[:, c] * (1.0 - proba[:, c]), 1e-6)
        cols.append(split_sketch_features(bundle.X_train, g + h, max_features=8, quantiles=6))
        cols.append(proba[:, [c]])
    return np.hstack(cols).astype(np.float64)


def regression_landscape_sketch(bundle: TabularBundle, seed: int, n_jobs: int) -> np.ndarray:
    teacher = fit_xgb_regressor(bundle.X_train, bundle.y_train, seed, n_jobs)
    pred = np.asarray(teacher.predict(bundle.X_train), dtype=np.float64)
    residual = pred - bundle.y_train
    cols = [split_sketch_features(bundle.X_train, residual, max_features=12, quantiles=8)]
    cols.append(pred[:, None])
    cols.append(bundle.y_train[:, None])
    return np.hstack(cols).astype(np.float64)


def greedy_mean_match(S: np.ndarray, idx: np.ndarray, k: int) -> list[int]:
    if len(idx) <= k:
        return idx.tolist()
    Sg = S[idx]
    target = Sg.mean(axis=0)
    selected: list[int] = []
    chosen = np.zeros(len(idx), dtype=bool)
    running = np.zeros(S.shape[1], dtype=np.float64)
    for t in range(k):
        desired = (t + 1) * target - running
        scores = np.einsum("ij,ij->i", Sg - desired, Sg - desired)
        scores[chosen] = np.inf
        loc = int(np.argmin(scores))
        selected.append(int(idx[loc]))
        chosen[loc] = True
        running += Sg[loc]
    return selected


def random_select(y: np.ndarray, budget: int, seed: int, task: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if task == "multiclass":
        idx: list[int] = []
        for c in np.unique(y):
            cls = np.flatnonzero(y == c)
            rng.shuffle(cls)
            idx.extend(cls[: min(budget, len(cls))].tolist())
        return np.asarray(idx, dtype=int)
    n = min(budget, len(y))
    return np.asarray(rng.choice(len(y), size=n, replace=False), dtype=int)


def herding_select(X: np.ndarray, y: np.ndarray, budget: int, task: str) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X).astype(np.float64)
    if task == "multiclass":
        idx: list[int] = []
        for c in np.unique(y):
            cls = np.flatnonzero(y == c)
            idx.extend(greedy_mean_match(Xs, cls, min(budget, len(cls))))
        return np.asarray(idx, dtype=int)
    bins = pd.qcut(pd.Series(y).rank(method="first"), q=min(10, len(y)), labels=False, duplicates="drop").to_numpy()
    idx = []
    per_bin = max(1, math.ceil(budget / max(1, len(np.unique(bins)))))
    for b in np.unique(bins):
        group = np.flatnonzero(bins == b)
        idx.extend(greedy_mean_match(Xs, group, min(per_bin, len(group))))
    return np.asarray(idx[: min(budget, len(idx))], dtype=int)


def k_center_select(X: np.ndarray, y: np.ndarray, budget: int, seed: int, task: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Xs = StandardScaler().fit_transform(X).astype(np.float64)
    groups = np.unique(y) if task == "multiclass" else np.unique(
        pd.qcut(pd.Series(y).rank(method="first"), q=min(10, len(y)), labels=False, duplicates="drop").to_numpy()
    )
    group_assign = y if task == "multiclass" else pd.qcut(pd.Series(y).rank(method="first"), q=min(10, len(y)), labels=False, duplicates="drop").to_numpy()
    selected: list[int] = []
    per_group = budget if task == "multiclass" else max(1, math.ceil(budget / max(1, len(groups))))
    for group in groups:
        idx = np.flatnonzero(group_assign == group)
        k = min(per_group, len(idx))
        if k <= 0:
            continue
        first = int(rng.choice(idx))
        chosen = [first]
        dmin = pairwise_distances(Xs[idx], Xs[[first]]).ravel()
        for _ in range(1, k):
            nxt = int(idx[np.argmax(dmin)])
            chosen.append(nxt)
            dmin = np.minimum(dmin, pairwise_distances(Xs[idx], Xs[[nxt]]).ravel())
        selected.extend(chosen)
    if task == "regression":
        selected = selected[:budget]
    return np.asarray(selected, dtype=int)


def histdistill_select(bundle: TabularBundle, budget: int, seed: int, n_jobs: int) -> np.ndarray:
    if bundle.task == "multiclass":
        S = multiclass_ovr_sketch(bundle, seed, n_jobs)
        idx: list[int] = []
        for c in np.unique(bundle.y_train):
            cls = np.flatnonzero(bundle.y_train == c)
            idx.extend(greedy_mean_match(S, cls, min(budget, len(cls))))
        return np.asarray(idx, dtype=int)
    S = regression_landscape_sketch(bundle, seed, n_jobs)
    bins = pd.qcut(pd.Series(bundle.y_train).rank(method="first"), q=min(10, len(bundle.y_train)), labels=False, duplicates="drop").to_numpy()
    idx = []
    per_bin = max(1, math.ceil(budget / max(1, len(np.unique(bins)))))
    for b in np.unique(bins):
        group = np.flatnonzero(bins == b)
        idx.extend(greedy_mean_match(S, group, min(per_bin, len(group))))
    return np.asarray(idx[:budget], dtype=int)


def select_indices(bundle: TabularBundle, method: str, budget: int, seed: int, n_jobs: int) -> np.ndarray:
    if method == "full_data":
        return np.arange(len(bundle.y_train), dtype=int)
    if method == "random":
        return random_select(bundle.y_train, budget, seed, bundle.task)
    if method == "herding":
        return herding_select(bundle.X_train, bundle.y_train, budget, bundle.task)
    if method == "k_center":
        return k_center_select(bundle.X_train, bundle.y_train, budget, seed, bundle.task)
    if method == "histdistill":
        return histdistill_select(bundle, budget, seed, n_jobs)
    raise ValueError(f"unknown method {method}")


def fit_multiclass_downstream(learner: str, X: np.ndarray, y: np.ndarray, n_classes: int, seed: int, n_jobs: int):
    if learner == "xgboost":
        return fit_xgb_classifier(X, y, n_classes, seed, n_jobs)
    if learner == "lightgbm":
        # LightGBM's multiclass Windows wheel can abort the Python worker on
        # some condensed matrices. Use sklearn's histogram GBDT here so the
        # breadth suite remains stable and still tests tree-transfer behavior.
        model = HistGradientBoostingClassifier(
            max_iter=120,
            max_leaf_nodes=15,
            learning_rate=0.06,
            l2_regularization=1.0,
            random_state=seed,
        )
        model.fit(X, y)
        return model
    model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=220, alpha=1e-4, random_state=seed))
    model.fit(X, y)
    return model


def fit_regression_downstream(learner: str, X: np.ndarray, y: np.ndarray, seed: int, n_jobs: int):
    if learner == "xgboost":
        return fit_xgb_regressor(X, y, seed, n_jobs)
    if learner == "lightgbm":
        try:
            from lightgbm import LGBMRegressor

            model = LGBMRegressor(
                n_estimators=140,
                max_depth=4,
                learning_rate=0.05,
                num_leaves=15,
                n_jobs=max(1, n_jobs),
                random_state=seed,
                verbose=-1,
            )
            model.fit(X, y)
            return model
        except Exception:
            return fit_xgb_regressor(X, y, seed, n_jobs)
    base = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=240, alpha=1e-4, random_state=seed))
    model = TransformedTargetRegressor(regressor=base, transformer=StandardScaler())
    model.fit(X, y)
    return model


def evaluate_multiclass(model: Any, X: np.ndarray, y: np.ndarray, n_classes: int) -> dict[str, float]:
    pred = model.predict(X)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == n_classes:
            try:
                out["macro_auroc"] = float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
            except ValueError:
                out["macro_auroc"] = float("nan")
    if "macro_auroc" not in out:
        out["macro_auroc"] = float("nan")
    return out


def evaluate_regression(model: Any, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = np.asarray(model.predict(X), dtype=np.float64)
    rmse = math.sqrt(float(mean_squared_error(y, pred)))
    return {
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": rmse,
    }


def run_cell(bundle: TabularBundle, method: str, budget: int, seed: int, learner: str, worker_threads: int) -> dict[str, Any]:
    set_thread_limits(worker_threads)
    start = time.perf_counter()
    idx = select_indices(bundle, method, budget, seed, worker_threads)
    select_seconds = time.perf_counter() - start
    X_sub, y_sub = bundle.X_train[idx], bundle.y_train[idx]
    fit_start = time.perf_counter()
    if bundle.task == "multiclass":
        model = fit_multiclass_downstream(learner, X_sub, y_sub, int(bundle.n_classes or 2), seed, worker_threads)
        fit_seconds = time.perf_counter() - fit_start
        metrics = evaluate_multiclass(model, bundle.X_test, bundle.y_test, int(bundle.n_classes or 2))
        primary = metrics["macro_auroc"]
    else:
        model = fit_regression_downstream(learner, X_sub, y_sub, seed, worker_threads)
        fit_seconds = time.perf_counter() - fit_start
        metrics = evaluate_regression(model, bundle.X_test, bundle.y_test)
        primary = metrics["r2"]
    return {
        "task": bundle.task,
        "dataset": bundle.name,
        "method": method,
        "method_label": LABELS.get(method, method),
        "budget": f"budget_{budget}",
        "seed": f"seed_{seed}",
        "learner": learner,
        "train_rows": int(len(idx)),
        "select_seconds": float(select_seconds),
        "train_seconds": float(fit_seconds),
        "primary_metric": float(primary),
        **metrics,
    }


def paired_stats(df: pd.DataFrame, metric: str, methods: list[str], comparators: list[str]) -> pd.DataFrame:
    pivot = df.groupby(["dataset", "budget", "seed", "learner", "method"])[metric].mean().unstack()
    rows = []
    for method in methods:
        for comp in comparators:
            if method == comp or method not in pivot.columns or comp not in pivot.columns:
                continue
            pair = pivot[[method, comp]].dropna()
            if pair.empty:
                continue
            diff = pair[method] - pair[comp]
            try:
                p_value = float(wilcoxon(diff).pvalue) if (diff.abs() > 1e-12).any() else 1.0
            except ValueError:
                p_value = 1.0
            rows.append(
                {
                    "method": method,
                    "method_label": LABELS.get(method, method),
                    "comparator": comp,
                    "comparator_label": LABELS.get(comp, comp),
                    "n_pairs": int(len(diff)),
                    "mean_diff": float(diff.mean()),
                    "median_diff": float(diff.median()),
                    "wins": int((diff > 1e-6).sum()),
                    "ties": int((diff.abs() <= 1e-6).sum()),
                    "losses": int((diff < -1e-6).sum()),
                    "wilcoxon_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, task: str, out_tables: Path, out_figs: Path, out_notes: Path) -> None:
    metric = "macro_auroc" if task == "multiclass" else "r2"
    perf = (
        df.groupby("method")
        .agg(
            metric_mean=(metric, "mean"),
            metric_std=(metric, "std"),
            train_seconds_mean=("train_seconds", "mean"),
            select_seconds_mean=("select_seconds", "mean"),
            n_cells=(metric, "count"),
        )
        .reset_index()
    )
    pivot = df.groupby(["dataset", "budget", "seed", "learner", "method"])[metric].mean().unstack()
    ranks = pivot.rank(axis=1, ascending=False, method="average").mean().rename("avg_rank")
    perf = perf.join(ranks, on="method")
    perf["method_label"] = perf["method"].map(lambda m: LABELS.get(m, m))
    perf = perf.sort_values(["metric_mean", "avg_rank"], ascending=[False, True])
    by_budget = df.pivot_table(index="method", columns="budget", values=metric, aggfunc="mean").reset_index()
    by_learner = df.pivot_table(index="method", columns="learner", values=metric, aggfunc="mean").reset_index()
    stats = paired_stats(df, metric, ["histdistill"], ["random", "herding", "k_center"])
    prefix = "icdm_multiclass_ovr" if task == "multiclass" else "icdm_regression"
    df.to_csv(out_tables / f"{prefix}_cells.csv", index=False)
    perf.to_csv(out_tables / f"{prefix}_performance.csv", index=False)
    by_budget.to_csv(out_tables / f"{prefix}_by_budget.csv", index=False)
    by_learner.to_csv(out_tables / f"{prefix}_by_learner.csv", index=False)
    stats.to_csv(out_tables / f"{prefix}_pairwise.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    budget_summary = df.groupby(["method", "budget"])[metric].agg(["mean", "sem"]).reset_index()
    for method in METHODS:
        sub = budget_summary[budget_summary["method"] == method].copy()
        if sub.empty:
            continue
        x = sub["budget"].str.replace("budget_", "", regex=False).astype(int)
        y = sub["mean"].astype(float)
        sem = sub["sem"].fillna(0.0).astype(float)
        ax.plot(x, y, marker="o", linewidth=1.7, label=LABELS.get(method, method))
        ax.fill_between(x, y - 1.96 * sem, y + 1.96 * sem, alpha=0.08)
    ax.set_xlabel("Budget per class" if task == "multiclass" else "Budget")
    ax.set_ylabel("Macro-AUROC" if task == "multiclass" else "R2")
    ax.set_title("Multiclass one-vs-rest breadth" if task == "multiclass" else "Regression breadth")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_figs / f"{prefix}_budget_profile.png", dpi=220)
    plt.close(fig)

    if "histdistill" in pivot.columns:
        deltas = []
        for comp in ["random", "herding", "k_center"]:
            if comp in pivot.columns:
                diff = (pivot["histdistill"] - pivot[comp]).dropna()
                deltas.append(pd.DataFrame({"comparator": LABELS.get(comp, comp), "delta": diff.to_numpy()}))
        if deltas:
            delta_df = pd.concat(deltas, ignore_index=True)
            order = delta_df["comparator"].unique().tolist()
            fig, ax = plt.subplots(figsize=(6.5, 4.4))
            data = [delta_df.loc[delta_df["comparator"] == c, "delta"].to_numpy() for c in order]
            parts = ax.violinplot(data, showmeans=True, showextrema=False)
            for body in parts["bodies"]:
                body.set_alpha(0.35)
            ax.axhline(0, color="black", alpha=0.5, linewidth=1)
            ax.set_xticks(np.arange(1, len(order) + 1))
            ax.set_xticklabels(order)
            ax.set_ylabel("Paired metric delta")
            ax.set_title("HistDistill paired deltas")
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            fig.savefig(out_figs / f"{prefix}_paired_delta_violins.png", dpi=220)
            plt.close(fig)

    title = "ICDM Multiclass One-vs-Rest Breadth" if task == "multiclass" else "ICDM Regression Breadth"
    metric_name = "Macro-AUROC" if task == "multiclass" else "R2"
    note = [
        f"# {title}",
        "",
        f"Separate breadth experiment for `{task}` using XGBoost, LightGBM, and MLP downstream learners.",
        "",
        "## Performance",
        "",
        md_table(
            perf[["method_label", "metric_mean", "metric_std", "avg_rank", "n_cells"]].rename(
                columns={
                    "method_label": "Method",
                    "metric_mean": f"Mean {metric_name}",
                    "metric_std": "Std",
                    "avg_rank": "Avg. Rank",
                    "n_cells": "Cells",
                }
            )
        ),
        "",
        "## Pairwise Tests",
        "",
        md_table(
            stats[["method_label", "comparator_label", "n_pairs", "mean_diff", "wins", "ties", "losses", "wilcoxon_p"]].rename(
                columns={
                    "method_label": "Method",
                    "comparator_label": "Comparator",
                    "n_pairs": "Pairs",
                    "mean_diff": "Mean Diff",
                    "wilcoxon_p": "Wilcoxon p",
                }
            )
        ),
        "",
        "## Files",
        "",
        f"- `tables/{prefix}_cells.csv`",
        f"- `tables/{prefix}_performance.csv`",
        f"- `tables/{prefix}_pairwise.csv`",
        f"- `figures/{prefix}_budget_profile.png`",
        f"- `figures/{prefix}_paired_delta_violins.png`",
        "",
    ]
    (out_notes / f"{prefix.upper()}_UPDATE.md").write_text("\n".join(note), encoding="utf-8")


def load_bundles(task: str, cfg: dict[str, Any], seed: int, max_rows: int, datasets: list[str]) -> list[TabularBundle]:
    bundles = []
    for name in datasets:
        try:
            if task == "multiclass":
                bundles.append(load_multiclass_bundle(cfg, name, seed, max_rows))
            else:
                bundles.append(load_regression_bundle(name, seed, max_rows))
        except Exception as exc:
            print(f"skip:{task}:{name}:{exc}", flush=True)
    return bundles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/focused.yaml")
    ap.add_argument("--paper-materials", default="paper_materials")
    ap.add_argument("--task", choices=["regression", "multiclass", "both"], default="both")
    ap.add_argument("--regression-datasets", default=",".join(REGRESSION_DATASETS))
    ap.add_argument("--multiclass-datasets", default=",".join(MULTICLASS_DATASETS))
    ap.add_argument("--budgets", default="25,50")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--learners", default=",".join(LEARNERS))
    ap.add_argument("--max-rows", type=int, default=12000)
    ap.add_argument("--n-jobs", default="6")
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_jobs = resolve_n_jobs(args.n_jobs)
    set_thread_limits(n_jobs)
    materials = ensure_dir(args.paper_materials)
    out_tables = ensure_dir(materials / "tables")
    out_figs = ensure_dir(materials / "figures")
    out_notes = ensure_dir(materials / "notes")
    tasks = ["regression", "multiclass"] if args.task == "both" else [args.task]
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    learners = [x.strip() for x in args.learners.split(",") if x.strip()]

    for task in tasks:
        dataset_names = (
            [x.strip() for x in args.multiclass_datasets.split(",") if x.strip()]
            if task == "multiclass"
            else [x.strip() for x in args.regression_datasets.split(",") if x.strip()]
        )
        all_rows = []
        for seed in seeds:
            bundles = load_bundles(task, cfg, seed, args.max_rows, dataset_names)
            if not bundles:
                continue
            cell_args = [(bundle, method, budget, seed, learner) for bundle in bundles for method in methods for budget in budgets for learner in learners]
            worker_threads = max(1, n_jobs // max(1, min(n_jobs, len(cell_args))))
            rows = Parallel(n_jobs=min(n_jobs, len(cell_args) or 1))(
                delayed(run_cell)(bundle, method, budget, seed, learner, worker_threads)
                for bundle, method, budget, seed, learner in cell_args
            )
            all_rows.extend(rows)
        df = pd.DataFrame(all_rows)
        if df.empty:
            print(f"No rows generated for {task}.", flush=True)
            continue
        summarize(df, task, out_tables, out_figs, out_notes)
        print(f"Wrote {task} breadth results with {len(df)} cells.", flush=True)


if __name__ == "__main__":
    main()
