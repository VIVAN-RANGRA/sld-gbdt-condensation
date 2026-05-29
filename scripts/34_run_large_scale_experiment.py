from __future__ import annotations

import argparse
import gzip
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.datasets import fetch_covtype
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_large"
TABLES = ROOT / "paper_materials" / "tables"
FIGS = ROOT / "paper" / "figures"

URLS = {
    "higgs": "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz",
    "susy": "https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz",
}

METHOD_LABELS = {
    "histdistill_refined": "HD-Refined",
    "histdistill_density": "HD-Density",
    "herding": "Herding",
    "gradient_sampling": "Gradient sampling",
    "random": "Random",
}


@dataclass
class SplitData:
    name: str
    label: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


@dataclass
class Landscape:
    features: np.ndarray
    thresholds: list[np.ndarray]
    candidates: list[tuple[int, int]]
    gains: np.ndarray
    best_gain: float
    parent_hessian: float


def ensure_dirs() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"[data] using cached {path}")
        return
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, 8):
        done = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={done}-"} if done > 0 else {}
        mode = "ab" if done > 0 else "wb"
        try:
            action = "resuming" if done > 0 else "downloading"
            print(f"[data] {action} {url} at {done / 1e9:.2f} GB", flush=True)
            with requests.get(url, stream=True, timeout=(30, 120), headers=headers) as resp:
                if done > 0 and resp.status_code == 200:
                    print("[data] server ignored Range; restarting download", flush=True)
                    done = 0
                    mode = "wb"
                elif resp.status_code not in (200, 206):
                    resp.raise_for_status()
                total_header = int(resp.headers.get("content-length", "0") or 0)
                total = done + total_header if done > 0 and resp.status_code == 206 else total_header
                last = time.perf_counter()
                with tmp.open(mode) as f:
                    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        done += len(chunk)
                        now = time.perf_counter()
                        if now - last > 10:
                            if total:
                                print(f"[data] {path.name}: {done / 1e9:.2f}/{total / 1e9:.2f} GB", flush=True)
                            else:
                                print(f"[data] {path.name}: {done / 1e9:.2f} GB", flush=True)
                            last = now
            tmp.replace(path)
            return
        except Exception as exc:
            print(f"[data] download attempt {attempt} failed: {exc}", flush=True)
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"failed to download {url}")


def read_first_column_gzip(path: Path, max_rows: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    nrows = None if max_rows is None or max_rows <= 0 else int(max_rows)
    df = pd.read_csv(path, header=None, dtype=np.float32, nrows=nrows)
    y = df.iloc[:, 0].to_numpy(dtype=np.int8, copy=True)
    X = df.iloc[:, 1:].to_numpy(dtype=np.float32, copy=True)
    del df
    X = np.nan_to_num(X, copy=False)
    return X, y


def load_higgs(max_rows: int | None) -> SplitData:
    path = RAW / "higgs" / "HIGGS.csv.gz"
    download(URLS["higgs"], path)
    X, y = read_first_column_gzip(path, max_rows=max_rows)
    test_n = min(500_000, max(50_000, len(y) // 20))
    X_train, y_train = X[:-test_n], y[:-test_n]
    X_test, y_test = X[-test_n:], y[-test_n:]
    return SplitData("higgs", "HIGGS", X_train, y_train, X_test, y_test)


def load_susy(max_rows: int | None) -> SplitData:
    path = RAW / "susy" / "SUSY.csv.gz"
    download(URLS["susy"], path)
    X, y = read_first_column_gzip(path, max_rows=max_rows)
    test_n = min(500_000, max(50_000, len(y) // 10))
    X_train, y_train = X[:-test_n], y[:-test_n]
    X_test, y_test = X[-test_n:], y[-test_n:]
    return SplitData("susy", "SUSY", X_train, y_train, X_test, y_test)


def load_covertype(max_rows: int | None, seed: int) -> SplitData:
    cache = RAW / "covertype" / "covertype_binary.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        arr = np.load(cache)
        X = arr["X"].astype(np.float32, copy=False)
        y = arr["y"].astype(np.int8, copy=False)
    else:
        bunch = fetch_covtype(data_home=str(cache.parent), shuffle=False)
        X = np.asarray(bunch.data, dtype=np.float32)
        # Binary task: class 2 versus all other cover types.
        y = (np.asarray(bunch.target) == 2).astype(np.int8)
        np.savez_compressed(cache, X=X, y=y)
    if max_rows is not None and max_rows > 0 and max_rows < len(y):
        rng = np.random.default_rng(seed)
        idx = stratified_take(y, int(max_rows), rng)
        X, y = X[idx], y[idx]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=min(100_000, max(50_000, len(y) // 5)),
        random_state=seed,
        stratify=y,
    )
    return SplitData("covertype", "Covertype", X_train, y_train, X_test, y_test)


def load_dataset(name: str, max_rows: int | None, seed: int) -> SplitData:
    if name == "covertype":
        return load_covertype(max_rows, seed)
    if name == "higgs":
        return load_higgs(max_rows)
    if name == "susy":
        return load_susy(max_rows)
    raise ValueError(f"unknown dataset {name}")


def stratified_take(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    n = min(int(n), len(y))
    classes, counts = np.unique(y, return_counts=True)
    raw = counts / counts.sum() * n
    quotas = np.floor(raw).astype(int)
    quotas = np.maximum(quotas, 1)
    while quotas.sum() > n:
        i = int(np.argmax(quotas))
        quotas[i] -= 1
    order = np.argsort(-(raw - np.floor(raw)))
    while quotas.sum() < n:
        quotas[order[(quotas.sum() - n) % len(order)]] += 1
    idx: list[np.ndarray] = []
    for c, q in zip(classes, quotas):
        cls = np.flatnonzero(y == c)
        take = min(int(q), len(cls))
        idx.append(rng.choice(cls, size=take, replace=False))
    out = np.concatenate(idx).astype(np.int64)
    rng.shuffle(out)
    return out


def quotas_for_budget(y: np.ndarray, total_budget: int) -> dict[int, int]:
    classes, counts = np.unique(y, return_counts=True)
    total_budget = min(int(total_budget), len(y))
    raw = counts / counts.sum() * total_budget
    quotas = np.floor(raw).astype(int)
    quotas = np.maximum(quotas, 1)
    while quotas.sum() > total_budget:
        i = int(np.argmax(quotas))
        quotas[i] -= 1
    order = np.argsort(-(raw - np.floor(raw)))
    pos = 0
    while quotas.sum() < total_budget:
        quotas[order[pos % len(order)]] += 1
        pos += 1
    return {int(c): int(q) for c, q in zip(classes, quotas)}


def class_mass_weights(y: np.ndarray, selected: np.ndarray) -> np.ndarray:
    weights = np.ones(len(selected), dtype=np.float32)
    if len(selected) == 0:
        return weights
    for c in np.unique(y):
        mask = y[selected] == c
        if mask.any():
            weights[mask] = float(np.sum(y == c)) / float(np.sum(mask))
    weights *= len(weights) / max(1e-8, float(weights.sum()))
    return weights.astype(np.float32)


def fit_lgbm(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_jobs: int,
    sample_weight: np.ndarray | None = None,
    n_estimators: int = 80,
) -> tuple[LGBMClassifier, float]:
    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.07,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary",
        n_jobs=n_jobs,
        random_state=seed,
        force_col_wise=True,
        verbose=-1,
    )
    start = time.perf_counter()
    model.fit(X, y, sample_weight=sample_weight)
    return model, time.perf_counter() - start


def predict_positive(model: LGBMClassifier, X: np.ndarray, chunk_size: int = 500_000) -> np.ndarray:
    out = np.empty(len(X), dtype=np.float32)
    for start in range(0, len(X), chunk_size):
        end = min(start + chunk_size, len(X))
        out[start:end] = model.predict_proba(X[start:end])[:, 1].astype(np.float32)
    return out


def auc(model: LGBMClassifier, X: np.ndarray, y: np.ndarray) -> float:
    p = predict_positive(model, X)
    return float(roc_auc_score(y, p))


def gain_from_sums(GL: np.ndarray, HL: np.ndarray, G: float, H: float, reg_lambda: float = 1.0) -> np.ndarray:
    GR = G - GL
    HR = H - HL
    return (GL * GL) / (HL + reg_lambda) + (GR * GR) / (HR + reg_lambda) - (G * G) / (H + reg_lambda)


def build_landscape(
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    model: LGBMClassifier,
    max_features: int,
    quantiles: int,
    seed: int,
) -> Landscape:
    rng = np.random.default_rng(seed)
    importances = np.asarray(getattr(model, "feature_importances_", np.ones(X.shape[1])), dtype=np.float64)
    if np.all(importances <= 0):
        importances = np.var(X, axis=0)
    features = np.argsort(-importances)[: min(max_features, X.shape[1])]
    q_idx = stratified_take(y, min(250_000, len(y)), rng)
    q_grid = np.linspace(0.03, 0.97, quantiles)
    thresholds: list[np.ndarray] = []
    candidates: list[tuple[int, int]] = []
    gains: list[float] = []
    G = float(np.sum(g))
    H = float(np.sum(h))
    for j in features:
        vals = X[q_idx, int(j)]
        thr = np.unique(np.quantile(vals, q_grid).astype(np.float32))
        if len(thr) == 0:
            thresholds.append(thr)
            continue
        bin_id = np.searchsorted(thr, X[:, int(j)], side="right")
        GL_bins = np.bincount(bin_id, weights=g, minlength=len(thr) + 1)
        HL_bins = np.bincount(bin_id, weights=h, minlength=len(thr) + 1)
        GL = np.cumsum(GL_bins)[:-1]
        HL = np.cumsum(HL_bins)[:-1]
        fgains = gain_from_sums(GL, HL, G, H)
        local_index = len(thresholds)
        thresholds.append(thr)
        for b, gain in enumerate(fgains):
            candidates.append((local_index, int(b)))
            gains.append(float(gain))
    gains_arr = np.asarray(gains, dtype=np.float64)
    best_gain = float(np.max(gains_arr)) if len(gains_arr) else 0.0
    return Landscape(np.asarray(features, dtype=np.int64), thresholds, candidates, gains_arr, best_gain, H)


def condensed_gains(
    landscape: Landscape,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
) -> np.ndarray:
    if sample_weight is not None:
        w0 = np.asarray(sample_weight, dtype=np.float64)
        p0 = float(np.sum(w0 * y) / max(1e-8, float(np.sum(w0))))
    else:
        w0 = None
        p0 = float(np.mean(y))
    p0 = float(np.clip(p0, 1e-6, 1.0 - 1e-6))
    g = (np.full(len(y), p0, dtype=np.float64) - y).astype(np.float64)
    h = np.full(len(y), max(p0 * (1.0 - p0), 1e-6), dtype=np.float64)
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64).copy()
        current_hessian = float(np.sum(h * w))
        if current_hessian > 1e-12:
            w *= float(landscape.parent_hessian) / current_hessian
        g = g * w
        h = h * w
    G = float(np.sum(g))
    H = float(np.sum(h))
    out: list[float] = []
    for local_j, thr in enumerate(landscape.thresholds):
        if len(thr) == 0:
            continue
        j = int(landscape.features[local_j])
        bin_id = np.searchsorted(thr, X[:, j], side="right")
        GL_bins = np.bincount(bin_id, weights=g, minlength=len(thr) + 1)
        HL_bins = np.bincount(bin_id, weights=h, minlength=len(thr) + 1)
        GL = np.cumsum(GL_bins)[:-1]
        HL = np.cumsum(HL_bins)[:-1]
        out.extend(gain_from_sums(GL, HL, G, H).tolist())
    return np.asarray(out, dtype=np.float64)


def split_regret(landscape: Landscape, gains_other: np.ndarray) -> float:
    if len(landscape.gains) == 0 or len(gains_other) == 0:
        return float("nan")
    other_best = int(np.argmax(gains_other))
    chosen_full_gain = float(landscape.gains[other_best])
    regret = max(0.0, landscape.best_gain - chosen_full_gain)
    scale = max(1.0, abs(landscape.best_gain), float(np.max(np.abs(landscape.gains))))
    return float(regret / scale)


def select_random(y: np.ndarray, total_budget: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    quotas = quotas_for_budget(y, total_budget)
    idx: list[np.ndarray] = []
    for c, q in quotas.items():
        cls = np.flatnonzero(y == c)
        idx.append(rng.choice(cls, size=min(q, len(cls)), replace=False))
    out = np.concatenate(idx).astype(np.int64)
    rng.shuffle(out)
    return out


def select_gradient(y: np.ndarray, g: np.ndarray, total_budget: int) -> np.ndarray:
    quotas = quotas_for_budget(y, total_budget)
    idx: list[np.ndarray] = []
    for c, q in quotas.items():
        cls = np.flatnonzero(y == c)
        order = cls[np.argsort(-np.abs(g[cls]))]
        idx.append(order[: min(q, len(order))])
    return np.concatenate(idx).astype(np.int64)


def pool_indices(y: np.ndarray, g: np.ndarray, pool_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pool_size = min(pool_size, len(y))
    random_part = stratified_take(y, max(pool_size // 2, 1), rng)
    grad_part = []
    quotas = quotas_for_budget(y, pool_size - len(random_part))
    for c, q in quotas.items():
        cls = np.flatnonzero(y == c)
        order = cls[np.argsort(-np.abs(g[cls]))]
        grad_part.append(order[: min(q, len(order))])
    merged = np.unique(np.concatenate([random_part] + grad_part)).astype(np.int64)
    if len(merged) > pool_size:
        merged = rng.choice(merged, size=pool_size, replace=False)
    return merged


def sketch_matrix(
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    idx: np.ndarray,
    landscape: Landscape,
    density: bool,
    scaler: StandardScaler | None,
    top_candidates: int = 28,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    order = np.argsort(-landscape.gains)[: min(top_candidates, len(landscape.gains))]
    cols: list[np.ndarray] = []
    targets: dict[int, list[float]] = {int(c): [] for c in np.unique(y)}
    for cand in order:
        local_j, b = landscape.candidates[int(cand)]
        j = int(landscape.features[local_j])
        thr = float(landscape.thresholds[local_j][b])
        left_pool = (X[idx, j] <= thr).astype(np.float32)
        cols.append(left_pool)
        gs = max(float(np.std(g)), 1e-6)
        hs = max(float(np.std(h)), 1e-6)
        cols.append(left_pool * (g[idx] / gs).astype(np.float32))
        cols.append(left_pool * (h[idx] / hs).astype(np.float32))
        for c in targets:
            mask = y == c
            left_full = X[mask, j] <= thr
            if np.any(mask):
                targets[c].append(float(np.mean(left_full)))
                targets[c].append(float(np.mean(left_full * (g[mask] / gs))))
                targets[c].append(float(np.mean(left_full * (h[mask] / hs))))
            else:
                targets[c].extend([0.0, 0.0, 0.0])
    if density:
        if scaler is None:
            scaler = StandardScaler().fit(X[idx])
        raw = scaler.transform(X[idx]).astype(np.float32)
        max_raw = min(16, raw.shape[1])
        for j in range(max_raw):
            cols.append(raw[:, j])
        for c in targets:
            mask = y == c
            if np.any(mask):
                raw_c = scaler.transform(X[mask][: min(200_000, int(np.sum(mask)))])
                targets[c].extend(np.mean(raw_c[:, :max_raw], axis=0).astype(float).tolist())
            else:
                targets[c].extend([0.0] * max_raw)
    S = np.vstack(cols).T.astype(np.float32)
    scale = np.maximum(np.std(S, axis=0), 1e-6).astype(np.float32)
    S = S / scale[None, :]
    target_arrays = {c: np.asarray(v, dtype=np.float32) / scale for c, v in targets.items()}
    return S, target_arrays


def fast_mean_match(
    S: np.ndarray,
    idx_pool: np.ndarray,
    y_pool: np.ndarray,
    y_full: np.ndarray,
    total_budget: int,
    targets: dict[int, np.ndarray],
    seed: int,
    batch_size: int = 20_000,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected_local: list[int] = []
    quotas = quotas_for_budget(y_full, total_budget)
    chosen = np.zeros(len(idx_pool), dtype=bool)
    for c, q in quotas.items():
        cls_local = np.flatnonzero(y_pool == c)
        if len(cls_local) == 0:
            continue
        running = np.zeros(S.shape[1], dtype=np.float32)
        target = targets.get(int(c), np.zeros(S.shape[1], dtype=np.float32))
        cls_chosen = np.zeros(len(cls_local), dtype=bool)
        for t in range(min(q, len(cls_local))):
            available = np.flatnonzero(~cls_chosen)
            if len(available) > batch_size:
                available = rng.choice(available, size=batch_size, replace=False)
            cand_local = cls_local[available]
            desired = (t + 1) * target - running
            diff = S[cand_local] - desired[None, :]
            scores = np.einsum("ij,ij->i", diff, diff)
            best_pos = int(np.argmin(scores))
            best_cls_pos = int(available[best_pos])
            best_local = int(cls_local[best_cls_pos])
            selected_local.append(best_local)
            cls_chosen[best_cls_pos] = True
            chosen[best_local] = True
            running += S[best_local]
    return idx_pool[np.asarray(selected_local, dtype=np.int64)]


def select_herding(X: np.ndarray, y: np.ndarray, total_budget: int, seed: int, pool: np.ndarray) -> np.ndarray:
    scaler = StandardScaler().fit(X[pool])
    S = scaler.transform(X[pool]).astype(np.float32)
    max_cols = min(24, S.shape[1])
    S = S[:, :max_cols]
    scale = np.maximum(np.std(S, axis=0), 1e-6)
    S = S / scale[None, :]
    targets: dict[int, np.ndarray] = {}
    for c in np.unique(y):
        mask = y == c
        sample = np.flatnonzero(mask)[: min(200_000, int(np.sum(mask)))]
        targets[int(c)] = (scaler.transform(X[sample])[:, :max_cols].mean(axis=0) / scale).astype(np.float32)
    return fast_mean_match(S, pool, y[pool], y, total_budget, targets, seed)


def select_histdistill(
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    landscape: Landscape,
    total_budget: int,
    seed: int,
    pool: np.ndarray,
    density: bool,
) -> np.ndarray:
    scaler = StandardScaler().fit(X[pool]) if density else None
    S, targets = sketch_matrix(X, y, g, h, pool, landscape, density=density, scaler=scaler)
    return fast_mean_match(S, pool, y[pool], y, total_budget, targets, seed)


def existing_cells(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def should_skip(existing: pd.DataFrame, dataset: str, n_train: int, method: str, force: bool) -> bool:
    if force or existing.empty:
        return False
    mask = (
        existing["dataset"].eq(dataset)
        & existing["n_train"].eq(int(n_train))
        & existing["method"].eq(method)
    )
    return bool(mask.any())


def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def n_grid(max_n: int, requested: Iterable[int]) -> list[int]:
    vals = sorted({int(n) for n in requested if int(n) <= max_n})
    if max_n not in vals:
        vals.append(int(max_n))
    return vals


def run_dataset(
    data: SplitData,
    requested_n: list[int],
    methods: list[str],
    total_budget: int,
    seed: int,
    n_jobs: int,
    force: bool,
    quick: bool,
) -> None:
    cells_path = TABLES / "large_scale_cells.csv"
    existing = existing_cells(cells_path)
    max_train = len(data.y_train)
    grid = n_grid(max_train, requested_n)
    rng = np.random.default_rng(seed)
    print(f"[run] {data.label}: train={len(data.y_train):,}, test={len(data.y_test):,}, grid={grid}")
    for n_train in grid:
        if quick and n_train > 100_000:
            continue
        sub_idx = stratified_take(data.y_train, n_train, rng)
        X_train = data.X_train[sub_idx]
        y_train = data.y_train[sub_idx]
        print(f"[run] {data.label} N={n_train:,}: full-data LightGBM")
        full_model, full_seconds = fit_lgbm(X_train, y_train, seed, n_jobs)
        full_auc = auc(full_model, data.X_test, data.y_test)
        p_train = predict_positive(full_model, X_train)
        g_model = (p_train - y_train).astype(np.float32)
        p0 = float(np.clip(np.mean(y_train), 1e-6, 1.0 - 1e-6))
        g = (np.full(len(y_train), p0, dtype=np.float32) - y_train).astype(np.float32)
        h = np.full(len(y_train), max(p0 * (1.0 - p0), 1e-6), dtype=np.float32)
        print(f"[run] {data.label} N={n_train:,}: landscape")
        landscape_start = time.perf_counter()
        landscape = build_landscape(
            X_train,
            y_train,
            g,
            h,
            full_model,
            max_features=8 if quick else 16,
            quantiles=8 if quick else 16,
            seed=seed,
        )
        landscape_seconds = time.perf_counter() - landscape_start
        pool_size = min(30_000 if quick else 90_000, len(y_train))
        pool = pool_indices(y_train, g_model, pool_size, seed)
        rows: list[dict[str, object]] = []
        for method in methods:
            if should_skip(existing, data.name, n_train, method, force):
                print(f"[skip] {data.label} N={n_train:,} {method}")
                continue
            print(f"[run] {data.label} N={n_train:,} {method}")
            select_start = time.perf_counter()
            if method == "random":
                selected = select_random(y_train, total_budget, seed)
            elif method == "gradient_sampling":
                selected = select_gradient(y_train, g_model, total_budget)
            elif method == "herding":
                selected = select_herding(X_train, y_train, total_budget, seed, pool)
            elif method == "histdistill_refined":
                selected = select_histdistill(X_train, y_train, g, h, landscape, total_budget, seed, pool, density=False)
            elif method == "histdistill_density":
                selected = select_histdistill(X_train, y_train, g, h, landscape, total_budget, seed, pool, density=True)
            else:
                raise ValueError(f"unknown method {method}")
            select_seconds = time.perf_counter() - select_start
            weights = class_mass_weights(y_train, selected)
            sub_model, sub_seconds = fit_lgbm(
                X_train[selected],
                y_train[selected],
                seed,
                n_jobs,
                sample_weight=weights,
            )
            sub_auc = auc(sub_model, data.X_test, data.y_test)
            regret_start = time.perf_counter()
            other = condensed_gains(landscape, X_train[selected], y_train[selected], weights)
            regret = split_regret(landscape, other)
            regret_seconds = time.perf_counter() - regret_start
            rows.append(
                {
                    "dataset": data.name,
                    "dataset_label": data.label,
                    "n_train": int(n_train),
                    "n_test": int(len(data.y_test)),
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "budget_rows": int(len(selected)),
                    "full_train_seconds": full_seconds,
                    "condensed_train_seconds": sub_seconds,
                    "select_seconds": select_seconds,
                    "landscape_seconds": landscape_seconds,
                    "split_regret_seconds": regret_seconds,
                    "speedup": full_seconds / max(sub_seconds, 1e-6),
                    "full_auroc": full_auc,
                    "auroc": sub_auc,
                    "auroc_gap": full_auc - sub_auc,
                    "split_regret": regret,
                }
            )
        append_rows(cells_path, rows)


def summarize() -> tuple[pd.DataFrame, pd.DataFrame]:
    cells_path = TABLES / "large_scale_cells.csv"
    cells = pd.read_csv(cells_path)
    summary_rows: list[dict[str, object]] = []
    for dataset, sub in cells.groupby("dataset"):
        max_n = int(sub["n_train"].max())
        last = sub[sub["n_train"].eq(max_n)].copy()
        rho = float("nan")
        pval = float("nan")
        if len(last) >= 3 and last["split_regret"].nunique() > 1 and last["auroc"].nunique() > 1:
            rho, pval = spearmanr(last["split_regret"], last["auroc"])
        summary_rows.append(
            {
                "dataset": dataset,
                "dataset_label": str(last["dataset_label"].iloc[0]),
                "n_train": max_n,
                "methods": int(last["method"].nunique()),
                "spearman_regret_auroc": rho,
                "spearman_p": pval,
                "top_speedup": float(last["speedup"].max()),
                "top_speedup_method": str(last.loc[last["speedup"].idxmax(), "method_label"]),
                "best_auroc": float(last["auroc"].max()),
                "best_auroc_method": str(last.loc[last["auroc"].idxmax(), "method_label"]),
                "full_auroc": float(last["full_auroc"].iloc[0]),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("n_train")
    summary.to_csv(TABLES / "large_scale_summary.csv", index=False)
    tiny = summary[["dataset_label", "n_train", "spearman_regret_auroc", "top_speedup", "top_speedup_method"]].copy()
    tiny.to_csv(TABLES / "large_scale_tiny_table.csv", index=False)
    return cells, summary


def plot_speedup(cells: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.9, 4.3))
    colors = {
        "histdistill_refined": "#2A6FBB",
        "histdistill_density": "#00897B",
        "herding": "#7B5EA7",
        "gradient_sampling": "#C45A2A",
        "random": "#666666",
    }
    for method, sub in cells.groupby("method"):
        agg = (
            sub.groupby("n_train", as_index=False)
            .agg(speedup=("speedup", "median"))
            .sort_values("n_train")
        )
        ax.plot(
            agg["n_train"],
            agg["speedup"],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            label=METHOD_LABELS.get(method, method),
            color=colors.get(method),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training rows")
    ax.set_ylabel("Retrain speedup over full data")
    ax.set_title("Large-scale retraining payoff")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, frameon=True, ncol=1)
    fig.tight_layout()
    out = FIGS / "fig_large_scale_speedup.png"
    fig.savefig(out, dpi=240)
    print(f"[plot] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="covertype,susy,higgs")
    ap.add_argument("--methods", default="histdistill_refined,histdistill_density,herding,gradient_sampling,random")
    ap.add_argument("--n-grid", default="20000,100000,500000,1000000,5000000,11000000")
    ap.add_argument("--budget-rows", type=int, default=400)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--max-rows", type=int, default=0, help="Optional cap per raw dataset for debugging.")
    ap.add_argument("--quick", action="store_true", help="Run only small smoke points.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    requested_n = [int(x) for x in args.n_grid.split(",") if x.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    max_rows = None if args.max_rows <= 0 else int(args.max_rows)
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        data = load_dataset(name, max_rows=max_rows, seed=args.seed)
        run_dataset(
            data,
            requested_n=requested_n,
            methods=methods,
            total_budget=args.budget_rows,
            seed=args.seed,
            n_jobs=args.n_jobs,
            force=args.force,
            quick=args.quick,
        )
        del data
    cells, summary = summarize()
    plot_speedup(cells)
    print("[summary]")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
