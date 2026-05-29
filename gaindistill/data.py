from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .utils import ensure_dir, read_json, slugify, write_json


@dataclass
class DatasetBundle:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta: dict[str, Any]


def _save_raw_frame(out_dir: Path, name: str, frame: pd.DataFrame, target: str, source: str) -> None:
    ds_dir = ensure_dir(out_dir / slugify(name))
    frame.to_csv(ds_dir / "data.csv", index=False)
    write_json(ds_dir / "metadata.json", {"name": name, "target": target, "source": source, "rows": len(frame)})


def download_openml_curated(names: list[str], out_dir: str | Path, max_datasets: int | None = None) -> list[str]:
    saved: list[str] = []
    out = ensure_dir(out_dir)
    for name in names[: max_datasets or len(names)]:
        try:
            ds = fetch_openml(name=name, version="active", as_frame=True, parser="auto")
            frame = ds.frame.copy()
            target = ds.target_names[0] if ds.target_names else "target"
            if target not in frame.columns:
                frame[target] = ds.target
            _save_raw_frame(out, name, frame, target, "openml")
            saved.append(slugify(name))
        except Exception as exc:  # pragma: no cover - depends on network/catalog state.
            warnings.warn(f"OpenML download failed for {name}: {exc}")
    return saved


def download_openml_suite(suite_id: int, out_dir: str | Path, max_datasets: int | None = None) -> list[str]:
    try:
        import openml
    except ImportError:
        return []

    suite = openml.study.get_suite(suite_id)
    saved: list[str] = []
    out = ensure_dir(out_dir)
    for task_id in suite.tasks[: max_datasets or len(suite.tasks)]:
        try:
            task = openml.tasks.get_task(task_id)
            ds = task.get_dataset()
            X, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
            frame = X.copy()
            target = ds.default_target_attribute or "target"
            frame[target] = y
            _save_raw_frame(out, ds.name, frame, target, "openml_suite")
            saved.append(slugify(ds.name))
        except Exception as exc:  # pragma: no cover - depends on network/catalog state.
            warnings.warn(f"OpenML suite task {task_id} failed: {exc}")
    return saved


def download_pmlb(out_dir: str | Path, max_datasets: int | None = None) -> list[str]:
    try:
        from pmlb import classification_dataset_names, fetch_data
    except ImportError:
        raise RuntimeError("pmlb is not installed. Run `pip install -r requirements.txt` to enable PMLB downloads.")

    saved: list[str] = []
    out = ensure_dir(out_dir)
    for name in classification_dataset_names[: max_datasets or len(classification_dataset_names)]:
        try:
            frame = fetch_data(name)
            target = "target"
            _save_raw_frame(out, name, frame, target, "pmlb")
            saved.append(slugify(name))
        except Exception as exc:  # pragma: no cover - depends on network/catalog state.
            warnings.warn(f"PMLB download failed for {name}: {exc}")
    return saved


def create_smoke_dataset(out_dir: str | Path, n_samples: int = 1500, seed: int = 0) -> str:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=seed,
        class_sep=1.15,
    )
    cols = [f"x{i}" for i in range(X.shape[1])]
    frame = pd.DataFrame(X, columns=cols)
    frame["target"] = y
    _save_raw_frame(ensure_dir(out_dir), "smoke_binary", frame, "target", "sklearn")
    return "smoke_binary"


def _infer_and_encode(frame: pd.DataFrame, target: str, max_cat_bins: int, rare_min_count: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y_raw = frame[target]
    X_df = frame.drop(columns=[target]).copy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw.astype(str))
    keep_classes = np.unique(y)
    if len(keep_classes) != 2:
        raise ValueError(f"Initial implementation supports binary classification; found {len(keep_classes)} classes.")

    feature_meta: list[dict[str, Any]] = []
    encoded_cols: list[np.ndarray] = []
    for col in X_df.columns:
        s = X_df[col]
        if pd.api.types.is_numeric_dtype(s):
            vals = pd.to_numeric(s, errors="coerce")
            fill = float(vals.median()) if vals.notna().any() else 0.0
            arr = vals.fillna(fill).to_numpy(dtype=np.float32)
            feature_meta.append({"name": str(col), "kind": "numeric", "fill": fill})
            encoded_cols.append(arr)
        else:
            obj = s.astype("string").fillna("__MISSING__")
            counts = obj.value_counts(dropna=False)
            common = set(counts[counts >= rare_min_count].index[: max_cat_bins - 2])
            obj = obj.where(obj.isin(common), "__RARE__")
            cats = sorted(obj.unique().tolist())
            mapping = {v: i for i, v in enumerate(cats)}
            arr = obj.map(mapping).to_numpy(dtype=np.float32)
            feature_meta.append({"name": str(col), "kind": "categorical", "categories": cats})
            encoded_cols.append(arr)
    X = np.vstack(encoded_cols).T.astype(np.float32)
    meta = {"target": target, "classes": label_encoder.classes_.tolist(), "features": feature_meta}
    return X, y.astype(np.int64), meta


def _make_bins(X_train: np.ndarray, features: list[dict[str, Any]], numerical_bins: int, max_cat_bins: int) -> list[dict[str, Any]]:
    bins: list[dict[str, Any]] = []
    for j, fmeta in enumerate(features):
        col = X_train[:, j]
        if fmeta["kind"] == "categorical":
            values = np.unique(col.astype(int))
            values = values[:max_cat_bins]
            edges = (values + 0.5).astype(float).tolist()
            bins.append({"kind": "categorical", "values": values.astype(float).tolist(), "thresholds": edges[:-1] if len(edges) > 1 else edges})
        else:
            quantiles = np.linspace(0, 1, numerical_bins + 1)[1:-1]
            thresholds = np.unique(np.quantile(col, quantiles)).astype(float)
            if thresholds.size == 0:
                thresholds = np.array([float(np.median(col))], dtype=float)
            bins.append({"kind": "numeric", "thresholds": thresholds.tolist()})
    return bins


def preprocess_raw_dataset(raw_dir: str | Path, processed_dir: str | Path, dataset: str, cfg: dict[str, Any], seed: int) -> Path:
    raw_ds = Path(raw_dir) / dataset
    meta_raw = read_json(raw_ds / "metadata.json")
    frame = pd.read_csv(raw_ds / "data.csv")
    if cfg["data"].get("max_rows") and len(frame) > int(cfg["data"]["max_rows"]):
        frame = frame.sample(int(cfg["data"]["max_rows"]), random_state=seed).reset_index(drop=True)
    X, y, meta = _infer_and_encode(
        frame,
        meta_raw["target"],
        int(cfg["data"]["max_categorical_bins"]),
        int(cfg["data"]["rare_category_min_count"]),
    )
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=float(cfg["data"]["test_size"]), random_state=seed, stratify=y
    )
    val_ratio = float(cfg["data"]["val_size"]) / (1.0 - float(cfg["data"]["test_size"]))
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=seed, stratify=y_temp)
    meta["name"] = dataset
    meta["source"] = meta_raw.get("source", "unknown")
    meta["bins"] = _make_bins(X_train, meta["features"], int(cfg["data"]["numerical_bins"]), int(cfg["data"]["max_categorical_bins"]))
    meta["n_train"] = int(len(y_train))
    out = ensure_dir(Path(processed_dir) / dataset)
    np.savez_compressed(out / "arrays.npz", X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
    write_json(out / "metadata.json", meta)
    return out


def load_processed(path: str | Path) -> DatasetBundle:
    p = Path(path)
    arr = np.load(p / "arrays.npz")
    meta = read_json(p / "metadata.json")
    return DatasetBundle(
        name=meta["name"],
        X_train=arr["X_train"],
        y_train=arr["y_train"],
        X_val=arr["X_val"],
        y_val=arr["y_val"],
        X_test=arr["X_test"],
        y_test=arr["y_test"],
        meta=meta,
    )
