from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier


@dataclass
class FitResult:
    model: Any
    train_seconds: float
    learner: str
    backend: str


def make_xgb_like(cfg: dict[str, Any], n_jobs: int):
    try:
        from xgboost import XGBClassifier

        tcfg = cfg["teacher"]
        return XGBClassifier(
            n_estimators=int(tcfg["n_estimators"]),
            max_depth=int(tcfg["max_depth"]),
            learning_rate=float(tcfg["learning_rate"]),
            reg_lambda=float(tcfg["reg_lambda"]),
            gamma=float(tcfg["gamma"]),
            subsample=float(tcfg["subsample"]),
            colsample_bytree=float(tcfg["colsample_bytree"]),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=n_jobs,
            random_state=int(cfg["project"]["seed"]),
        ), "xgboost"
    except ImportError:
        return HistGradientBoostingClassifier(
            max_iter=int(cfg["teacher"]["n_estimators"]),
            max_leaf_nodes=2 ** int(cfg["teacher"]["max_depth"]),
            learning_rate=float(cfg["teacher"]["learning_rate"]),
            l2_regularization=float(cfg["teacher"]["reg_lambda"]),
            random_state=int(cfg["project"]["seed"]),
        ), "sklearn_hist_gradient_boosting"


def fit_teacher(X: np.ndarray, y: np.ndarray, cfg: dict[str, Any], n_jobs: int, sample_weight: np.ndarray | None = None) -> FitResult:
    model, backend = make_xgb_like(cfg, n_jobs)
    start = time.perf_counter()
    try:
        model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(X, y)
    return FitResult(model=model, train_seconds=time.perf_counter() - start, learner="teacher", backend=backend)


def fit_downstream(
    learner: str,
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict[str, Any],
    n_jobs: int,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> FitResult:
    if learner == "xgboost":
        model, backend = make_xgb_like(cfg, n_jobs)
    elif learner == "lightgbm":
        try:
            from lightgbm import LGBMClassifier

            model = LGBMClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.06,
                num_leaves=15,
                subsample=0.9,
                colsample_bytree=0.9,
                n_jobs=n_jobs,
                random_state=seed,
                verbose=-1,
            )
            backend = "lightgbm"
        except ImportError:
            model = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=seed)
            backend = "sklearn_hist_gradient_boosting"
    elif learner == "catboost":
        try:
            from catboost import CatBoostClassifier

            model = CatBoostClassifier(
                iterations=120,
                depth=4,
                learning_rate=0.06,
                loss_function="Logloss",
                eval_metric="AUC",
                thread_count=n_jobs,
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
            )
            backend = "catboost"
        except ImportError:
            model = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=seed)
            backend = "sklearn_hist_gradient_boosting"
    elif learner == "random_forest":
        model = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=n_jobs, random_state=seed)
        backend = "sklearn_random_forest"
    elif learner == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, max_iter=300, random_state=seed)
        backend = "sklearn_mlp"
    else:
        model = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=seed)
        backend = "sklearn_hist_gradient_boosting"
    start = time.perf_counter()
    try:
        model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(X, y)
    return FitResult(model=model, train_seconds=time.perf_counter() - start, learner=learner, backend=backend)


def predict_proba_positive(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 else proba
    raw = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-raw))


def evaluate_binary(model: Any, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    p = np.clip(predict_proba_positive(model, X), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    out = {"accuracy": float(accuracy_score(y, pred)), "log_loss": float(log_loss(y, p))}
    try:
        out["auroc"] = float(roc_auc_score(y, p))
    except ValueError:
        out["auroc"] = float("nan")
    return out
