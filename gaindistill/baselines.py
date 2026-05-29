from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import lsq_linear
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from .landscape import candidate_grid, compute_grad_hess, grad_hess_from_margin, raw_margin_at_checkpoint, route_mask


def _take_per_class(y: np.ndarray, budget: int, rng: np.random.Generator) -> list[int]:
    idx: list[int] = []
    for c in np.unique(y):
        cls = np.flatnonzero(y == c)
        rng.shuffle(cls)
        idx.extend(cls[: min(budget, len(cls))].tolist())
    return idx


def random_subset(X: np.ndarray, y: np.ndarray, budget: int, seed: int, **_: Any) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = _take_per_class(y, budget, rng)
    return {"X": X[idx].astype(np.float32), "y": y[idx].astype(np.int64), "weights": np.ones(len(idx), dtype=np.float32)}


def full_data(X: np.ndarray, y: np.ndarray, budget: int, seed: int, **_: Any) -> dict[str, np.ndarray]:
    del budget, seed
    return {"X": X.astype(np.float32), "y": y.astype(np.int64), "weights": np.ones(len(y), dtype=np.float32)}


def k_center(X: np.ndarray, y: np.ndarray, budget: int, seed: int, **_: Any) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    Xs = StandardScaler().fit_transform(X)
    selected: list[int] = []
    for c in np.unique(y):
        cls = np.flatnonzero(y == c)
        if len(cls) <= budget:
            selected.extend(cls.tolist())
            continue
        first = int(rng.choice(cls))
        chosen = [first]
        dmin = pairwise_distances(Xs[cls], Xs[[first]]).ravel()
        for _ in range(1, budget):
            nxt = int(cls[np.argmax(dmin)])
            chosen.append(nxt)
            dmin = np.minimum(dmin, pairwise_distances(Xs[cls], Xs[[nxt]]).ravel())
        selected.extend(chosen)
    return {"X": X[selected].astype(np.float32), "y": y[selected].astype(np.int64), "weights": np.ones(len(selected), dtype=np.float32)}


def herding(X: np.ndarray, y: np.ndarray, budget: int, seed: int, **_: Any) -> dict[str, np.ndarray]:
    del seed
    Xs = StandardScaler().fit_transform(X)
    selected: list[int] = []
    for c in np.unique(y):
        cls = np.flatnonzero(y == c)
        target = Xs[cls].mean(axis=0)
        running = np.zeros_like(target)
        chosen: list[int] = []
        remaining = set(cls.tolist())
        for t in range(min(budget, len(cls))):
            best = min(remaining, key=lambda i: np.linalg.norm((running + Xs[i]) / (t + 1) - target))
            chosen.append(best)
            remaining.remove(best)
            running += Xs[best]
        selected.extend(chosen)
    return {"X": X[selected].astype(np.float32), "y": y[selected].astype(np.int64), "weights": np.ones(len(selected), dtype=np.float32)}


def gradient_sampling(X: np.ndarray, y: np.ndarray, budget: int, seed: int, teacher: Any | None = None, **_: Any) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    if teacher is None:
        return random_subset(X, y, budget, seed)
    g, _ = compute_grad_hess(teacher, X, y)
    selected: list[int] = []
    for c in np.unique(y):
        cls = np.flatnonzero(y == c)
        order = cls[np.argsort(-np.abs(g[cls]))]
        if len(order) < budget:
            extra = rng.choice(cls, size=budget - len(order), replace=True)
            order = np.concatenate([order, extra])
        selected.extend(order[: min(budget, len(order))].tolist())
    return {"X": X[selected].astype(np.float32), "y": y[selected].astype(np.int64), "weights": np.ones(len(selected), dtype=np.float32)}


def distribution_matching(X: np.ndarray, y: np.ndarray, budget: int, seed: int, meta: dict[str, Any], **_: Any) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for c in np.unique(y):
        cls = np.flatnonzero(y == c)
        Xc = X[cls]
        for _i in range(min(budget, len(cls))):
            row = np.zeros(X.shape[1], dtype=np.float32)
            for j, bmeta in enumerate(meta["bins"]):
                thresholds = np.asarray(bmeta["thresholds"], dtype=np.float32)
                ids = np.searchsorted(thresholds, Xc[:, j], side="left")
                probs = np.bincount(ids, minlength=len(thresholds) + 1).astype(float)
                probs /= max(1.0, probs.sum())
                bid = int(rng.choice(np.arange(len(probs)), p=probs))
                if bid == 0:
                    row[j] = thresholds[0] - 1e-3 if len(thresholds) else float(np.median(Xc[:, j]))
                elif bid >= len(thresholds):
                    row[j] = thresholds[-1] + 1e-3 if len(thresholds) else float(np.median(Xc[:, j]))
                else:
                    row[j] = 0.5 * (thresholds[bid - 1] + thresholds[bid])
            rows.append(row)
            labels.append(int(c))
    return {"X": np.vstack(rows).astype(np.float32), "y": np.asarray(labels, dtype=np.int64), "weights": np.ones(len(labels), dtype=np.float32)}


def tree_region_sampling(X: np.ndarray, y: np.ndarray, budget: int, seed: int, teacher: Any | None = None, **kwargs: Any) -> dict[str, np.ndarray]:
    return gradient_sampling(X, y, budget, seed, teacher=teacher, **kwargs)


def _thresholds(meta: dict[str, Any], j: int) -> np.ndarray:
    return np.asarray(meta["bins"][j]["thresholds"], dtype=np.float32)


def _gain_sketch(
    X: np.ndarray,
    y: np.ndarray,
    meta: dict[str, Any],
    teacher: Any | None,
    landscape: Any | None,
    include_grad: bool = True,
    include_hess: bool = True,
    include_conf: bool = True,
    include_raw: bool = True,
    include_gain_weight: bool = True,
    include_path: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if landscape is not None:
        candidates = np.asarray(landscape.candidates, dtype=np.int64)
    else:
        candidates = candidate_grid(meta)
        if len(candidates) > 128:
            candidates = candidates[:128]
    if teacher is not None:
        g, h = compute_grad_hess(teacher, X, y)
    else:
        p = np.full(len(y), float(np.mean(y)))
        g = p - y
        h = np.maximum(p * (1.0 - p), 1e-6)
    g_scale = max(float(np.std(g)), 1e-3)
    h_scale = max(float(np.std(h)), 1e-3)
    cols: list[np.ndarray] = []
    gain_weight = None
    if include_gain_weight and landscape is not None and len(getattr(landscape, "gains", [])) == len(candidates):
        gain_weight = np.sqrt(np.maximum(0.0, np.asarray(landscape.gains, dtype=np.float64) - np.min(landscape.gains) + 1e-6))
        gain_weight = gain_weight / max(float(np.mean(gain_weight)), 1e-6)
    for idx, (j, b) in enumerate(candidates):
        thr = _thresholds(meta, int(j))[int(b)]
        left = (X[:, int(j)] <= thr).astype(np.float64)
        scale = 1.0 if gain_weight is None else float(gain_weight[idx])
        cols.append(scale * left)
        if include_grad:
            cols.append(scale * left * (g / g_scale))
        if include_hess:
            cols.append(scale * left * (h / h_scale))
    # Add teacher confidence and raw standardized features so the coreset remains useful downstream.
    if include_conf and teacher is not None and hasattr(teacher, "predict_proba"):
        proba = teacher.predict_proba(X)
        if proba.ndim == 2:
            cols.append(np.asarray(proba[:, 1], dtype=np.float64))
    if include_raw:
        raw = StandardScaler().fit_transform(X).astype(np.float64)
        max_raw = min(32, raw.shape[1])
        for j in range(max_raw):
            cols.append(raw[:, j])
    if include_path and teacher is not None and hasattr(teacher, "apply"):
        try:
            leaves = np.asarray(teacher.apply(X))
            if leaves.ndim == 1:
                leaves = leaves[:, None]
            for t in range(min(16, leaves.shape[1])):
                values, counts = np.unique(leaves[:, t], return_counts=True)
                top = values[np.argsort(-counts)[:8]]
                for val in top:
                    cols.append((leaves[:, t] == val).astype(np.float64))
        except Exception:
            pass
    S = np.vstack(cols).T.astype(np.float64)
    scale = np.maximum(np.std(S, axis=0), 1e-6)
    return S, scale


def _normalize_sketch(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.maximum(np.std(S, axis=0), 1e-6)
    return S / scale[None, :], scale


def _select_from_sketch(S_match: np.ndarray, y: np.ndarray, budget: int, refine: bool = True) -> list[int]:
    selected: list[int] = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        selected.extend(_greedy_mean_match(S_match, idx, min(budget, len(idx))))
    if refine:
        selected = _refine_selection(S_match, y, selected, rounds=2)
    return selected


def _append_probe_features(
    cols: list[np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    meta: dict[str, Any],
    landscape: Any | None,
    g: np.ndarray,
    h: np.ndarray,
    max_probes: int = 8,
    max_candidates_per_probe: int = 10,
) -> None:
    if landscape is None or not getattr(landscape, "probes", None):
        return
    g_scale = max(float(np.std(g)), 1e-3)
    h_scale = max(float(np.std(h)), 1e-3)
    probes = sorted(getattr(landscape, "probes", []), key=lambda p: float(np.max(p.gains)) if len(p.gains) else 0.0, reverse=True)
    for probe in probes[:max_probes]:
        region = route_mask(X, probe.path).astype(np.float64)
        if float(region.sum()) < 2.0:
            continue
        gains = np.asarray(probe.gains, dtype=np.float64)
        order = np.argsort(-gains)[: min(max_candidates_per_probe, len(gains))]
        base = np.sqrt(np.maximum(0.0, gains - np.min(gains) + 1e-6))
        base = base / max(float(np.mean(base)), 1e-6)
        cols.append(region)
        for idx in order:
            j, b = probe.candidates[int(idx)]
            thr = _thresholds(meta, int(j))[int(b)]
            left = region * (X[:, int(j)] <= thr).astype(np.float64)
            scale = float(base[int(idx)])
            cols.append(scale * left)
            cols.append(scale * left * (g / g_scale))
            cols.append(scale * left * (h / h_scale))


def _trajectory_sketch(
    X: np.ndarray,
    y: np.ndarray,
    meta: dict[str, Any],
    teacher: Any | None,
    landscape: Any | None,
    include_raw: bool = True,
    include_path: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    S, _ = _gain_sketch(X, y, meta, teacher, landscape, include_raw=include_raw, include_path=include_path)
    cols = [S[:, j].astype(np.float64) for j in range(S.shape[1])]
    if teacher is not None:
        g, h = compute_grad_hess(teacher, X, y)
    else:
        p = np.full(len(y), float(np.mean(y)))
        g = p - y
        h = np.maximum(p * (1.0 - p), 1e-6)
    _append_probe_features(cols, X, y, meta, landscape, g, h)
    S2 = np.vstack(cols).T.astype(np.float64)
    return _normalize_sketch(S2)


def _margin_sketch(
    X: np.ndarray,
    y: np.ndarray,
    meta: dict[str, Any],
    teacher: Any | None,
    landscape: Any | None,
    include_path: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    S, _ = _gain_sketch(X, y, meta, teacher, landscape, include_path=include_path)
    cols = [S[:, j].astype(np.float64) for j in range(S.shape[1])]
    if landscape is not None and len(getattr(landscape, "candidates", [])) > 1:
        candidates = np.asarray(landscape.candidates, dtype=np.int64)
        gains = np.asarray(landscape.gains, dtype=np.float64)
        order = np.argsort(-gains)
        best = int(order[0])
        j_best, b_best = candidates[best]
        left_best = (X[:, int(j_best)] <= _thresholds(meta, int(j_best))[int(b_best)]).astype(np.float64)
        spread = max(float(np.std(gains)), 1e-6)
        for idx in order[1 : min(17, len(order))]:
            j, b = candidates[int(idx)]
            left = (X[:, int(j)] <= _thresholds(meta, int(j))[int(b)]).astype(np.float64)
            closeness = np.exp(-max(0.0, float(gains[best] - gains[int(idx)])) / spread)
            cols.append(closeness * (left_best - left))
    S2 = np.vstack(cols).T.astype(np.float64)
    return _normalize_sketch(S2)


def _greedy_mean_match(S: np.ndarray, idx: np.ndarray, budget: int) -> list[int]:
    if len(idx) <= budget:
        return idx.tolist()
    S_cls = S[idx]
    target = S_cls.mean(axis=0)
    selected_local: list[int] = []
    chosen = np.zeros(len(idx), dtype=bool)
    running = np.zeros(S.shape[1], dtype=np.float64)
    for t in range(budget):
        desired = (t + 1) * target - running
        scores = np.einsum("ij,ij->i", S_cls - desired, S_cls - desired)
        scores[chosen] = np.inf
        loc = int(np.argmin(scores))
        selected_local.append(loc)
        chosen[loc] = True
        running += S_cls[loc]
    return idx[selected_local].tolist()


def _fit_weights(S: np.ndarray, scale: np.ndarray, y: np.ndarray, selected: list[int]) -> np.ndarray:
    if not selected:
        return np.empty(0, dtype=np.float32)
    S_scaled = S / scale[None, :]
    A = S_scaled[selected].T
    b = S_scaled.sum(axis=0)
    class_rows = []
    class_targets = []
    constraint_weight = np.sqrt(S.shape[1])
    for c in np.unique(y):
        class_rows.append((y[selected] == c).astype(np.float64) * constraint_weight)
        class_targets.append(float(np.sum(y == c)) * constraint_weight)
    A_aug = np.vstack([A, np.ones((1, len(selected))) * constraint_weight, np.vstack(class_rows)])
    b_aug = np.concatenate([b, [len(y) * constraint_weight], np.asarray(class_targets, dtype=np.float64)])
    try:
        res = lsq_linear(A_aug, b_aug, bounds=(0.0, np.inf), max_iter=300, lsmr_tol="auto")
        w = np.maximum(res.x, 1e-8)
    except Exception:
        w = np.ones(len(selected), dtype=np.float64)
    w = w * (len(y) / max(1e-8, float(np.sum(w))))
    return w.astype(np.float32)


def _state_list(landscape: Any | None, max_probes: int = 8) -> list[dict[str, Any]]:
    if landscape is None or len(getattr(landscape, "candidates", [])) == 0:
        return []
    root = {
        "candidates": np.asarray(landscape.candidates, dtype=np.int64),
        "gains": np.asarray(landscape.gains, dtype=np.float64),
        "hist": np.asarray(landscape.hist, dtype=np.float64),
        "parent": np.asarray(landscape.parent, dtype=np.float64),
        "best_index": int(landscape.best_index),
        "checkpoint": 0,
        "path": [],
        "depth": 0,
    }
    probes = []
    for probe in getattr(landscape, "probes", []) or []:
        gains = np.asarray(probe.gains, dtype=np.float64)
        probes.append(
            {
                "candidates": np.asarray(probe.candidates, dtype=np.int64),
                "gains": gains,
                "hist": np.asarray(probe.hist, dtype=np.float64),
                "parent": np.asarray(probe.parent, dtype=np.float64),
                "best_index": int(probe.best_index),
                "checkpoint": int(probe.checkpoint),
                "path": list(probe.path),
                "depth": int(probe.depth),
                "priority": float(np.max(gains)) if len(gains) else 0.0,
            }
        )
    probes = sorted(probes, key=lambda p: (int(p["depth"]), -float(p["priority"])))[:max_probes]
    return [root] + probes


def _checkpoint_grad_hess(
    X: np.ndarray,
    y: np.ndarray,
    teacher: Any | None,
    checkpoint: int,
) -> tuple[np.ndarray, np.ndarray]:
    if teacher is None:
        p = np.clip(float(np.mean(y)), 1e-6, 1.0 - 1e-6)
        g = np.full(len(y), p, dtype=np.float64) - y
        h = np.full(len(y), max(p * (1.0 - p), 1e-6), dtype=np.float64)
        return g, h
    margin = raw_margin_at_checkpoint(teacher, X, y, int(checkpoint))
    return grad_hess_from_margin(margin, y)


def _histdistill_matrices(
    X: np.ndarray,
    y: np.ndarray,
    meta: dict[str, Any],
    teacher: Any | None,
    landscape: Any | None,
    max_probes: int = 8,
    max_candidates_per_state: int = 24,
    include_density: bool = False,
    root_only: bool = False,
    linear_coverage: bool = False,
    density_weight: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = _state_list(landscape, max_probes=0 if root_only else max_probes)
    if not states:
        S, scale = _gain_sketch(X, y, meta, teacher, landscape, include_path=not root_only)
        target = S.sum(axis=0)
        caps = np.maximum(np.abs(target), 1.0)
        omega = np.ones(S.shape[1], dtype=np.float64)
        return np.abs(S), caps, omega, S, target

    coverage_cols: list[np.ndarray] = []
    coverage_caps: list[float] = []
    coverage_weights: list[float] = []
    design_cols: list[np.ndarray] = []
    design_targets: list[float] = []

    for state in states:
        candidates = np.asarray(state["candidates"], dtype=np.int64)
        gains = np.asarray(state["gains"], dtype=np.float64)
        hist = np.asarray(state["hist"], dtype=np.float64)
        parent = np.asarray(state["parent"], dtype=np.float64)
        if len(candidates) == 0:
            continue
        order = np.argsort(-gains)[: min(max_candidates_per_state, len(gains))]
        region = route_mask(X, state["path"]).astype(np.float64)
        if float(region.sum()) < 1.0:
            continue
        g, h = _checkpoint_grad_hess(X, y, teacher, int(state["checkpoint"]))
        mass = region * (np.abs(g) + h)
        gain_shift = np.maximum(0.0, gains - float(np.min(gains)) + 1e-8)
        gain_scale = np.sqrt(gain_shift)
        gain_scale = gain_scale / max(float(np.mean(gain_scale[order])), 1e-8)

        design_cols.extend([region, region * g, region * h])
        design_targets.extend([float(parent[2]), float(parent[0]), float(parent[1])])

        for idx in order:
            j, b = candidates[int(idx)]
            thr = _thresholds(meta, int(j))[int(b)]
            left = region * (X[:, int(j)] <= thr).astype(np.float64)
            right = region - left
            left_mass = left * (np.abs(g) + h)
            right_mass = right * (np.abs(g) + h)
            weight = float(gain_scale[int(idx)])

            if linear_coverage:
                # Linear coverage is an ablation of the saturating cap objective.
                cap_l = max(float(left_mass.sum()), 1.0)
                cap_r = max(float(right_mass.sum()), 1.0)
            else:
                cap_l = max(float(left_mass.sum()), 1e-8)
                cap_r = max(float(right_mass.sum()), 1e-8)
            coverage_cols.extend([left_mass, right_mass])
            coverage_caps.extend([cap_l, cap_r])
            coverage_weights.extend([weight, weight])

            left_target = hist[int(idx)]
            right_target = np.array(
                [
                    float(parent[2] - left_target[0]),
                    float(parent[0] - left_target[1]),
                    float(parent[1] - left_target[2]),
                ],
                dtype=np.float64,
            )
            design_cols.extend([left, left * g, left * h, right, right * g, right * h])
            design_targets.extend(
                [
                    float(left_target[0]),
                    float(left_target[1]),
                    float(left_target[2]),
                    float(right_target[0]),
                    float(right_target[1]),
                    float(right_target[2]),
                ]
            )

    if include_density:
        raw = StandardScaler().fit_transform(X).astype(np.float64)
        max_raw = min(32, raw.shape[1])
        for j in range(max_raw):
            col = raw[:, j]
            shifted = col - float(np.min(col))
            coverage_cols.append(density_weight * shifted)
            coverage_caps.append(max(float(density_weight * np.sum(shifted)), 1e-8))
            coverage_weights.append(0.5)
            design_cols.append(density_weight * col)
            design_targets.append(float(density_weight * np.sum(col)))

    coverage = np.vstack(coverage_cols).T.astype(np.float64)
    caps = np.asarray(coverage_caps, dtype=np.float64)
    omega = np.asarray(coverage_weights, dtype=np.float64)
    design = np.vstack(design_cols).T.astype(np.float64)
    target = np.asarray(design_targets, dtype=np.float64)

    cscale = np.maximum(np.percentile(np.abs(coverage), 90, axis=0), 1e-8)
    coverage = coverage / cscale[None, :]
    caps = caps / cscale
    dscale = np.maximum(np.abs(target), np.percentile(np.abs(design), 90, axis=0) * max(1, len(y)))
    dscale = np.maximum(dscale, 1.0)
    design = design / dscale[None, :]
    target = target / dscale
    return coverage, caps, omega, design, target


def _histdistill_select(
    coverage: np.ndarray,
    caps: np.ndarray,
    omega: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    selected_mask = np.zeros(len(y), dtype=bool)
    covered = np.zeros(coverage.shape[1], dtype=np.float64)
    quotas = {int(c): min(budget, int(np.sum(y == c))) for c in np.unique(y)}
    remaining_quota = dict(quotas)
    classes = list(remaining_quota)
    while any(v > 0 for v in remaining_quota.values()):
        progressed = False
        for c in classes:
            if remaining_quota[c] <= 0:
                continue
            pool = np.flatnonzero((y == c) & (~selected_mask))
            if len(pool) == 0:
                remaining_quota[c] = 0
                continue
            rem = np.maximum(caps - covered, 0.0) * omega
            clipped = np.minimum(coverage[pool], np.maximum(caps - covered, 0.0)[None, :])
            scores = clipped @ omega
            if not np.isfinite(scores).any() or float(np.max(scores)) <= 1e-12:
                loc = int(rng.integers(0, len(pool)))
            else:
                loc = int(np.argmax(scores))
            chosen = int(pool[loc])
            selected.append(chosen)
            selected_mask[chosen] = True
            covered = np.minimum(caps, covered + coverage[chosen])
            remaining_quota[c] -= 1
            progressed = True
        if not progressed:
            break
    return selected


def _stabilize_weights(w: np.ndarray, total: float, clip_ratio: float = 8.0) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    w = np.maximum(w, 1e-8)
    w = w * (total / max(1e-8, float(np.sum(w))))
    mean = max(float(total) / max(1, len(w)), 1e-8)
    w = np.clip(w, 0.05 * mean, clip_ratio * mean)
    w = w * (total / max(1e-8, float(np.sum(w))))
    return w


def _histdistill_fit_weights(design: np.ndarray, target: np.ndarray, y: np.ndarray, selected: list[int]) -> np.ndarray:
    if not selected:
        return np.empty(0, dtype=np.float32)
    selected_arr = np.asarray(selected, dtype=int)
    A = design[selected_arr].T
    b = target.astype(np.float64)
    constraint_weight = max(1.0, np.sqrt(design.shape[1]))
    class_rows = []
    class_targets = []
    for c in np.unique(y):
        class_rows.append((y[selected_arr] == c).astype(np.float64) * constraint_weight)
        class_targets.append(float(np.sum(y == c)) * constraint_weight)
    A_aug = np.vstack([A, np.ones((1, len(selected))) * constraint_weight, np.vstack(class_rows)])
    b_aug = np.concatenate([b, [len(y) * constraint_weight], np.asarray(class_targets, dtype=np.float64)])
    try:
        res = lsq_linear(A_aug, b_aug, bounds=(0.0, np.inf), max_iter=400, lsmr_tol="auto")
        w = np.maximum(res.x, 1e-8)
    except Exception:
        w = np.ones(len(selected), dtype=np.float64)
    # Downstream learners are sensitive to full-dataset-mass weights because lambda
    # and tree stopping rules live on the sample-weight scale. We fit relative
    # histogram weights, then normalize to mean one for materialized training.
    w = _stabilize_weights(w, total=float(len(selected)), clip_ratio=8.0)
    return w.astype(np.float32)


def _histdistill_refine(
    design: np.ndarray,
    target: np.ndarray,
    y: np.ndarray,
    selected: list[int],
    rounds: int = 1,
    pool_size: int = 160,
) -> list[int]:
    if not selected:
        return selected
    selected_set = set(selected)
    refined: list[int] = []
    for c in np.unique(y):
        current = [i for i in selected if y[i] == c]
        cls = np.flatnonzero(y == c)
        if not current:
            continue
        target_share = target * (len(current) / max(1, len(selected)))
        current_sum = design[current].sum(axis=0)
        remaining = np.asarray([int(i) for i in cls if int(i) not in selected_set], dtype=int)
        if len(remaining) == 0:
            refined.extend(current)
            continue
        dist = np.einsum("ij,ij->i", design[remaining] - target_share / max(1, len(current)), design[remaining] - target_share / max(1, len(current)))
        pool = remaining[np.argsort(dist)[: min(pool_size, len(remaining))]]
        cur_obj = float(np.sum((current_sum - target_share) ** 2))
        for _ in range(rounds):
            improved = False
            for pos, old in enumerate(list(current)):
                base = current_sum - design[old]
                cand = base[None, :] + design[pool]
                obj = np.einsum("ij,ij->i", cand - target_share, cand - target_share)
                best = int(np.argmin(obj))
                if float(obj[best]) + 1e-12 < cur_obj:
                    new = int(pool[best])
                    selected_set.discard(old)
                    selected_set.add(new)
                    current[pos] = new
                    current_sum = base + design[new]
                    cur_obj = float(obj[best])
                    improved = True
            if not improved:
                break
        refined.extend(current)
    return refined


def _histdistill(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None,
    landscape: Any | None,
    fit_weights: bool = True,
    refine: bool = False,
    include_density: bool = False,
    root_only: bool = False,
    linear_coverage: bool = False,
    warm_start_gain_path: bool = False,
    weight_blend: float = 1.0,
    density_weight: float = 0.35,
) -> dict[str, np.ndarray]:
    coverage, caps, omega, design, target = _histdistill_matrices(
        X,
        y,
        meta,
        teacher,
        landscape,
        include_density=include_density,
        root_only=root_only,
        linear_coverage=linear_coverage,
        density_weight=density_weight,
    )
    if warm_start_gain_path:
        selected, _, _, S_match = _select_by_gain_sketch(X, y, budget, meta, teacher, landscape, include_path=True)
        selected = _refine_selection(S_match, y, selected, rounds=1)
    else:
        selected = _histdistill_select(coverage, caps, omega, y, budget, seed)
    if refine and not warm_start_gain_path:
        selected = _histdistill_refine(design, target, y, selected, rounds=1)
    if fit_weights:
        weights = _histdistill_fit_weights(design, target, y, selected)
        blend = float(np.clip(weight_blend, 0.0, 1.0))
        if blend < 1.0:
            weights = (blend * weights + (1.0 - blend) * np.ones(len(selected), dtype=np.float32)).astype(np.float32)
            weights = (weights * (len(selected) / max(1e-8, float(np.sum(weights))))).astype(np.float32)
    else:
        weights = np.ones(len(selected), dtype=np.float32)
    return {"X": X[selected].astype(np.float32), "y": y[selected].astype(np.int64), "weights": weights}


def histdistill_greedy(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    return _histdistill(X, y, budget, seed, meta, teacher, landscape, fit_weights=True, refine=False)


def histdistill_refined(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    return _histdistill(
        X,
        y,
        budget,
        seed,
        meta,
        teacher,
        landscape,
        fit_weights=True,
        refine=True,
        warm_start_gain_path=True,
        weight_blend=0.25,
    )


def histdistill_density(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    return _histdistill(
        X,
        y,
        budget,
        seed,
        meta,
        teacher,
        landscape,
        fit_weights=True,
        refine=True,
        include_density=True,
        warm_start_gain_path=True,
        weight_blend=0.25,
        density_weight=0.35,
    )


def histdistill_density_weak(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    return _histdistill(
        X,
        y,
        budget,
        seed,
        meta,
        teacher,
        landscape,
        fit_weights=True,
        refine=True,
        include_density=True,
        warm_start_gain_path=True,
        weight_blend=0.25,
        density_weight=0.15,
    )


def histdistill_density_mid(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    return _histdistill(
        X,
        y,
        budget,
        seed,
        meta,
        teacher,
        landscape,
        fit_weights=True,
        refine=True,
        include_density=True,
        warm_start_gain_path=True,
        weight_blend=0.25,
        density_weight=0.35,
    )


def histdistill_density_strong(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    return _histdistill(
        X,
        y,
        budget,
        seed,
        meta,
        teacher,
        landscape,
        fit_weights=True,
        refine=True,
        include_density=True,
        warm_start_gain_path=True,
        weight_blend=0.25,
        density_weight=0.70,
    )


def histdistill_density_xstrong(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    return _histdistill(
        X,
        y,
        budget,
        seed,
        meta,
        teacher,
        landscape,
        fit_weights=True,
        refine=True,
        include_density=True,
        warm_start_gain_path=True,
        weight_blend=0.25,
        density_weight=1.25,
    )


def histdistill_importance(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    del meta, landscape
    rng = np.random.default_rng(seed)
    if teacher is not None:
        g, h = compute_grad_hess(teacher, X, y)
        score = np.abs(g) + h
    else:
        score = np.ones(len(y), dtype=np.float64)
    selected: list[int] = []
    weights: list[float] = []
    for c in np.unique(y):
        cls = np.flatnonzero(y == c)
        n_take = min(budget, len(cls))
        p = np.asarray(score[cls], dtype=np.float64)
        p = np.maximum(p, 1e-8)
        p = p / max(float(p.sum()), 1e-8)
        replace = n_take > len(cls)
        loc = rng.choice(len(cls), size=n_take, replace=replace, p=p)
        selected.extend(cls[loc].tolist())
        weights.extend((1.0 / (max(1, n_take) * p[loc])).tolist())
    w = np.asarray(weights, dtype=np.float64)
    w = _stabilize_weights(w, total=float(len(selected)), clip_ratio=8.0)
    return {"X": X[selected].astype(np.float32), "y": y[selected].astype(np.int64), "weights": w.astype(np.float32)}


def _teacher_scores(X: np.ndarray, y: np.ndarray, teacher: Any | None, kind: str = "mvs") -> np.ndarray:
    if teacher is None:
        return np.ones(len(y), dtype=np.float64)
    g, h = compute_grad_hess(teacher, X, y)
    if kind == "grand":
        score = np.abs(g)
    elif kind == "el2n":
        score = np.abs(g)
    elif kind == "goss":
        score = np.abs(g)
    else:
        score = np.sqrt(g * g + h * h)
    return np.maximum(np.asarray(score, dtype=np.float64), 1e-8)


def _probability_sample_per_class(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    score: np.ndarray,
    clip_ratio: float = 8.0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    weights: list[float] = []
    for c in np.unique(y):
        cls = np.flatnonzero(y == c)
        n_take = min(int(budget), len(cls))
        if n_take <= 0:
            continue
        p = np.asarray(score[cls], dtype=np.float64)
        p = np.maximum(p, 1e-8)
        p = p / max(float(p.sum()), 1e-8)
        loc = rng.choice(len(cls), size=n_take, replace=False, p=p)
        selected.extend(cls[loc].tolist())
        weights.extend((1.0 / (max(1, n_take) * p[loc])).tolist())
    w = _stabilize_weights(np.asarray(weights, dtype=np.float64), total=float(len(selected)), clip_ratio=clip_ratio)
    return {"X": X[selected].astype(np.float32), "y": y[selected].astype(np.int64), "weights": w.astype(np.float32)}


def goss_coreset(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    teacher: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    score = _teacher_scores(X, y, teacher, kind="goss")
    selected: list[int] = []
    weights: list[float] = []
    for c in np.unique(y):
        cls = np.flatnonzero(y == c)
        n_take = min(int(budget), len(cls))
        if n_take <= 0:
            continue
        cls_score = score[cls]
        keep_hi = min(max(1, int(np.ceil(0.4 * n_take))), n_take)
        order = np.argsort(-cls_score)
        hi = order[:keep_hi]
        remaining = order[keep_hi:]
        selected.extend(cls[hi].tolist())
        weights.extend([1.0] * len(hi))
        low_take = n_take - len(hi)
        if low_take > 0 and len(remaining) > 0:
            probs = np.ones(len(remaining), dtype=np.float64) / len(remaining)
            loc = rng.choice(len(remaining), size=min(low_take, len(remaining)), replace=False, p=probs)
            chosen = remaining[loc]
            selected.extend(cls[chosen].tolist())
            weights.extend([len(remaining) / max(1, len(chosen))] * len(chosen))
    w = _stabilize_weights(np.asarray(weights, dtype=np.float64), total=float(len(selected)), clip_ratio=8.0)
    return {"X": X[selected].astype(np.float32), "y": y[selected].astype(np.int64), "weights": w.astype(np.float32)}


def mvs_coreset(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    teacher: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    score = _teacher_scores(X, y, teacher, kind="mvs")
    return _probability_sample_per_class(X, y, budget, seed, score, clip_ratio=8.0)


def grand_coreset(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    teacher: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    score = _teacher_scores(X, y, teacher, kind="grand")
    return _probability_sample_per_class(X, y, budget, seed, score, clip_ratio=8.0)


def el2n_coreset(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    teacher: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    score = _teacher_scores(X, y, teacher, kind="el2n")
    return _probability_sample_per_class(X, y, budget, seed, score, clip_ratio=8.0)


def _gradient_feature_matrix(X: np.ndarray, y: np.ndarray, teacher: Any | None) -> np.ndarray:
    raw = StandardScaler().fit_transform(X).astype(np.float64)
    cols = [raw[:, : min(32, raw.shape[1])]]
    if teacher is not None:
        g, h = compute_grad_hess(teacher, X, y)
        cols.append(g[:, None])
        cols.append(h[:, None])
        cols.append((np.abs(g) + h)[:, None])
        if hasattr(teacher, "predict_proba"):
            p = teacher.predict_proba(X)
            if getattr(p, "ndim", 1) == 2:
                cols.append(np.asarray(p[:, [1]], dtype=np.float64))
    return StandardScaler().fit_transform(np.hstack(cols)).astype(np.float64)


def craig_coreset(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    teacher: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    S = _gradient_feature_matrix(X, y, teacher)
    selected: list[int] = []
    weights: list[float] = []
    for c in np.unique(y):
        cls = np.flatnonzero(y == c)
        k = min(int(budget), len(cls))
        if k <= 0:
            continue
        first = int(rng.choice(cls))
        chosen = [first]
        dmin = pairwise_distances(S[cls], S[[first]]).ravel()
        for _ in range(1, k):
            nxt = int(cls[np.argmax(dmin)])
            chosen.append(nxt)
            dmin = np.minimum(dmin, pairwise_distances(S[cls], S[[nxt]]).ravel())
        nearest = np.argmin(pairwise_distances(S[cls], S[chosen]), axis=1)
        counts = np.bincount(nearest, minlength=len(chosen)).astype(np.float64)
        selected.extend(chosen)
        weights.extend(counts.tolist())
    w = _stabilize_weights(np.asarray(weights, dtype=np.float64), total=float(len(selected)), clip_ratio=8.0)
    return {"X": X[selected].astype(np.float32), "y": y[selected].astype(np.int64), "weights": w.astype(np.float32)}


def gradmatch_coreset(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    teacher: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    del seed
    S = _gradient_feature_matrix(X, y, teacher)
    selected: list[int] = []
    for c in np.unique(y):
        cls = np.flatnonzero(y == c)
        selected.extend(_greedy_mean_match(S, cls, min(int(budget), len(cls))))
    weights = _fit_weights(S, np.maximum(np.std(S, axis=0), 1e-6), y, selected)
    weights = _stabilize_weights(weights, total=float(len(selected)), clip_ratio=8.0).astype(np.float32)
    return {"X": X[selected].astype(np.float32), "y": y[selected].astype(np.int64), "weights": weights}


def histdistill_no_weight_fit(X: np.ndarray, y: np.ndarray, budget: int, seed: int, **kwargs: Any) -> dict[str, np.ndarray]:
    return _histdistill(X, y, budget, seed, fit_weights=False, refine=True, **kwargs)


def histdistill_linear_coverage(X: np.ndarray, y: np.ndarray, budget: int, seed: int, **kwargs: Any) -> dict[str, np.ndarray]:
    return _histdistill(X, y, budget, seed, fit_weights=True, refine=True, linear_coverage=True, **kwargs)


def histdistill_root_only(X: np.ndarray, y: np.ndarray, budget: int, seed: int, **kwargs: Any) -> dict[str, np.ndarray]:
    return _histdistill(X, y, budget, seed, fit_weights=True, refine=True, root_only=True, **kwargs)


def histdistill_no_refine(X: np.ndarray, y: np.ndarray, budget: int, seed: int, **kwargs: Any) -> dict[str, np.ndarray]:
    return _histdistill(X, y, budget, seed, fit_weights=True, refine=False, **kwargs)


def gain_herding(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    weighted: bool = False,
    include_grad: bool = True,
    include_hess: bool = True,
    include_conf: bool = True,
    include_raw: bool = True,
    include_gain_weight: bool = True,
    include_path: bool = False,
    **_: Any,
) -> dict[str, np.ndarray]:
    del seed
    S, scale = _gain_sketch(
        X,
        y,
        meta,
        teacher,
        landscape,
        include_grad=include_grad,
        include_hess=include_hess,
        include_conf=include_conf,
        include_raw=include_raw,
        include_gain_weight=include_gain_weight,
        include_path=include_path,
    )
    S_match = S / scale[None, :]
    selected: list[int] = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        selected.extend(_greedy_mean_match(S_match, idx, min(budget, len(idx))))
    weights = _fit_weights(S, scale, y, selected) if weighted else np.ones(len(selected), dtype=np.float32)
    return {"X": X[selected].astype(np.float32), "y": y[selected].astype(np.int64), "weights": weights}


def _refine_selection(S: np.ndarray, y: np.ndarray, selected: list[int], rounds: int = 2, pool_size: int = 256) -> list[int]:
    selected_set = set(selected)
    refined: list[int] = []
    for c in np.unique(y):
        cls = np.flatnonzero(y == c)
        sel = [i for i in selected if y[i] == c]
        if not sel:
            continue
        target = S[cls].mean(axis=0)
        current = list(sel)
        current_sum = S[current].sum(axis=0)
        remaining = np.array([i for i in cls if i not in selected_set], dtype=int)
        if len(remaining) == 0:
            refined.extend(current)
            continue
        # Candidate pool: rows close to the class sketch target.
        dist = np.einsum("ij,ij->i", S[remaining] - target, S[remaining] - target)
        pool = remaining[np.argsort(dist)[: min(pool_size, len(remaining))]]
        cur_obj = float(np.sum((current_sum / len(current) - target) ** 2))
        for _ in range(rounds):
            improved = False
            for pos, old in enumerate(list(current)):
                base_sum = current_sum - S[old]
                cand_mean = (base_sum[None, :] + S[pool]) / len(current)
                cand_obj = np.einsum("ij,ij->i", cand_mean - target, cand_mean - target)
                best_loc = int(np.argmin(cand_obj))
                if float(cand_obj[best_loc]) + 1e-12 < cur_obj:
                    new = int(pool[best_loc])
                    selected_set.discard(old)
                    selected_set.add(new)
                    current[pos] = new
                    current_sum = base_sum + S[new]
                    cur_obj = float(cand_obj[best_loc])
                    improved = True
            if not improved:
                break
        refined.extend(current)
    return refined


def gain_weighted_coreset(X: np.ndarray, y: np.ndarray, budget: int, seed: int, **kwargs: Any) -> dict[str, np.ndarray]:
    return gain_herding(X, y, budget, seed, weighted=True, **kwargs)


def gaindistill_hybrid(X: np.ndarray, y: np.ndarray, budget: int, seed: int, **kwargs: Any) -> dict[str, np.ndarray]:
    return gain_herding(X, y, budget, seed, weighted=True, **kwargs)


def gain_path_herding(X: np.ndarray, y: np.ndarray, budget: int, seed: int, **kwargs: Any) -> dict[str, np.ndarray]:
    return gain_herding(X, y, budget, seed, include_path=True, **kwargs)


def gaindistill_path_hybrid(X: np.ndarray, y: np.ndarray, budget: int, seed: int, **kwargs: Any) -> dict[str, np.ndarray]:
    return gain_herding(X, y, budget, seed, weighted=True, include_path=True, **kwargs)


def _select_by_gain_sketch(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    meta: dict[str, Any],
    teacher: Any | None,
    landscape: Any | None,
    include_path: bool,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    S, scale = _gain_sketch(X, y, meta, teacher, landscape, include_path=include_path)
    S_match = S / scale[None, :]
    selected: list[int] = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        selected.extend(_greedy_mean_match(S_match, idx, min(budget, len(idx))))
    return selected, S, scale, S_match


def _class_prior_weights(y: np.ndarray, selected: list[int]) -> np.ndarray:
    weights = np.ones(len(selected), dtype=np.float64)
    selected_arr = np.asarray(selected, dtype=int)
    total = max(1, len(y))
    kept = max(1, len(selected))
    for c in np.unique(y):
        cls_count = max(1, int(np.sum(y == c)))
        sel_mask = y[selected_arr] == c
        sel_count = max(1, int(sel_mask.sum()))
        weights[sel_mask] = (cls_count / total) / (sel_count / kept)
    weights = weights / max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


def _fill_with_raw_kcenter(Xs: np.ndarray, cls: np.ndarray, chosen: list[int], target_size: int) -> list[int]:
    current = list(chosen)
    if len(current) >= target_size:
        return current[:target_size]
    chosen_set = set(current)
    remaining = np.asarray([int(i) for i in cls if int(i) not in chosen_set], dtype=int)
    if len(remaining) == 0:
        return current
    if current:
        dmin = pairwise_distances(Xs[remaining], Xs[current]).min(axis=1)
    else:
        center = Xs[cls].mean(axis=0, keepdims=True)
        dmin = pairwise_distances(Xs[remaining], center).ravel()
    while len(current) < target_size and len(remaining) > 0:
        loc = int(np.argmax(dmin))
        nxt = int(remaining[loc])
        current.append(nxt)
        keep = np.arange(len(remaining)) != loc
        remaining = remaining[keep]
        if len(remaining) == 0:
            break
        dmin = dmin[keep]
        dmin = np.minimum(dmin, pairwise_distances(Xs[remaining], Xs[[nxt]]).ravel())
    return current


def gain_path_prior_weighted(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    del seed
    selected, _, _, _ = _select_by_gain_sketch(X, y, budget, meta, teacher, landscape, include_path=True)
    weights = _class_prior_weights(y, selected)
    return {"X": X[selected].astype(np.float32), "y": y[selected].astype(np.int64), "weights": weights}


def gain_path_diverse(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    del seed
    core_budget = max(1, int(round(0.70 * budget)))
    selected, _, _, _ = _select_by_gain_sketch(X, y, core_budget, meta, teacher, landscape, include_path=True)
    Xs = StandardScaler().fit_transform(X)
    final: list[int] = []
    for c in np.unique(y):
        cls = np.flatnonzero(y == c)
        chosen = [i for i in selected if y[i] == c]
        final.extend(_fill_with_raw_kcenter(Xs, cls, chosen, min(budget, len(cls))))
    return {"X": X[final].astype(np.float32), "y": y[final].astype(np.int64), "weights": np.ones(len(final), dtype=np.float32)}


def gain_path_dual(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    del seed
    path_budget = max(1, int(round(0.65 * budget)))
    path_selected, _, _, _ = _select_by_gain_sketch(X, y, path_budget, meta, teacher, landscape, include_path=True)
    root_order, _, _, _ = _select_by_gain_sketch(X, y, budget, meta, teacher, landscape, include_path=False)
    final: list[int] = []
    for c in np.unique(y):
        cls = np.flatnonzero(y == c)
        chosen = [i for i in path_selected if y[i] == c]
        seen = set(chosen)
        for idx in root_order:
            if y[idx] == c and idx not in seen:
                chosen.append(idx)
                seen.add(idx)
            if len(chosen) >= min(budget, len(cls)):
                break
        if len(chosen) < min(budget, len(cls)):
            for idx in cls:
                if int(idx) not in seen:
                    chosen.append(int(idx))
                if len(chosen) >= min(budget, len(cls)):
                    break
        final.extend(chosen[: min(budget, len(cls))])
    return {"X": X[final].astype(np.float32), "y": y[final].astype(np.int64), "weights": np.ones(len(final), dtype=np.float32)}


def gain_path_refined(X: np.ndarray, y: np.ndarray, budget: int, seed: int, meta: dict[str, Any], teacher: Any | None = None, landscape: Any | None = None, **_: Any) -> dict[str, np.ndarray]:
    selected, _, _, S_match = _select_by_gain_sketch(X, y, budget, meta, teacher, landscape, include_path=True)
    selected = _refine_selection(S_match, y, selected, rounds=2)
    return {"X": X[selected].astype(np.float32), "y": y[selected].astype(np.int64), "weights": np.ones(len(selected), dtype=np.float32)}


def gain_path_safeguarded(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    del seed
    selected, _, _, S_match = _select_by_gain_sketch(X, y, budget, meta, teacher, landscape, include_path=True)
    if X.shape[1] <= budget:
        selected = _refine_selection(S_match, y, selected, rounds=2)
    return {"X": X[selected].astype(np.float32), "y": y[selected].astype(np.int64), "weights": np.ones(len(selected), dtype=np.float32)}


def _gain_refined_variant(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    meta: dict[str, Any],
    teacher: Any | None,
    landscape: Any | None,
    include_grad: bool = True,
    include_hess: bool = True,
    include_conf: bool = True,
    include_raw: bool = True,
    include_gain_weight: bool = True,
    include_path: bool = True,
    refine: bool = True,
) -> dict[str, np.ndarray]:
    S, scale = _gain_sketch(
        X,
        y,
        meta,
        teacher,
        landscape,
        include_grad=include_grad,
        include_hess=include_hess,
        include_conf=include_conf,
        include_raw=include_raw,
        include_gain_weight=include_gain_weight,
        include_path=include_path,
    )
    S_match = S / scale[None, :]
    selected: list[int] = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        selected.extend(_greedy_mean_match(S_match, idx, min(budget, len(idx))))
    if refine:
        selected = _refine_selection(S_match, y, selected, rounds=2)
    return {"X": X[selected].astype(np.float32), "y": y[selected].astype(np.int64), "weights": np.ones(len(selected), dtype=np.float32)}


def refined_no_path(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    del seed
    return _gain_refined_variant(X, y, budget, meta, teacher, landscape, include_path=False)


def refined_no_refine(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    del seed
    return _gain_refined_variant(X, y, budget, meta, teacher, landscape, refine=False)


def refined_no_grad(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    del seed
    return _gain_refined_variant(X, y, budget, meta, teacher, landscape, include_grad=False)


def refined_no_hess(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    del seed
    return _gain_refined_variant(X, y, budget, meta, teacher, landscape, include_hess=False)


def refined_no_conf(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    del seed
    return _gain_refined_variant(X, y, budget, meta, teacher, landscape, include_conf=False)


def refined_no_raw(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    del seed
    return _gain_refined_variant(X, y, budget, meta, teacher, landscape, include_raw=False)


def refined_no_gain_weight(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    del seed
    return _gain_refined_variant(X, y, budget, meta, teacher, landscape, include_gain_weight=False)


def refined_path_occupancy_only(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    del seed
    return _gain_refined_variant(
        X,
        y,
        budget,
        meta,
        teacher,
        landscape,
        include_grad=False,
        include_hess=False,
        include_conf=False,
        include_raw=False,
        include_gain_weight=False,
        include_path=True,
    )


def trajectory_refined(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    del seed
    S_match, _ = _trajectory_sketch(X, y, meta, teacher, landscape)
    selected = _select_from_sketch(S_match, y, budget, refine=True)
    return {"X": X[selected].astype(np.float32), "y": y[selected].astype(np.int64), "weights": np.ones(len(selected), dtype=np.float32)}


def split_margin_refined(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    del seed
    S_match, _ = _margin_sketch(X, y, meta, teacher, landscape, include_path=True)
    selected = _select_from_sketch(S_match, y, budget, refine=True)
    return {"X": X[selected].astype(np.float32), "y": y[selected].astype(np.int64), "weights": np.ones(len(selected), dtype=np.float32)}


def threshold_witness(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    del seed
    if landscape is None or len(getattr(landscape, "candidates", [])) == 0:
        return gain_path_refined(X, y, budget, seed, meta=meta, teacher=teacher, landscape=landscape)
    if teacher is not None and hasattr(teacher, "predict_proba"):
        p = teacher.predict_proba(X)
        conf = np.abs((p[:, 1] if p.ndim == 2 else p) - 0.5)
    else:
        conf = np.zeros(len(y), dtype=np.float64)
    gains = np.asarray(landscape.gains, dtype=np.float64)
    order = np.argsort(-gains)[: min(16, len(gains))]
    selected: list[int] = []
    for c in np.unique(y):
        cls = np.flatnonzero(y == c)
        scores = np.full(len(cls), np.inf, dtype=np.float64)
        for rank, idx in enumerate(order):
            j, b = landscape.candidates[int(idx)]
            thr = _thresholds(meta, int(j))[int(b)]
            col = X[cls, int(j)]
            scale = max(float(np.std(X[:, int(j)])), 1e-6)
            dist = np.abs(col - thr) / scale
            scores = np.minimum(scores, dist + 0.02 * rank + 0.10 * conf[cls])
        chosen = cls[np.argsort(scores)[: min(budget, len(cls))]].tolist()
        if len(chosen) < min(budget, len(cls)):
            extra = [int(i) for i in cls if int(i) not in set(chosen)]
            chosen.extend(extra[: min(budget, len(cls)) - len(chosen)])
        selected.extend(chosen[: min(budget, len(cls))])
    return {"X": X[selected].astype(np.float32), "y": y[selected].astype(np.int64), "weights": np.ones(len(selected), dtype=np.float32)}


def _leaf_region_ids(X: np.ndarray, teacher: Any | None, max_trees: int = 8) -> np.ndarray:
    if teacher is None or not hasattr(teacher, "apply"):
        return np.zeros(len(X), dtype=np.int64)
    try:
        leaves = np.asarray(teacher.apply(X))
        if leaves.ndim == 1:
            leaves = leaves[:, None]
        leaves = leaves[:, : min(max_trees, leaves.shape[1])].astype(np.int64)
        _, inv = np.unique(leaves, axis=0, return_inverse=True)
        return inv.astype(np.int64)
    except Exception:
        return np.zeros(len(X), dtype=np.int64)


def _allocate_region_quota(weights: np.ndarray, total: int) -> np.ndarray:
    if len(weights) == 0 or total <= 0:
        return np.zeros(len(weights), dtype=int)
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 1e-8)
    raw = total * weights / max(float(weights.sum()), 1e-8)
    quota = np.floor(raw).astype(int)
    positive = weights > 0
    quota[(quota == 0) & positive] = 1
    while int(quota.sum()) > total:
        locs = np.flatnonzero(quota > 0)
        if len(locs) == 0:
            break
        loc = int(locs[np.argmin(raw[locs] - np.floor(raw[locs]))])
        quota[loc] -= 1
    while int(quota.sum()) < total:
        loc = int(np.argmax(raw - quota))
        quota[loc] += 1
    return quota


def region_adaptive_refined(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    del seed
    S, scale = _gain_sketch(X, y, meta, teacher, landscape, include_path=True)
    S_match = S / scale[None, :]
    regions = _leaf_region_ids(X, teacher, max_trees=8)
    if teacher is not None and hasattr(teacher, "predict_proba"):
        p = teacher.predict_proba(X)
        prob = p[:, 1] if p.ndim == 2 else p
        uncertainty = 1.0 - 2.0 * np.abs(prob - 0.5)
    else:
        uncertainty = np.ones(len(y), dtype=np.float64)
    selected: list[int] = []
    for c in np.unique(y):
        cls = np.flatnonzero(y == c)
        target = min(budget, len(cls))
        rvals, inv = np.unique(regions[cls], return_inverse=True)
        weights = np.zeros(len(rvals), dtype=np.float64)
        for r in range(len(rvals)):
            loc = cls[inv == r]
            weights[r] = np.sqrt(len(loc)) * (1.0 + float(np.mean(uncertainty[loc])))
        quotas = _allocate_region_quota(weights, target)
        chosen: list[int] = []
        for r, q in enumerate(quotas):
            if q <= 0:
                continue
            loc = cls[inv == r]
            chosen.extend(_greedy_mean_match(S_match, loc, min(int(q), len(loc))))
        seen = set(chosen)
        if len(chosen) < target:
            fill = _greedy_mean_match(S_match, np.asarray([i for i in cls if int(i) not in seen], dtype=int), target - len(chosen))
            chosen.extend(fill)
        selected.extend(chosen[:target])
    selected = _refine_selection(S_match, y, selected, rounds=1)
    return {"X": X[selected].astype(np.float32), "y": y[selected].astype(np.int64), "weights": np.ones(len(selected), dtype=np.float32)}


def region_trajectory_refined(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    del seed
    S_match, _ = _trajectory_sketch(X, y, meta, teacher, landscape)
    regions = _leaf_region_ids(X, teacher, max_trees=8)
    selected: list[int] = []
    for c in np.unique(y):
        cls = np.flatnonzero(y == c)
        target = min(budget, len(cls))
        rvals, inv = np.unique(regions[cls], return_inverse=True)
        weights = np.asarray([np.sqrt(np.sum(inv == r)) for r in range(len(rvals))], dtype=np.float64)
        quotas = _allocate_region_quota(weights, target)
        chosen: list[int] = []
        for r, q in enumerate(quotas):
            if q <= 0:
                continue
            loc = cls[inv == r]
            chosen.extend(_greedy_mean_match(S_match, loc, min(int(q), len(loc))))
        seen = set(chosen)
        if len(chosen) < target:
            fill = _greedy_mean_match(S_match, np.asarray([i for i in cls if int(i) not in seen], dtype=int), target - len(chosen))
            chosen.extend(fill)
        selected.extend(chosen[:target])
    selected = _refine_selection(S_match, y, selected, rounds=1)
    return {"X": X[selected].astype(np.float32), "y": y[selected].astype(np.int64), "weights": np.ones(len(selected), dtype=np.float32)}


def gain_path_refined_weighted(
    X: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
    meta: dict[str, Any],
    teacher: Any | None = None,
    landscape: Any | None = None,
    **_: Any,
) -> dict[str, np.ndarray]:
    selected, S, scale, S_match = _select_by_gain_sketch(X, y, budget, meta, teacher, landscape, include_path=True)
    selected = _refine_selection(S_match, y, selected, rounds=2)
    weights = _fit_weights(S, scale, y, selected)
    return {"X": X[selected].astype(np.float32), "y": y[selected].astype(np.int64), "weights": weights}


def sketch_no_grad(X: np.ndarray, y: np.ndarray, budget: int, seed: int, **kwargs: Any) -> dict[str, np.ndarray]:
    return gain_herding(X, y, budget, seed, include_grad=False, **kwargs)


def sketch_no_hess(X: np.ndarray, y: np.ndarray, budget: int, seed: int, **kwargs: Any) -> dict[str, np.ndarray]:
    return gain_herding(X, y, budget, seed, include_hess=False, **kwargs)


def sketch_no_conf(X: np.ndarray, y: np.ndarray, budget: int, seed: int, **kwargs: Any) -> dict[str, np.ndarray]:
    return gain_herding(X, y, budget, seed, include_conf=False, **kwargs)


def sketch_no_raw(X: np.ndarray, y: np.ndarray, budget: int, seed: int, **kwargs: Any) -> dict[str, np.ndarray]:
    return gain_herding(X, y, budget, seed, include_raw=False, **kwargs)


def sketch_no_gain_weight(X: np.ndarray, y: np.ndarray, budget: int, seed: int, **kwargs: Any) -> dict[str, np.ndarray]:
    return gain_herding(X, y, budget, seed, include_gain_weight=False, **kwargs)


def sketch_occupancy_only(X: np.ndarray, y: np.ndarray, budget: int, seed: int, **kwargs: Any) -> dict[str, np.ndarray]:
    return gain_herding(
        X,
        y,
        budget,
        seed,
        include_grad=False,
        include_hess=False,
        include_conf=False,
        include_raw=False,
        include_gain_weight=False,
        **kwargs,
    )


BASELINES = {
    "full_data": full_data,
    "random": random_subset,
    "k_center": k_center,
    "herding": herding,
    "gradient_sampling": gradient_sampling,
    "distribution_matching": distribution_matching,
    "tree_region": tree_region_sampling,
    "histdistill_greedy": histdistill_greedy,
    "histdistill_refined": histdistill_refined,
    "histdistill_density": histdistill_density,
    "histdistill_density_weak": histdistill_density_weak,
    "histdistill_density_mid": histdistill_density_mid,
    "histdistill_density_strong": histdistill_density_strong,
    "histdistill_density_xstrong": histdistill_density_xstrong,
    "histdistill_importance": histdistill_importance,
    "goss_coreset": goss_coreset,
    "mvs_coreset": mvs_coreset,
    "craig_coreset": craig_coreset,
    "gradmatch_coreset": gradmatch_coreset,
    "grand_coreset": grand_coreset,
    "el2n_coreset": el2n_coreset,
    "histdistill_no_weight_fit": histdistill_no_weight_fit,
    "histdistill_linear_coverage": histdistill_linear_coverage,
    "histdistill_root_only": histdistill_root_only,
    "histdistill_no_refine": histdistill_no_refine,
    "gain_herding": gain_herding,
    "gain_weighted_coreset": gain_weighted_coreset,
    "gaindistill_hybrid": gaindistill_hybrid,
    "gain_path_herding": gain_path_herding,
    "gaindistill_path_hybrid": gaindistill_path_hybrid,
    "gain_path_prior_weighted": gain_path_prior_weighted,
    "gain_path_diverse": gain_path_diverse,
    "gain_path_dual": gain_path_dual,
    "gain_path_safeguarded": gain_path_safeguarded,
    "gain_path_refined": gain_path_refined,
    "gain_path_refined_weighted": gain_path_refined_weighted,
    "refined_no_path": refined_no_path,
    "refined_no_refine": refined_no_refine,
    "refined_no_grad": refined_no_grad,
    "refined_no_hess": refined_no_hess,
    "refined_no_conf": refined_no_conf,
    "refined_no_raw": refined_no_raw,
    "refined_no_gain_weight": refined_no_gain_weight,
    "refined_path_occupancy_only": refined_path_occupancy_only,
    "trajectory_refined": trajectory_refined,
    "split_margin_refined": split_margin_refined,
    "threshold_witness": threshold_witness,
    "region_adaptive_refined": region_adaptive_refined,
    "region_trajectory_refined": region_trajectory_refined,
    "sketch_no_grad": sketch_no_grad,
    "sketch_no_hess": sketch_no_hess,
    "sketch_no_conf": sketch_no_conf,
    "sketch_no_raw": sketch_no_raw,
    "sketch_no_gain_weight": sketch_no_gain_weight,
    "sketch_occupancy_only": sketch_occupancy_only,
}
