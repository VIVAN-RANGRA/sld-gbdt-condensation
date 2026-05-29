from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import spearmanr

from .models import predict_proba_positive
from .utils import ensure_dir, read_json, write_json


@dataclass
class ProbeState:
    candidates: np.ndarray
    gains: np.ndarray
    hist: np.ndarray
    parent: np.ndarray
    best_index: int
    checkpoint: int
    tree_index: int
    node_id: int
    depth: int
    path: list[tuple[int, float, str]]


@dataclass
class Landscape:
    candidates: np.ndarray
    gains: np.ndarray
    hist: np.ndarray
    parent: np.ndarray
    best_index: int
    marginals: list[np.ndarray]
    pair_indices: list[tuple[int, int]]
    pair_tables: list[np.ndarray]
    probes: list[ProbeState] = field(default_factory=list)


def candidate_grid(meta: dict[str, Any]) -> np.ndarray:
    rows: list[tuple[int, int]] = []
    for j, bmeta in enumerate(meta["bins"]):
        for b in range(len(bmeta["thresholds"])):
            rows.append((j, b))
    return np.array(rows, dtype=np.int64)


def _thresholds(meta: dict[str, Any], j: int) -> np.ndarray:
    return np.asarray(meta["bins"][j]["thresholds"], dtype=np.float32)


def _feature_index(split_name: str) -> int:
    return int(str(split_name).lstrip("f"))


def _iter_internal_nodes(node: dict[str, Any], tree_index: int, path: list[tuple[int, float, str]], max_depth: int) -> list[dict[str, Any]]:
    if "leaf" in node or int(node.get("depth", 0)) > max_depth:
        return []
    feat = _feature_index(node["split"])
    thr = float(node["split_condition"])
    out = [
        {
            "tree_index": tree_index,
            "node_id": int(node["nodeid"]),
            "depth": int(node.get("depth", 0)),
            "path": list(path),
        }
    ]
    children = node.get("children", [])
    if len(children) >= 2:
        yes_id = int(node.get("yes", children[0]["nodeid"]))
        for child in children:
            side = "left" if int(child["nodeid"]) == yes_id else "right"
            out.extend(_iter_internal_nodes(child, tree_index, path + [(feat, thr, side)], max_depth))
    return out


def extract_xgb_trees(model: Any) -> list[dict[str, Any]]:
    if not hasattr(model, "get_booster"):
        return []
    return [json.loads(s) for s in model.get_booster().get_dump(dump_format="json")]


def route_mask(X: np.ndarray, path: list[tuple[int, float, str]]) -> np.ndarray:
    mask = np.ones(X.shape[0], dtype=bool)
    for j, thr, side in path:
        if side == "left":
            mask &= X[:, int(j)] <= float(thr)
        else:
            mask &= X[:, int(j)] > float(thr)
    return mask


def raw_margin_at_checkpoint(model: Any, X: np.ndarray, y: np.ndarray, checkpoint: int) -> np.ndarray:
    if checkpoint <= 0:
        p = np.clip(float(np.mean(y)), 1e-6, 1 - 1e-6)
        return np.full(X.shape[0], np.log(p / (1.0 - p)), dtype=np.float64)
    try:
        return np.asarray(model.predict(X, output_margin=True, iteration_range=(0, int(checkpoint))), dtype=np.float64)
    except Exception:
        p = np.clip(predict_proba_positive(model, X), 1e-6, 1 - 1e-6)
        return np.log(p / (1.0 - p))


def grad_hess_from_margin(margin: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.clip(1.0 / (1.0 + np.exp(-margin)), 1e-6, 1 - 1e-6)
    g = p - y
    h = np.maximum(p * (1.0 - p), 1e-6)
    return g.astype(np.float64), h.astype(np.float64)


def compute_grad_hess(model: Any, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.clip(predict_proba_positive(model, X), 1e-6, 1 - 1e-6)
    g = p - y
    h = np.maximum(p * (1.0 - p), 1e-6)
    return g.astype(np.float64), h.astype(np.float64)


def _candidate_landscape_for_mask(
    X: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    meta: dict[str, Any],
    mask: np.ndarray,
    reg_lambda: float,
    gamma: float,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    grid = candidate_grid(meta)
    gains = np.zeros(len(grid), dtype=np.float64)
    hist = np.zeros((len(grid), 3), dtype=np.float64)
    G0, H0, N0 = g[mask].sum(), h[mask].sum(), float(mask.sum())
    parent = np.array([G0, H0, N0], dtype=np.float64)
    parent_score = (G0 * G0) / (H0 + reg_lambda + 1e-8)
    for idx, (j, b) in enumerate(grid):
        thr = _thresholds(meta, int(j))[int(b)]
        left = mask & (X[:, int(j)] <= thr)
        GL = g[left].sum()
        HL = h[left].sum()
        GR = G0 - GL
        HR = H0 - HL
        gains[idx] = 0.5 * ((GL * GL) / (HL + reg_lambda + 1e-8) + (GR * GR) / (HR + reg_lambda + 1e-8) - parent_score) - gamma
        hist[idx] = [left.sum(), GL, HL]

    if len(grid) == 0:
        return grid, gains, hist, parent, 0
    top_n = int(cfg["landscape"]["top_candidates"])
    hard_n = int(cfg["landscape"]["hard_candidates"])
    rand_n = int(cfg["landscape"]["random_candidates"])
    order = np.argsort(-gains)
    top = order[: min(top_n, len(order))]
    margin = gains[order[0]] - gains
    hard = np.argsort(margin)[: min(top_n + hard_n, len(order))]
    rand = rng.choice(len(order), size=min(rand_n, len(order)), replace=False)
    keep = np.unique(np.concatenate([top, hard, rand]))
    gains_keep = gains[keep]
    return grid[keep], gains_keep, hist[keep], parent, int(np.argmax(gains_keep))


def select_probe_regions(model: Any, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    trees = extract_xgb_trees(model)
    if not trees:
        return []
    rounds = cfg.get("teacher", {}).get("checkpoint_rounds", [0, 1, 5, 10, 25, 50])
    max_depth = int(cfg.get("landscape", {}).get("probe_max_depth", 2))
    max_probes = int(cfg.get("landscape", {}).get("max_probe_regions", 24))
    probes: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for r in rounds:
        tree_idx = min(max(0, int(r)), len(trees) - 1)
        for node in _iter_internal_nodes(trees[tree_idx], tree_idx, [], max_depth):
            key = (node["tree_index"], node["node_id"])
            if key not in seen:
                seen.add(key)
                probes.append(node)
            if len(probes) >= max_probes:
                return probes
    return probes


def compute_landscape_from_arrays(
    X: np.ndarray,
    y: np.ndarray,
    meta: dict[str, Any],
    model: Any,
    reg_lambda: float,
    gamma: float,
    cfg: dict[str, Any],
    seed: int,
) -> Landscape:
    rng = np.random.default_rng(seed)
    root_margin = raw_margin_at_checkpoint(model, X, y, 0)
    g, h = grad_hess_from_margin(root_margin, y)
    root_mask = np.ones(X.shape[0], dtype=bool)
    candidates, gains_keep, hist_keep, parent, best_index = _candidate_landscape_for_mask(
        X, g, h, meta, root_mask, reg_lambda, gamma, cfg, rng
    )

    probes: list[ProbeState] = []
    min_region_support = int(cfg.get("landscape", {}).get("min_region_support", 32))
    for region in select_probe_regions(model, cfg):
        tree_idx = int(region["tree_index"])
        margin = raw_margin_at_checkpoint(model, X, y, tree_idx)
        rg, rh = grad_hess_from_margin(margin, y)
        mask = route_mask(X, region["path"])
        if int(mask.sum()) < min_region_support:
            continue
        pc, pg, ph, pp, pb = _candidate_landscape_for_mask(X, rg, rh, meta, mask, reg_lambda, gamma, cfg, rng)
        if len(pc) == 0:
            continue
        probes.append(
            ProbeState(
                candidates=pc,
                gains=pg,
                hist=ph,
                parent=pp,
                best_index=pb,
                checkpoint=tree_idx,
                tree_index=tree_idx,
                node_id=int(region["node_id"]),
                depth=int(region["depth"]),
                path=list(region["path"]),
            )
        )

    marginals = []
    for j, bmeta in enumerate(meta["bins"]):
        thresholds = np.asarray(bmeta["thresholds"], dtype=np.float32)
        ids = np.searchsorted(thresholds, X[:, j], side="left")
        counts = np.bincount(ids, minlength=len(thresholds) + 1).astype(np.float64)
        marginals.append(counts / max(1.0, counts.sum()))

    feat_scores = np.zeros(X.shape[1], dtype=np.float64)
    for (j, _), gain in zip(candidates, gains_keep):
        feat_scores[int(j)] += max(0.0, float(gain))
    top_features = np.argsort(-feat_scores)[: min(int(cfg["distill"]["top_pair_count"]), X.shape[1])]
    pair_indices: list[tuple[int, int]] = []
    pair_tables: list[np.ndarray] = []
    for a in range(len(top_features)):
        for b in range(a + 1, len(top_features)):
            j, k = int(top_features[a]), int(top_features[b])
            tj, tk = _thresholds(meta, j), _thresholds(meta, k)
            zj = np.searchsorted(tj, X[:, j], side="left")
            zk = np.searchsorted(tk, X[:, k], side="left")
            table = np.zeros((len(tj) + 1, len(tk) + 1), dtype=np.float64)
            np.add.at(table, (zj, zk), 1.0)
            pair_indices.append((j, k))
            pair_tables.append(table / max(1.0, table.sum()))
            if len(pair_indices) >= int(cfg["distill"]["top_pair_count"]):
                break
        if len(pair_indices) >= int(cfg["distill"]["top_pair_count"]):
            break

    return Landscape(
        candidates=candidates,
        gains=gains_keep,
        hist=hist_keep,
        parent=parent,
        best_index=best_index,
        marginals=marginals,
        pair_indices=pair_indices,
        pair_tables=pair_tables,
        probes=probes,
    )


def compute_gains_for_candidates(
    X: np.ndarray,
    y: np.ndarray,
    meta: dict[str, Any],
    model: Any,
    candidates: np.ndarray,
    reg_lambda: float,
    gamma: float,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    g, h = compute_grad_hess(model, X, y)
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64)
        g = g * w
        h = h * w
    G0, H0 = g.sum(), h.sum()
    parent_score = (G0 * G0) / (H0 + reg_lambda + 1e-8)
    gains = np.zeros(len(candidates), dtype=np.float64)
    for idx, (j, b) in enumerate(candidates):
        thr = _thresholds(meta, int(j))[int(b)]
        left = X[:, int(j)] <= thr
        GL = g[left].sum()
        HL = h[left].sum()
        GR = G0 - GL
        HR = H0 - HL
        gains[idx] = 0.5 * ((GL * GL) / (HL + reg_lambda + 1e-8) + (GR * GR) / (HR + reg_lambda + 1e-8) - parent_score) - gamma
    return gains


def _state_gains_for_candidates(
    X: np.ndarray,
    y: np.ndarray,
    meta: dict[str, Any],
    model: Any,
    candidates: np.ndarray,
    path: list[tuple[int, float, str]],
    checkpoint: int,
    reg_lambda: float,
    gamma: float,
    sample_weight: np.ndarray | None = None,
    target_hessian_mass: float | None = None,
) -> np.ndarray:
    if int(checkpoint) <= 0:
        if sample_weight is not None:
            w0 = np.asarray(sample_weight, dtype=np.float64)
            p0 = float(np.sum(w0 * y) / max(1e-8, np.sum(w0)))
        else:
            p0 = float(np.mean(y))
        p0 = float(np.clip(p0, 1e-6, 1.0 - 1e-6))
        margin = np.full(X.shape[0], np.log(p0 / (1.0 - p0)), dtype=np.float64)
    else:
        margin = raw_margin_at_checkpoint(model, X, y, int(checkpoint))
    g, h = grad_hess_from_margin(margin, y)
    mask = route_mask(X, path)
    if sample_weight is not None or target_hessian_mass is not None:
        w = np.ones(X.shape[0], dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).copy()
        if target_hessian_mass is not None:
            current_hessian_mass = float(np.sum(h[mask] * w[mask]))
            if current_hessian_mass > 1e-12:
                w *= float(target_hessian_mass) / current_hessian_mass
        g = g * w
        h = h * w
    G0, H0 = g[mask].sum(), h[mask].sum()
    parent_score = (G0 * G0) / (H0 + reg_lambda + 1e-8)
    gains = np.zeros(len(candidates), dtype=np.float64)
    for idx, (j, b) in enumerate(candidates):
        thr = _thresholds(meta, int(j))[int(b)]
        left = mask & (X[:, int(j)] <= thr)
        GL = g[left].sum()
        HL = h[left].sum()
        GR = G0 - GL
        HR = H0 - HL
        gains[idx] = 0.5 * ((GL * GL) / (HL + reg_lambda + 1e-8) + (GR * GR) / (HR + reg_lambda + 1e-8) - parent_score) - gamma
    return gains


def split_landscape_discrepancy(
    full: Landscape,
    X: np.ndarray,
    y: np.ndarray,
    meta: dict[str, Any],
    teacher: Any,
    reg_lambda: float,
    gamma: float,
    sample_weight: np.ndarray | None = None,
    max_probes: int | None = None,
) -> dict[str, float]:
    states: list[tuple[str, np.ndarray, np.ndarray, list[tuple[int, float, str]], int, int, float]] = [
        ("root", full.candidates, full.gains, [], 0, int(full.best_index), float(full.parent[1]))
    ]
    probes = list(full.probes)
    if max_probes is not None:
        probes = probes[: max(0, int(max_probes))]
    for probe in probes:
        states.append(
            (
                f"probe_depth_{int(probe.depth)}",
                probe.candidates,
                probe.gains,
                probe.path,
                int(probe.checkpoint),
                int(probe.best_index),
                float(probe.parent[1]),
            )
        )

    rows = []
    for name, candidates, target, path, checkpoint, best_index, target_hessian_mass in states:
        if len(candidates) == 0 or len(target) == 0:
            continue
        other = _state_gains_for_candidates(
            X,
            y,
            meta,
            teacher,
            candidates,
            path,
            checkpoint,
            reg_lambda,
            gamma,
            sample_weight=sample_weight,
            target_hessian_mass=target_hessian_mass,
        )
        diff = np.asarray(other, dtype=np.float64) - np.asarray(target, dtype=np.float64)
        order = np.argsort(-target)
        if len(order) > 1:
            margin = float(target[order[0]] - target[order[1]])
        else:
            margin = 0.0
        scale = max(1.0, float(np.max(np.abs(target))) if len(target) else 1.0)
        other_order = np.argsort(-other)
        corr = spearmanr(target, other).correlation if len(target) > 2 else np.nan
        rows.append(
            {
                "name": name,
                "sld_inf": float(np.max(np.abs(diff))) if len(diff) else 0.0,
                "sld_l1": float(np.sum(np.abs(diff))),
                "sld_inf_norm": float(np.max(np.abs(diff)) / scale) if len(diff) else 0.0,
                "sld_l1_norm": float(np.mean(np.abs(diff)) / scale) if len(diff) else 0.0,
                "agreement": float(int(other_order[0]) == int(best_index)) if len(other_order) else 0.0,
                "margin": margin,
                "margin_satisfied": float((np.max(np.abs(diff)) if len(diff) else 0.0) < margin / 2.0) if margin > 0 else 0.0,
                "rank_corr": float(0.0 if np.isnan(corr) else corr),
                "depth": 0 if name == "root" else int(name.rsplit("_", 1)[-1]),
            }
        )

    if not rows:
        return {}
    root = rows[0]
    out = {
        "sld_inf": float(np.mean([r["sld_inf"] for r in rows])),
        "sld_l1": float(np.mean([r["sld_l1"] for r in rows])),
        "sld_inf_norm": float(np.mean([r["sld_inf_norm"] for r in rows])),
        "sld_l1_norm": float(np.mean([r["sld_l1_norm"] for r in rows])),
        "sld_root_inf": float(root["sld_inf"]),
        "sld_root_l1": float(root["sld_l1"]),
        "sld_root_inf_norm": float(root["sld_inf_norm"]),
        "sld_root_l1_norm": float(root["sld_l1_norm"]),
        "sld_root_margin": float(root["margin"]),
        "sld_root_margin_satisfied": float(root["margin_satisfied"]),
        "sld_root_agreement_direct": float(root["agreement"]),
        "sld_rank_correlation_direct": float(np.mean([r["rank_corr"] for r in rows])),
        "sld_margin_satisfied_rate": float(np.mean([r["margin_satisfied"] for r in rows])),
        "sld_states": float(len(rows)),
    }
    for depth in sorted({int(r["depth"]) for r in rows}):
        drows = [r for r in rows if int(r["depth"]) == depth]
        out[f"sld_depth_{depth}_inf_norm"] = float(np.mean([r["sld_inf_norm"] for r in drows]))
        out[f"sld_depth_{depth}_agreement"] = float(np.mean([r["agreement"] for r in drows]))
    return out


def split_regret_metrics(
    full: Landscape,
    X: np.ndarray,
    y: np.ndarray,
    meta: dict[str, Any],
    teacher: Any,
    reg_lambda: float,
    gamma: float,
    sample_weight: np.ndarray | None = None,
    max_probes: int | None = None,
) -> dict[str, float]:
    states: list[tuple[str, np.ndarray, np.ndarray, list[tuple[int, float, str]], int, float]] = [
        ("root", full.candidates, full.gains, [], 0, float(full.parent[1]))
    ]
    probes = list(full.probes)
    if max_probes is not None:
        probes = probes[: max(0, int(max_probes))]
    for probe in probes:
        states.append(
            (
                f"probe_depth_{int(probe.depth)}",
                probe.candidates,
                probe.gains,
                probe.path,
                int(probe.checkpoint),
                float(probe.parent[1]),
            )
        )
    rows = []
    for name, candidates, target, path, checkpoint, target_hessian_mass in states:
        if len(candidates) == 0 or len(target) == 0:
            continue
        other = _state_gains_for_candidates(
            X,
            y,
            meta,
            teacher,
            candidates,
            path,
            checkpoint,
            reg_lambda,
            gamma,
            sample_weight=sample_weight,
            target_hessian_mass=target_hessian_mass,
        )
        full_best = int(np.argmax(target))
        other_best = int(np.argmax(other))
        best_gain = float(target[full_best])
        chosen_gain = float(target[other_best])
        regret = max(0.0, best_gain - chosen_gain)
        scale = max(1.0, abs(best_gain), float(np.max(np.abs(target))) if len(target) else 1.0)
        weight = max(1.0, abs(best_gain))
        rows.append(
            {
                "name": name,
                "depth": 0 if name == "root" else int(name.rsplit("_", 1)[-1]),
                "split_regret": regret,
                "split_regret_norm": regret / scale,
                "agreement": float(full_best == other_best),
                "weight": weight,
            }
        )
    if not rows:
        return {}
    weights = np.asarray([r["weight"] for r in rows], dtype=np.float64)
    weights = weights / max(float(weights.sum()), 1e-12)
    regrets = np.asarray([r["split_regret"] for r in rows], dtype=np.float64)
    regrets_norm = np.asarray([r["split_regret_norm"] for r in rows], dtype=np.float64)
    agreements = np.asarray([r["agreement"] for r in rows], dtype=np.float64)
    root = rows[0]
    out = {
        "split_regret": float(np.sum(weights * regrets)),
        "split_regret_norm": float(np.sum(weights * regrets_norm)),
        "split_regret_mean": float(np.mean(regrets)),
        "split_regret_norm_mean": float(np.mean(regrets_norm)),
        "split_regret_root": float(root["split_regret"]),
        "split_regret_root_norm": float(root["split_regret_norm"]),
        "split_regret_agreement": float(np.mean(agreements)),
        "split_regret_weighted_agreement": float(np.sum(weights * agreements)),
        "split_regret_states": float(len(rows)),
    }
    for depth in sorted({int(r["depth"]) for r in rows}):
        drows = [r for r in rows if int(r["depth"]) == depth]
        out[f"split_regret_depth_{depth}_norm"] = float(np.mean([r["split_regret_norm"] for r in drows]))
        out[f"split_regret_depth_{depth}_agreement"] = float(np.mean([r["agreement"] for r in drows]))
    return out


def save_landscape(path: str | Path, landscape: Landscape) -> None:
    p = ensure_dir(path)
    extra: dict[str, np.ndarray] = {}
    probe_meta: list[dict[str, Any]] = []
    for i, probe in enumerate(landscape.probes):
        extra[f"probe_{i}_candidates"] = probe.candidates
        extra[f"probe_{i}_gains"] = probe.gains
        extra[f"probe_{i}_hist"] = probe.hist
        extra[f"probe_{i}_parent"] = probe.parent
        extra[f"probe_{i}_best_index"] = np.array([probe.best_index], dtype=np.int64)
        probe_meta.append(
            {
                "checkpoint": int(probe.checkpoint),
                "tree_index": int(probe.tree_index),
                "node_id": int(probe.node_id),
                "depth": int(probe.depth),
                "path": [[int(j), float(thr), str(side)] for j, thr, side in probe.path],
            }
        )
    np.savez_compressed(
        p / "landscape.npz",
        candidates=landscape.candidates,
        gains=landscape.gains,
        hist=landscape.hist,
        parent=landscape.parent,
        best_index=np.array([landscape.best_index]),
        pair_indices=np.asarray(landscape.pair_indices, dtype=np.int64) if landscape.pair_indices else np.empty((0, 2), dtype=np.int64),
        **{f"marginal_{i}": m for i, m in enumerate(landscape.marginals)},
        **{f"pair_{i}": t for i, t in enumerate(landscape.pair_tables)},
        **extra,
    )
    write_json(
        p / "metadata.json",
        {
            "num_candidates": int(len(landscape.candidates)),
            "best_index": int(landscape.best_index),
            "num_probes": int(len(landscape.probes)),
            "probes": probe_meta,
        },
    )


def load_landscape(path: str | Path) -> Landscape:
    p = Path(path)
    arr = np.load(p / "landscape.npz")
    meta = read_json(p / "metadata.json") if (p / "metadata.json").exists() else {}
    marginals = [arr[k] for k in sorted([k for k in arr.files if k.startswith("marginal_")], key=lambda x: int(x.split("_")[1]))]
    pair_keys = [k for k in arr.files if k.startswith("pair_") and k.split("_")[1].isdigit()]
    pair_tables = [arr[k] for k in sorted(pair_keys, key=lambda x: int(x.split("_")[1]))]
    probes: list[ProbeState] = []
    for i, pmeta in enumerate(meta.get("probes", [])):
        key = f"probe_{i}_candidates"
        if key not in arr.files:
            continue
        probes.append(
            ProbeState(
                candidates=arr[key],
                gains=arr[f"probe_{i}_gains"],
                hist=arr[f"probe_{i}_hist"],
                parent=arr[f"probe_{i}_parent"],
                best_index=int(arr[f"probe_{i}_best_index"][0]),
                checkpoint=int(pmeta["checkpoint"]),
                tree_index=int(pmeta["tree_index"]),
                node_id=int(pmeta["node_id"]),
                depth=int(pmeta["depth"]),
                path=[(int(j), float(thr), str(side)) for j, thr, side in pmeta.get("path", [])],
            )
        )
    return Landscape(
        candidates=arr["candidates"],
        gains=arr["gains"],
        hist=arr["hist"],
        parent=arr["parent"],
        best_index=int(arr["best_index"][0]),
        marginals=marginals,
        pair_indices=[tuple(map(int, row)) for row in arr["pair_indices"]],
        pair_tables=pair_tables,
        probes=probes,
    )


def split_fidelity(full: Landscape, other_gains: np.ndarray, top_k: int = 5) -> dict[str, float]:
    k = min(top_k, len(full.gains), len(other_gains))
    full_order = np.argsort(-full.gains)
    other_order = np.argsort(-other_gains)
    corr = spearmanr(full.gains, other_gains).correlation if len(full.gains) > 2 else np.nan
    diff = np.asarray(other_gains, dtype=np.float64) - np.asarray(full.gains, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(full.gains))) if len(full.gains) else 1.0)
    margin = float(full.gains[full_order[0]] - full.gains[full_order[1]]) if len(full_order) > 1 else 0.0
    return {
        "root_agreement": float(full_order[0] == other_order[0]),
        "top5_overlap": float(len(set(full_order[:k]).intersection(set(other_order[:k]))) / max(1, k)),
        "gain_rank_correlation": float(0.0 if np.isnan(corr) else corr),
        "root_sld_inf": float(np.max(np.abs(diff))) if len(diff) else 0.0,
        "root_sld_l1": float(np.sum(np.abs(diff))),
        "root_sld_inf_norm": float(np.max(np.abs(diff)) / scale) if len(diff) else 0.0,
        "root_sld_l1_norm": float(np.mean(np.abs(diff)) / scale) if len(diff) else 0.0,
        "root_gain_margin": margin,
        "root_margin_satisfied": float((np.max(np.abs(diff)) if len(diff) else 0.0) < margin / 2.0) if margin > 0 else 0.0,
    }


def load_teacher(path: str | Path) -> Any:
    return joblib.load(path)
