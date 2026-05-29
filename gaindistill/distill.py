from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .landscape import Landscape, ProbeState, extract_xgb_trees
from .utils import ensure_dir, write_json


def _js(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * (p * (p / m).log()).sum() + 0.5 * (q * (q / m).log()).sum()


def _bin_ids(X: np.ndarray, meta: dict[str, Any]) -> list[np.ndarray]:
    ids: list[np.ndarray] = []
    for j, bmeta in enumerate(meta["bins"]):
        thresholds = np.asarray(bmeta["thresholds"], dtype=np.float32)
        ids.append(np.searchsorted(thresholds, X[:, j], side="left"))
    return ids


def _threshold_prob(probs: list[torch.Tensor], j: int, threshold: float, meta: dict[str, Any]) -> torch.Tensor:
    thresholds = np.asarray(meta["bins"][j]["thresholds"], dtype=np.float32)
    if len(thresholds) == 0:
        return torch.ones(probs[j].shape[0], dtype=probs[j].dtype, device=probs[j].device)
    idx = int(np.searchsorted(thresholds, float(threshold), side="left")) + 1
    idx = max(1, min(idx, probs[j].shape[1]))
    return probs[j][:, :idx].sum(dim=1)


def _path_prob(probs: list[torch.Tensor], path: list[tuple[int, float, str]], meta: dict[str, Any]) -> torch.Tensor:
    out = torch.ones(probs[0].shape[0], dtype=probs[0].dtype, device=probs[0].device)
    for j, threshold, side in path:
        left = _threshold_prob(probs, int(j), float(threshold), meta)
        out = out * (left if side == "left" else (1.0 - left))
    return out


def _tree_leaf_paths(node: dict[str, Any], path: list[tuple[int, float, str]] | None = None) -> list[tuple[list[tuple[int, float, str]], float]]:
    path = path or []
    if "leaf" in node:
        return [(path, float(node["leaf"]))]
    feat = int(str(node["split"]).lstrip("f"))
    threshold = float(node["split_condition"])
    children = node.get("children", [])
    yes = int(node.get("yes", children[0]["nodeid"] if children else -1))
    leaves: list[tuple[list[tuple[int, float, str]], float]] = []
    for child in children:
        side = "left" if int(child["nodeid"]) == yes else "right"
        leaves.extend(_tree_leaf_paths(child, path + [(feat, threshold, side)]))
    return leaves


def _soft_checkpoint_margins(
    probs: list[torch.Tensor],
    teacher: Any,
    checkpoints: list[int],
    meta: dict[str, Any],
    base_logit: float,
    enabled: bool = True,
) -> dict[int, torch.Tensor]:
    unique = sorted(set(int(c) for c in checkpoints))
    out: dict[int, torch.Tensor] = {}
    base = torch.full((probs[0].shape[0],), float(base_logit), dtype=probs[0].dtype, device=probs[0].device)
    if not enabled:
        return {c: base for c in unique}
    trees = extract_xgb_trees(teacher)
    if not trees:
        return {c: base for c in unique}
    leaf_cache = [_tree_leaf_paths(tree) for tree in trees[: max(unique + [0])]]
    running = base.clone()
    out[0] = base
    max_ckpt = max(unique) if unique else 0
    for t in range(max_ckpt):
        if t >= len(leaf_cache):
            break
        add = torch.zeros_like(base)
        for path, leaf_value in leaf_cache[t]:
            add = add + _path_prob(probs, path, meta) * float(leaf_value)
        running = running + add
        if (t + 1) in unique:
            out[t + 1] = running.clone()
    for c in unique:
        out.setdefault(c, running.clone())
    return out


def _fallback_bin_value(thresholds: np.ndarray, bid: int) -> float:
    if bid == 0:
        return float(thresholds[0] - 1e-3) if len(thresholds) else 0.0
    if bid >= len(thresholds):
        return float(thresholds[-1] + 1e-3) if len(thresholds) else 0.0
    return float(0.5 * (thresholds[bid - 1] + thresholds[bid]))


def _decode_rows(
    final_probs: list[np.ndarray],
    meta: dict[str, Any],
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    X_anchor: np.ndarray | None = None,
    y_syn: np.ndarray | None = None,
    strategy: str = "midpoint",
) -> np.ndarray:
    decoded_bins = np.vstack([p.argmax(axis=1) for p in final_probs]).T
    X_syn = np.zeros((decoded_bins.shape[0], len(final_probs)), dtype=np.float32)
    anchor_ids = _bin_ids(X_anchor, meta) if X_anchor is not None and strategy.startswith("anchor") else None
    train_ids = _bin_ids(X_train, meta) if X_train is not None and "median" in strategy else None
    for j, bmeta in enumerate(meta["bins"]):
        thresholds = np.asarray(bmeta["thresholds"], dtype=np.float32)
        for a in range(decoded_bins.shape[0]):
            bid = int(decoded_bins[a, j])
            if anchor_ids is not None and int(anchor_ids[j][a]) == bid:
                val = float(X_anchor[a, j])
            elif train_ids is not None:
                mask = train_ids[j] == bid
                if strategy.endswith("class_median") and y_train is not None and y_syn is not None:
                    class_mask = mask & (y_train == int(y_syn[a]))
                    if class_mask.sum() >= 3:
                        mask = class_mask
                val = float(np.median(X_train[mask, j])) if mask.any() else _fallback_bin_value(thresholds, bid)
            else:
                val = _fallback_bin_value(thresholds, bid)
            X_syn[a, j] = val
    return X_syn


def _init_from_real_rows(
    X: np.ndarray,
    y: np.ndarray,
    meta: dict[str, Any],
    landscape: Landscape,
    teacher: Any | None,
    budget: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from .baselines import gain_path_herding

        selected = gain_path_herding(X, y, budget, seed, meta=meta, teacher=teacher, landscape=landscape)
        return selected["X"], selected["y"]
    except Exception:
        rng = np.random.default_rng(seed)
        idx: list[int] = []
        for c in np.unique(y):
            cls = np.flatnonzero(y == c)
            rng.shuffle(cls)
            idx.extend(cls[: min(budget, len(cls))].tolist())
        return X[idx], y[idx]


def distill_multi_probe(
    X: np.ndarray,
    y: np.ndarray,
    meta: dict[str, Any],
    landscape: Landscape,
    cfg: dict[str, Any],
    seed: int,
    budget_per_class: int | None = None,
    teacher: Any | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dc = cfg["distill"]
    budget = int(budget_per_class or dc["budget_per_class"])
    X_init, y_init = _init_from_real_rows(X, y, meta, landscape, teacher, budget, seed)
    m = int(len(y_init))
    n_train = float(len(y))
    bin_counts = [len(b["thresholds"]) + 1 for b in meta["bins"]]
    max_bins = max(bin_counts)

    logits = torch.full((m, len(bin_counts), max_bins), -8.0, dtype=torch.float32)
    init_ids = _bin_ids(X_init, meta)
    init_bin_targets: list[torch.Tensor] = []
    for j, bc in enumerate(bin_counts):
        logits[:, j, bc:] = -30.0
        ids_j = np.asarray(init_ids[j], dtype=np.int64)
        init_bin_targets.append(torch.tensor(np.clip(ids_j, 0, bc - 1), dtype=torch.long))
        for a, bid in enumerate(init_ids[j]):
            logits[a, j, int(min(max(bid, 0), bc - 1))] = 8.0
    logits = logits + 0.01 * torch.randn_like(logits)
    logits.requires_grad_(True)
    rho = torch.zeros(m, dtype=torch.float32, requires_grad=True)
    phi = torch.tensor(np.where(y_init > 0, 4.0, -4.0), dtype=torch.float32, requires_grad=True)
    y_init_t = torch.tensor(y_init.astype(np.float32), dtype=torch.float32)
    opt = torch.optim.Adam([logits, rho, phi], lr=float(dc["lr"]))

    all_probes = [
        ProbeState(
            candidates=landscape.candidates,
            gains=landscape.gains,
            hist=landscape.hist,
            parent=landscape.parent,
            best_index=landscape.best_index,
            checkpoint=0,
            tree_index=0,
            node_id=0,
            depth=0,
            path=[],
        )
    ] + list(landscape.probes)
    max_train_probes = int(dc.get("max_train_probes", 12))
    all_probes = all_probes[: max_train_probes]

    reg_lambda = float(cfg["teacher"]["reg_lambda"])
    gamma = float(cfg["teacher"]["gamma"])
    y_mean = float(np.mean(y))
    base_logit = float(np.log(np.clip(y_mean, 1e-6, 1 - 1e-6) / np.clip(1 - y_mean, 1e-6, 1 - 1e-6)))
    class_prior = torch.tensor([1.0 - y_mean, y_mean], dtype=torch.float32)
    marginal_targets = [torch.tensor(mg, dtype=torch.float32) for mg in landscape.marginals]
    pair_targets = [torch.tensor(pt, dtype=torch.float32) for pt in landscape.pair_tables]
    checkpoints = [p.checkpoint for p in all_probes]

    probe_targets = []
    for probe in all_probes:
        probe_targets.append(
            {
                "candidates": torch.tensor(probe.candidates, dtype=torch.long),
                "gains": torch.tensor(probe.gains, dtype=torch.float32),
                "pi": F.softmax(torch.tensor(probe.gains, dtype=torch.float32) / float(cfg["landscape"]["tau_gain"]), dim=0),
                "hist": torch.tensor(probe.hist, dtype=torch.float32),
                "parent": torch.tensor(probe.parent, dtype=torch.float32),
                "best": int(probe.best_index),
                "path": probe.path,
                "checkpoint": int(probe.checkpoint),
            }
        )

    loss_trace: list[dict[str, float]] = []
    steps = int(dc["steps"])
    for step in range(steps):
        frac = step / max(1, steps - 1)
        tau = float(dc["temperature_start"]) * (float(dc["temperature_end"]) / float(dc["temperature_start"])) ** frac
        probs = [F.softmax(logits[:, j, :bc] / tau, dim=-1) for j, bc in enumerate(bin_counts)]
        weights = F.softplus(rho) + 1e-4
        weights = weights * (n_train / weights.sum())
        y_soft = torch.sigmoid(phi)
        use_soft_routing = bool(dc.get("use_soft_routing", True))
        margins = _soft_checkpoint_margins(
            probs,
            teacher,
            checkpoints,
            meta,
            base_logit,
            enabled=bool(dc.get("use_soft_teacher_routing", True)) and use_soft_routing,
        )

        losses = {"gain": [], "margin": [], "newton": [], "child": [], "hist": []}
        for target in probe_targets:
            Q = _path_prob(probs, target["path"], meta) if use_soft_routing else torch.ones(m, dtype=weights.dtype, device=weights.device)
            pred = torch.sigmoid(margins[int(target["checkpoint"])])
            g = pred - y_soft
            h = torch.clamp(pred * (1.0 - pred), min=1e-6)
            WQ = weights * Q
            G0 = (WQ * g).sum()
            H0 = torch.clamp((WQ * h).sum(), min=1e-6)
            N0 = WQ.sum()
            parent_score = (G0 * G0) / (H0 + reg_lambda + 1e-8)
            synth_gains = []
            synth_hist = []
            for row in target["candidates"]:
                j = int(row[0].item())
                b = int(row[1].item())
                A = probs[j][:, : b + 1].sum(dim=1)
                GL = (WQ * A * g).sum()
                HL = torch.clamp((WQ * A * h).sum(), min=1e-6)
                NL = (WQ * A).sum()
                GR = G0 - GL
                HR = torch.clamp(H0 - HL, min=1e-6)
                synth_gains.append(0.5 * ((GL * GL) / (HL + reg_lambda + 1e-8) + (GR * GR) / (HR + reg_lambda + 1e-8) - parent_score) - gamma)
                synth_hist.append(torch.stack([NL, GL, HL]))
            synth_gains_t = torch.stack(synth_gains)
            synth_hist_t = torch.stack(synth_hist)
            synth_pi = F.softmax(synth_gains_t / float(cfg["landscape"]["tau_gain"]), dim=0)
            losses["gain"].append(_js(target["pi"], synth_pi))
            full_delta = target["gains"][target["best"]] - target["gains"]
            synth_delta = synth_gains_t[target["best"]] - synth_gains_t
            mask = torch.ones_like(full_delta, dtype=torch.bool)
            mask[target["best"]] = False
            losses["margin"].append(torch.relu(float(dc["margin_alpha"]) * full_delta[mask] - synth_delta[mask]).pow(2).mean())
            target_parent = target["parent"]
            scale = max(1.0, float(target_parent[2].item()))
            losses["newton"].append(F.mse_loss(torch.stack([G0, H0]) / scale, target_parent[:2] / scale))
            topk = torch.argsort(target["gains"], descending=True)[: min(16, len(target["gains"]))]
            losses["child"].append(F.l1_loss(synth_hist_t[topk, 1:] / scale, target["hist"][topk, 1:] / scale))
            synth_norm = torch.stack([synth_hist_t[:, 0] / max(1.0, N0.detach().item()), synth_hist_t[:, 1] / scale, synth_hist_t[:, 2] / scale], dim=1)
            target_norm = torch.stack([target["hist"][:, 0] / scale, target["hist"][:, 1] / scale, target["hist"][:, 2] / scale], dim=1)
            losses["hist"].append(F.l1_loss(synth_norm, target_norm))

        loss_gain = torch.stack(losses["gain"]).mean()
        loss_margin = torch.stack(losses["margin"]).mean()
        loss_newton = torch.stack(losses["newton"]).mean()
        loss_child = torch.stack(losses["child"]).mean()
        loss_hist = torch.stack(losses["hist"]).mean()
        synth_class = torch.stack([(weights * (1.0 - y_soft)).sum() / n_train, (weights * y_soft).sum() / n_train])
        loss_class = F.mse_loss(synth_class, class_prior)
        loss_marg = torch.tensor(0.0)
        for j, p in enumerate(probs):
            mg = (weights[:, None] * p).sum(dim=0) / n_train
            loss_marg = loss_marg + _js(marginal_targets[j], mg)
        loss_marg = loss_marg / max(1, len(probs))
        loss_pair = torch.tensor(0.0)
        for (j, k), target in zip(landscape.pair_indices, pair_targets):
            table = torch.einsum("mb,mc,m->bc", probs[j], probs[k], weights) / n_train
            loss_pair = loss_pair + _js(target.flatten(), table.flatten())
        loss_pair = loss_pair / max(1, len(pair_targets))
        entropy = torch.stack([-(p * torch.clamp(p, 1e-8, 1.0).log()).sum(dim=1).mean() for p in probs]).mean()
        loss_anchor = torch.stack([F.nll_loss(torch.clamp(p, 1e-8, 1.0).log(), init_bin_targets[j]) for j, p in enumerate(probs)]).mean()
        loss_label_anchor = F.binary_cross_entropy(y_soft, y_init_t)

        loss = (
            float(dc["weight_gain"]) * loss_gain
            + float(dc["weight_margin"]) * loss_margin
            + float(dc["weight_newton"]) * loss_newton
            + float(dc["weight_child"]) * loss_child
            + float(dc["weight_hist"]) * loss_hist
            + float(dc["weight_class"]) * loss_class
            + float(dc["weight_marginal"]) * loss_marg
            + float(dc["weight_pairwise"]) * loss_pair
            + float(dc["weight_entropy"]) * entropy
            + float(dc.get("weight_anchor", 0.0)) * loss_anchor
            + float(dc.get("weight_label_anchor", 0.0)) * loss_label_anchor
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for j, bc in enumerate(bin_counts):
                logits[:, j, bc:] = -30.0
        if step == 0 or (step + 1) % max(20, steps // 10) == 0:
            loss_trace.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "gain": float(loss_gain.detach()),
                    "margin": float(loss_margin.detach()),
                    "anchor": float(loss_anchor.detach()),
                    "probes": len(all_probes),
                }
            )

    with torch.no_grad():
        final_probs = [F.softmax(logits[:, j, :bc] / float(dc["temperature_end"]), dim=-1).cpu().numpy() for j, bc in enumerate(bin_counts)]
        y_prob = torch.sigmoid(phi).cpu().numpy()
        y_syn = (y_prob >= 0.5).astype(np.int64)
        # Avoid degenerate hard-label decode after soft-label optimization.
        if len(np.unique(y_syn)) < 2:
            y_syn = y_init.astype(np.int64)
        X_syn = _decode_rows(
            final_probs,
            meta,
            X_train=X,
            y_train=y,
            X_anchor=X_init,
            y_syn=y_syn,
            strategy=str(dc.get("decode_strategy", "midpoint")),
        )
        w_syn = (F.softplus(rho) + 1e-4).cpu().numpy()
        w_syn = w_syn * (len(y) / max(1e-8, w_syn.sum()))
    return {"X": X_syn, "y": y_syn, "weights": w_syn.astype(np.float32), "trace": loss_trace}


def distill_root(
    X: np.ndarray,
    y: np.ndarray,
    meta: dict[str, Any],
    landscape: Landscape,
    cfg: dict[str, Any],
    seed: int,
    budget_per_class: int | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dc = cfg["distill"]
    budget = int(budget_per_class or dc["budget_per_class"])
    class_counts = np.bincount(y, minlength=2)
    m_per = [min(budget, max(1, int(class_counts[c]))) for c in range(2)]
    labels = torch.tensor([0] * m_per[0] + [1] * m_per[1], dtype=torch.float32)
    m = int(labels.numel())
    n_train = float(len(y))
    bin_counts = [len(b["thresholds"]) + 1 for b in meta["bins"]]
    max_bins = max(bin_counts)

    logits = torch.randn((m, len(bin_counts), max_bins), dtype=torch.float32) * 0.05
    for j, bc in enumerate(bin_counts):
        logits[:, j, bc:] = -30.0
    logits.requires_grad_(True)
    rho = torch.zeros(m, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([logits, rho], lr=float(dc["lr"]))

    y_mean = float(y.mean())
    pred = torch.full((m,), y_mean, dtype=torch.float32)
    g = pred - labels
    h = torch.clamp(pred * (1.0 - pred), min=1e-6)
    cand = torch.tensor(landscape.candidates, dtype=torch.long)
    target_gains = torch.tensor(landscape.gains, dtype=torch.float32)
    target_pi = F.softmax(target_gains / float(cfg["landscape"]["tau_gain"]), dim=0)
    target_hist = torch.tensor(landscape.hist, dtype=torch.float32)
    parent_target = torch.tensor([landscape.parent[0] / n_train, landscape.parent[1] / n_train], dtype=torch.float32)
    class_prior = torch.tensor([1.0 - y_mean, y_mean], dtype=torch.float32)
    marginal_targets = [torch.tensor(mg, dtype=torch.float32) for mg in landscape.marginals]
    pair_targets = [torch.tensor(pt, dtype=torch.float32) for pt in landscape.pair_tables]
    reg_lambda = float(cfg["teacher"]["reg_lambda"])
    gamma = float(cfg["teacher"]["gamma"])
    best_idx = int(landscape.best_index)

    loss_trace: list[dict[str, float]] = []
    steps = int(dc["steps"])
    for step in range(steps):
        frac = step / max(1, steps - 1)
        tau = float(dc["temperature_start"]) * (float(dc["temperature_end"]) / float(dc["temperature_start"])) ** frac
        probs = []
        for j, bc in enumerate(bin_counts):
            probs.append(F.softmax(logits[:, j, :bc] / tau, dim=-1))
        weights = F.softplus(rho) + 1e-4
        weights = weights * (n_train / weights.sum())
        G0 = (weights * g).sum()
        H0 = torch.clamp((weights * h).sum(), min=1e-6)
        parent_score = (G0 * G0) / (H0 + reg_lambda + 1e-8)

        synth_gains = []
        synth_hist = []
        for row in cand:
            j = int(row[0].item())
            b = int(row[1].item())
            A = probs[j][:, : b + 1].sum(dim=1)
            GL = (weights * A * g).sum()
            HL = torch.clamp((weights * A * h).sum(), min=1e-6)
            GR = G0 - GL
            HR = torch.clamp(H0 - HL, min=1e-6)
            synth_gains.append(0.5 * ((GL * GL) / (HL + reg_lambda + 1e-8) + (GR * GR) / (HR + reg_lambda + 1e-8) - parent_score) - gamma)
            synth_hist.append(torch.stack([A.sum(), GL, HL]))
        synth_gains_t = torch.stack(synth_gains)
        synth_hist_t = torch.stack(synth_hist)

        synth_pi = F.softmax(synth_gains_t / float(cfg["landscape"]["tau_gain"]), dim=0)
        loss_gain = _js(target_pi, synth_pi)
        full_delta = target_gains[best_idx] - target_gains
        synth_delta = synth_gains_t[best_idx] - synth_gains_t
        mask = torch.ones_like(full_delta, dtype=torch.bool)
        mask[best_idx] = False
        loss_margin = torch.relu(float(dc["margin_alpha"]) * full_delta[mask] - synth_delta[mask]).pow(2).mean()
        synth_parent = torch.stack([G0 / n_train, H0 / n_train])
        loss_newton = F.mse_loss(synth_parent, parent_target)
        topk = torch.argsort(target_gains, descending=True)[: min(16, len(target_gains))]
        loss_child = F.l1_loss(synth_hist_t[topk, 1:] / n_train, target_hist[topk, 1:] / n_train)
        norm_hist = torch.stack([synth_hist_t[:, 0] / m, synth_hist_t[:, 1] / n_train, synth_hist_t[:, 2] / n_train], dim=1)
        target_norm_hist = torch.stack([target_hist[:, 0] / n_train, target_hist[:, 1] / n_train, target_hist[:, 2] / n_train], dim=1)
        loss_hist = F.l1_loss(norm_hist, target_norm_hist)
        synth_class = torch.stack([(weights[labels == 0].sum() / n_train), (weights[labels == 1].sum() / n_train)])
        loss_class = F.mse_loss(synth_class, class_prior)
        loss_marg = torch.tensor(0.0)
        for j, p in enumerate(probs):
            mg = (weights[:, None] * p).sum(dim=0) / n_train
            loss_marg = loss_marg + _js(marginal_targets[j], mg)
        loss_marg = loss_marg / max(1, len(probs))
        loss_pair = torch.tensor(0.0)
        for (j, k), target in zip(landscape.pair_indices, pair_targets):
            table = torch.einsum("mb,mc,m->bc", probs[j], probs[k], weights) / n_train
            loss_pair = loss_pair + _js(target.flatten(), table.flatten())
        loss_pair = loss_pair / max(1, len(pair_targets))
        entropy = torch.stack([-(p * torch.clamp(p, 1e-8, 1.0).log()).sum(dim=1).mean() for p in probs]).mean()

        loss = (
            float(dc["weight_gain"]) * loss_gain
            + float(dc["weight_margin"]) * loss_margin
            + float(dc["weight_newton"]) * loss_newton
            + float(dc["weight_child"]) * loss_child
            + float(dc["weight_hist"]) * loss_hist
            + float(dc["weight_class"]) * loss_class
            + float(dc["weight_marginal"]) * loss_marg
            + float(dc["weight_pairwise"]) * loss_pair
            + float(dc["weight_entropy"]) * entropy
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for j, bc in enumerate(bin_counts):
                logits[:, j, bc:] = -30.0
        if step == 0 or (step + 1) % max(20, steps // 10) == 0:
            loss_trace.append({"step": step + 1, "loss": float(loss.detach()), "gain": float(loss_gain.detach()), "margin": float(loss_margin.detach())})

    with torch.no_grad():
        final_probs = [F.softmax(logits[:, j, :bc] / float(dc["temperature_end"]), dim=-1).cpu().numpy() for j, bc in enumerate(bin_counts)]
        decoded_bins = np.vstack([p.argmax(axis=1) for p in final_probs]).T
        X_syn = np.zeros((m, len(bin_counts)), dtype=np.float32)
        for j, bmeta in enumerate(meta["bins"]):
            thresholds = np.asarray(bmeta["thresholds"], dtype=np.float32)
            for a in range(m):
                bid = int(decoded_bins[a, j])
                if bid == 0:
                    val = thresholds[0] - 1e-3 if len(thresholds) else 0.0
                elif bid >= len(thresholds):
                    val = thresholds[-1] + 1e-3 if len(thresholds) else 0.0
                else:
                    val = 0.5 * (thresholds[bid - 1] + thresholds[bid])
                X_syn[a, j] = val
        y_syn = labels.cpu().numpy().astype(np.int64)
        w_syn = (F.softplus(rho) + 1e-4).cpu().numpy()
        w_syn = w_syn * (len(y) / max(1e-8, w_syn.sum()))
    return {"X": X_syn, "y": y_syn, "weights": w_syn.astype(np.float32), "trace": loss_trace}


def save_distilled(path: str | Path, distilled: dict[str, Any], meta: dict[str, Any]) -> None:
    p = ensure_dir(path)
    np.savez_compressed(p / "distilled.npz", X=distilled["X"], y=distilled["y"], weights=distilled["weights"])
    write_json(p / "metadata.json", {"rows": int(len(distilled["y"])), "trace": distilled["trace"], "dataset": meta["name"]})
