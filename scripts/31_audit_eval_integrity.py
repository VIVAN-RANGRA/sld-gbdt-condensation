from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd


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


def add_check(rows: list[dict[str, object]], area: str, check: str, status: str, detail: str) -> None:
    rows.append({"area": area, "check": check, "status": status, "detail": detail})


def status(ok: bool, warn: bool = False) -> str:
    if ok:
        return "pass"
    return "warn" if warn else "fail"


def finite_bad(series: pd.Series) -> int:
    return int((~np.isfinite(pd.to_numeric(series, errors="coerce"))).sum())


def audit_generated_metrics(paper_tables: Path, materials: Path, rows: list[dict[str, object]]) -> None:
    downstream_path = paper_tables / "all_downstream_metrics.csv"
    fidelity_path = paper_tables / "all_split_fidelity_metrics.csv"
    if downstream_path.exists():
        df = pd.read_csv(downstream_path)
        keys = ["dataset", "method", "budget", "seed", "learner"]
        dups = int(df.duplicated(keys).sum()) if set(keys).issubset(df.columns) else -1
        add_check(rows, "binary downstream", "unique metric cells", status(dups == 0), f"duplicate keys: {dups}")
        auroc_bad = int(df["auroc"].isna().sum() + ((df["auroc"] < 0) | (df["auroc"] > 1)).sum()) if "auroc" in df else -1
        add_check(rows, "binary downstream", "AUROC range", status(auroc_bad == 0), f"bad values: {auroc_bad}")
        acc_bad = int(((df["accuracy"] < 0) | (df["accuracy"] > 1)).sum()) if "accuracy" in df else -1
        add_check(rows, "binary downstream", "accuracy range", status(acc_bad == 0), f"bad values: {acc_bad}")
        ll_bad = finite_bad(df["log_loss"]) if "log_loss" in df else -1
        add_check(rows, "binary downstream", "log-loss finite", status(ll_bad == 0), f"bad values: {ll_bad}")
        key_methods = {"histdistill_refined", "histdistill_density", "gain_path_refined", "herding", "random"}
        key_budgets = {f"budget_{b}" for b in [10, 25, 50, 100, 200]}
        key_seeds = {f"seed_{s}" for s in range(5)}
        key_learners = {"xgboost", "lightgbm", "catboost", "random_forest", "mlp"}
        sub = df[
            df["method"].isin(key_methods)
            & df["budget"].isin(key_budgets)
            & df["seed"].isin(key_seeds)
            & df["learner"].isin(key_learners)
            & (df["dataset"] != "smoke_binary")
        ].copy()
        expected = 26 * len(key_budgets) * len(key_seeds) * len(key_learners)
        cov = sub.groupby("method").size().to_dict()
        complete = all(int(cov.get(m, 0)) == expected for m in key_methods)
        add_check(rows, "binary downstream", "key 5-budget grid coverage", status(complete), f"expected/method: {expected}; observed: {cov}")
    else:
        add_check(rows, "binary downstream", "metrics file exists", "fail", str(downstream_path))

    if fidelity_path.exists():
        fid = pd.read_csv(fidelity_path)
        keys = ["dataset", "method", "budget", "seed"]
        dups = int(fid.duplicated(keys).sum()) if set(keys).issubset(fid.columns) else -1
        add_check(rows, "split fidelity", "unique fidelity cells", status(dups == 0), f"duplicate keys: {dups}")
        for col in ["root_agreement", "top5_overlap", "sld_root_agreement_direct", "sld_margin_satisfied_rate"]:
            if col in fid:
                sub = fid[col].dropna()
                bad = int(((sub < 0) | (sub > 1)).sum())
                add_check(rows, "split fidelity", f"{col} range", status(bad == 0), f"bad values: {bad}; missing: {int(fid[col].isna().sum())}")
        for col in ["gain_rank_correlation", "sld_rank_correlation_direct"]:
            if col in fid:
                sub = fid[col].dropna()
                bad = int(((sub < -1) | (sub > 1)).sum())
                add_check(rows, "split fidelity", f"{col} range", status(bad == 0), f"bad values: {bad}; missing: {int(fid[col].isna().sum())}")
    else:
        add_check(rows, "split fidelity", "metrics file exists", "fail", str(fidelity_path))

    breadth_specs = [
        ("regression", materials / "tables" / "icdm_regression_cells.csv", "r2", None),
        ("multiclass", materials / "tables" / "icdm_multiclass_ovr_cells.csv", "macro_auroc", (0.0, 1.0)),
    ]
    for name, path, metric, bounds in breadth_specs:
        if not path.exists():
            add_check(rows, name, "cells file exists", "warn", str(path))
            continue
        df = pd.read_csv(path)
        keys = ["dataset", "method", "budget", "seed", "learner"]
        dups = int(df.duplicated(keys).sum()) if set(keys).issubset(df.columns) else -1
        add_check(rows, name, "unique cells", status(dups == 0), f"duplicate keys: {dups}")
        bad_finite = finite_bad(df[metric])
        add_check(rows, name, f"{metric} finite", status(bad_finite == 0), f"bad values: {bad_finite}")
        if bounds is not None:
            lo, hi = bounds
            bad = int(((df[metric] < lo) | (df[metric] > hi)).sum())
            add_check(rows, name, f"{metric} range", status(bad == 0), f"bad values: {bad}")
        if name == "regression":
            severe = int((df[metric] < -10).sum())
            add_check(rows, name, "severe negative-R2 outliers", status(severe == 0, warn=True), f"cells below -10 R2: {severe}")
        if name == "multiclass" and "learner" in df:
            lgbm_cells = int((df["learner"] == "lightgbm").sum())
            add_check(
                rows,
                name,
                "lightgbm label caveat",
                "warn" if lgbm_cells else "pass",
                f"{lgbm_cells} cells are labelled lightgbm; current code uses sklearn HistGradientBoosting fallback for this slot.",
            )


class SourceVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.uses_test_eval: list[tuple[int, str]] = []
        self.uses_val_eval: list[tuple[int, str]] = []
        self.preprocessing_before_split: list[tuple[int, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        text = ast.unparse(node) if hasattr(ast, "unparse") else ""
        if "evaluate_binary" in text and "X_test" in text:
            self.uses_test_eval.append((node.lineno, text[:180]))
        if "evaluate_multiclass" in text and "X_test" in text:
            self.uses_test_eval.append((node.lineno, text[:180]))
        if "evaluate_regression" in text and "X_test" in text:
            self.uses_test_eval.append((node.lineno, text[:180]))
        if "roc_auc_score" in text and "y_eval" in text:
            self.uses_val_eval.append((node.lineno, text[:180]))
        if "train_test_split" in text:
            self.generic_visit(node)
            return
        self.generic_visit(node)


def audit_source(rows: list[dict[str, object]]) -> None:
    source_files = [
        Path("scripts/06_train_downstream.py"),
        Path("scripts/07_eval_split_fidelity.py"),
        Path("scripts/24_run_hpo_experiment.py"),
        Path("scripts/29_run_breadth_experiments.py"),
        Path("gaindistill/data.py"),
        Path("gaindistill/models.py"),
        Path("gaindistill/landscape.py"),
        Path("gaindistill/baselines.py"),
    ]
    missing = [str(p) for p in source_files if not p.exists()]
    add_check(rows, "source", "expected files present", status(not missing), ", ".join(missing) if missing else "all present")
    for path in source_files:
        if not path.exists():
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"))
            add_check(rows, "source", f"{path} parses", "pass", "")
        except SyntaxError as exc:
            add_check(rows, "source", f"{path} parses", "fail", f"{exc}")

    downstream_text = Path("scripts/06_train_downstream.py").read_text(encoding="utf-8")
    add_check(
        rows,
        "source",
        "binary downstream evaluates held-out test",
        status("bundle.X_test" in downstream_text and "bundle.y_test" in downstream_text),
        "scripts/06_train_downstream.py",
    )
    breadth_text = Path("scripts/29_run_breadth_experiments.py").read_text(encoding="utf-8")
    add_check(
        rows,
        "source",
        "breadth evaluates held-out test",
        status("bundle.X_test" in breadth_text and "bundle.y_test" in breadth_text),
        "scripts/29_run_breadth_experiments.py",
    )
    data_text = Path("gaindistill/data.py").read_text(encoding="utf-8")
    leak_pattern = data_text.find("_infer_and_encode(") < data_text.find("train_test_split(")
    add_check(
        rows,
        "protocol",
        "main preprocessing fit after split",
        status(not leak_pattern, warn=True),
        "Current binary preprocessing encodes/imputes on the full raw frame before splitting; bins are train-only.",
    )
    breadth_leak = breadth_text.find("encode_features(") < breadth_text.find("split_bundle(")
    add_check(
        rows,
        "protocol",
        "breadth preprocessing fit after split",
        status(not breadth_leak, warn=True),
        "Current breadth preprocessing encodes/imputes before train/val/test splitting.",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-tables", default="paper_tables")
    ap.add_argument("--paper-materials", default="paper_materials")
    args = ap.parse_args()
    materials = Path(args.paper_materials)
    out_tables = materials / "tables"
    out_notes = materials / "notes"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_notes.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    audit_source(rows)
    audit_generated_metrics(Path(args.paper_tables), materials, rows)
    report = pd.DataFrame(rows)
    report.to_csv(out_tables / "eval_integrity_audit.csv", index=False)
    counts = report.groupby("status").size().reset_index(name="count").sort_values("status")
    lines = [
        "# Evaluation Integrity Audit",
        "",
        "This audit checks source-level assumptions and generated metric invariants for the current experiment package.",
        "",
        "## Status Counts",
        "",
        md_table(counts),
        "",
        "## Checks",
        "",
        md_table(report),
        "",
        "## Interpretation",
        "",
        "- `fail` means a generated artifact or code invariant is broken.",
        "- `warn` means results may still be usable, but the paper must disclose the caveat or the experiment should be rerun after a protocol fix.",
        "- The current high-risk warnings are preprocessing-before-split in binary/breadth data preparation, severe regression outliers, and the multiclass LightGBM fallback label.",
        "",
    ]
    (out_notes / "EVAL_INTEGRITY_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote evaluation integrity audit.")


if __name__ == "__main__":
    main()
