from __future__ import annotations

import argparse
from pathlib import Path

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


def read_table(root: Path, name: str) -> pd.DataFrame:
    path = root / "tables" / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-materials", default="paper_materials")
    args = ap.parse_args()
    root = Path(args.paper_materials)
    perf = read_table(root, "icdm_core_performance.csv")
    headline = read_table(root, "icdm_binary_headline_5seed_performance.csv")
    headline_pair = read_table(root, "icdm_binary_headline_5seed_pairwise.csv")
    pair = read_table(root, "icdm_core_pairwise.csv")
    fidelity = read_table(root, "icdm_core_fidelity.csv")
    ablation = read_table(root, "icdm_histdistill_ablation_pairwise.csv")
    theory = read_table(root, "icdm_sld_law_summary.csv")
    density = read_table(root, "icdm_density_pareto_summary.csv")
    hpo = read_table(root, "icdm_hpo_overall.csv")
    regression = read_table(root, "icdm_regression_performance.csv")
    regression_pair = read_table(root, "icdm_regression_pairwise.csv")
    multiclass = read_table(root, "icdm_multiclass_ovr_performance.csv")
    multiclass_pair = read_table(root, "icdm_multiclass_ovr_pairwise.csv")
    status = read_table(root, "icdm_experiment_status.csv")
    eval_audit = read_table(root, "eval_integrity_audit.csv")
    phase2_e16 = read_table(root, "phase2_e16_error_decomposition_summary.csv")
    phase2_e16_corr = read_table(root, "phase2_e16_decomposition_correlations.csv")
    phase2_e17 = read_table(root, "phase2_e17_metric_law_summary.csv")
    phase2_e18 = read_table(root, "phase2_e18_dose_response_summary.csv")
    phase2_e18_downstream = read_table(root, "phase2_e18_downstream_dose_response_summary.csv")
    phase2_e19 = read_table(root, "phase2_e19_universal_sld_map.csv")
    phase2_e19_resid = read_table(root, "phase2_e19_residual_summary.csv")
    phase2_e20 = read_table(root, "phase2_e20_training_free_selection_summary.csv")
    phase2_e21 = read_table(root, "phase2_e21_generality_scope.csv")
    phase2_e21_reg = read_table(root, "phase2_e21_regression_decomposition_summary.csv")
    phase2_e21_reg_corr = read_table(root, "phase2_e21_regression_decomposition_correlations.csv")

    sections: list[str] = [
        "# Final Results Digest",
        "",
        "This is the paper-writing digest for the current Split-Gain Landscape Distillation experiment package. "
        "The completed evidence is strongest for binary and multiclass classification; regression breadth is included but should be framed cautiously.",
        "",
        "## Binary Headline Performance",
        "",
    ]
    if not headline.empty:
        sections.append(
            md_table(
                headline[["method_label", "auroc_mean", "auroc_std", "avg_rank", "n_cells"]].rename(
                    columns={
                        "method_label": "Method",
                        "auroc_mean": "Mean AUROC",
                        "auroc_std": "Std",
                        "avg_rank": "Avg. Rank",
                        "n_cells": "Cells",
                    }
                )
            )
        )
        sections.extend(["", "## Binary Headline Pairwise", ""])
        sections.append(
            md_table(
                headline_pair[["method_label", "comparator_label", "n_pairs", "mean_diff", "wins", "ties", "losses", "wilcoxon_p"]].rename(
                    columns={
                        "method_label": "Method",
                        "comparator_label": "Comparator",
                        "n_pairs": "Pairs",
                        "mean_diff": "Mean Diff",
                        "wilcoxon_p": "Wilcoxon p",
                    }
                )
            )
        )
        sections.extend(["", "## Five-Budget Binary Core", ""])
    if not perf.empty:
        sections.append(
            md_table(
                perf[["method_label", "auroc_mean", "auroc_std", "avg_rank", "n_cells"]].rename(
                    columns={
                        "method_label": "Method",
                        "auroc_mean": "Mean AUROC",
                        "auroc_std": "Std",
                        "avg_rank": "Avg. Rank",
                        "n_cells": "Cells",
                    }
                )
            )
        )
    sections.extend(["", "## Key Pairwise Tests", ""])
    if not pair.empty:
        keep = pair[
            pair["method"].isin(["histdistill_density", "histdistill_refined"])
            & pair["comparator"].isin(["gain_path_refined", "herding", "random"])
        ]
        sections.append(
            md_table(
                keep[["method_label", "comparator_label", "n_pairs", "mean_diff", "wins", "ties", "losses", "wilcoxon_p"]].rename(
                    columns={
                        "method_label": "Method",
                        "comparator_label": "Comparator",
                        "n_pairs": "Pairs",
                        "mean_diff": "Mean Diff",
                        "wilcoxon_p": "Wilcoxon p",
                    }
                )
            )
        )
    sections.extend(["", "## Fidelity and SLD", ""])
    if not fidelity.empty:
        sections.append(
            md_table(
                fidelity[
                    [
                        "method_label",
                        "gain_rank_correlation",
                        "sld_inf_norm",
                        "sld_root_agreement_direct",
                        "sld_rank_correlation_direct",
                        "n_cells",
                    ]
                ].rename(
                    columns={
                        "method_label": "Method",
                        "gain_rank_correlation": "Retrained Gain Corr.",
                        "sld_inf_norm": "Direct SLD",
                        "sld_root_agreement_direct": "Direct Root Agree",
                        "sld_rank_correlation_direct": "Direct Rank Corr.",
                        "n_cells": "Cells",
                    }
                )
            )
        )
    sections.extend(["", "## Theory Diagnostics", ""])
    if not theory.empty:
        sections.append(
            md_table(
                theory[
                    [
                        "method_label",
                        "within_dataset_spearman_sld_vs_auroc",
                        "mean_sld_inf_norm",
                        "prop1_cells",
                        "prop1_root_agreement",
                    ]
                ].rename(
                    columns={
                        "method_label": "Method",
                        "within_dataset_spearman_sld_vs_auroc": "Within-dataset SLD/AUROC",
                        "mean_sld_inf_norm": "Mean SLD",
                        "prop1_cells": "Prop1 Cells",
                        "prop1_root_agreement": "Prop1 Agree",
                    }
                )
            )
        )
    sections.extend(["", "## Phase 2 Mechanism: Structure vs Leaf Estimates", ""])
    if not phase2_e16.empty:
        show = phase2_e16[
            phase2_e16["method"].isin(["histdistill_greedy", "histdistill_refined", "histdistill_density", "gain_path_refined", "random", "herding"])
        ]
        sections.append(
            md_table(
                show[
                    [
                        "method_label",
                        "deployed_auc",
                        "leaf_refit_auc",
                        "structure_ref_auc",
                        "structure_error",
                        "leaf_estimate_recovery",
                        "leaf_population_l1",
                        "sld_inf_norm",
                    ]
                ].rename(
                    columns={
                        "method_label": "Method",
                        "deployed_auc": "Deployed AUROC",
                        "leaf_refit_auc": "Leaf-refit AUROC",
                        "structure_ref_auc": "Structure Ref. AUROC",
                        "structure_error": "Structure Error",
                        "leaf_estimate_recovery": "Leaf Recovery",
                        "leaf_population_l1": "Leaf TV",
                        "sld_inf_norm": "SLD",
                    }
                )
            )
        )
        sections.append("")
        sections.append(
            "E16 partly falsifies the most optimistic prediction: greedy coverage has both high structure error and high leaf-population mismatch. "
            "The defensible result is not 'coverage fixes structure'; it is 'coverage alone is insufficient, and split-regret/SLD expose why.'"
        )
    if not phase2_e16_corr.empty:
        sections.extend(["", "Phase 2 E16 correlation checks:", ""])
        sections.append(md_table(phase2_e16_corr))
    sections.extend(["", "## Phase 2 Split-Regret and Universal Map", ""])
    if not phase2_e17.empty:
        sections.append(md_table(phase2_e17))
    if not phase2_e19.empty:
        top = phase2_e19[
            [
                "method_label",
                "auroc",
                "split_regret_norm",
                "split_regret_agreement",
                "line_residual_auc",
                "leaf_estimate_recovery",
                "structure_error",
                "n_cells",
            ]
        ].rename(
            columns={
                "method_label": "Method",
                "auroc": "AUROC",
                "split_regret_norm": "Split Regret",
                "split_regret_agreement": "Agreement",
                "line_residual_auc": "Line Residual",
                "leaf_estimate_recovery": "Leaf Recovery",
                "structure_error": "Structure Error",
                "n_cells": "Cells",
            }
        )
        sections.extend(["", "Universal diagnostic map:", ""])
        sections.append(md_table(top))
    if not phase2_e19_resid.empty:
        sections.extend(["", "Universal map residual checks:", ""])
        sections.append(md_table(phase2_e19_resid))
    sections.extend(["", "## Phase 2 Controlled/Practical Scope", ""])
    if not phase2_e18.empty:
        sections.append("E18 is a landscape-only controlled perturbation test; it supports the Prop. 1 threshold behavior but does not by itself prove downstream AUROC causality.")
        sections.append("")
        sections.append(md_table(phase2_e18))
    if not phase2_e18_downstream.empty:
        sections.extend(["", "E18 downstream dose-response with full-data leaf estimates:", ""])
        sections.append(md_table(phase2_e18_downstream))
    if not phase2_e20.empty:
        sections.extend(["", "Training-free selection:", ""])
        sections.append(md_table(phase2_e20))
        sections.append("")
        sections.append("E20 is weak as an argmin selection rule; keep it as a diagnostic ordering result, not a main practical claim.")
    if not phase2_e21.empty:
        sections.extend(["", "Generality scope:", ""])
        sections.append(md_table(phase2_e21))
    if not phase2_e21_reg.empty:
        sections.extend(["", "E21 regression structure-vs-leaf decomposition:", ""])
        sections.append(md_table(phase2_e21_reg))
    if not phase2_e21_reg_corr.empty:
        sections.extend(["", "E21 regression decomposition correlations:", ""])
        sections.append(md_table(phase2_e21_reg_corr))
    sections.extend(["", "## Component Ablations", ""])
    if not ablation.empty:
        sections.append(
            md_table(
                ablation[["component", "better_label", "ablation_label", "mean_delta", "wins", "ties", "losses", "wilcoxon_p"]].rename(
                    columns={
                        "component": "Component",
                        "better_label": "Reference",
                        "ablation_label": "Ablation",
                        "mean_delta": "Mean Delta",
                        "wilcoxon_p": "Wilcoxon p",
                    }
                )
            )
        )
    sections.extend(["", "## Density Pareto", ""])
    if not density.empty:
        sections.append(
            md_table(
                density[["method_label", "density_beta", "mmd2", "sld_inf_norm", "xgboost_auroc", "mlp_auroc", "n_cells"]].rename(
                    columns={
                        "method_label": "Method",
                        "density_beta": "Beta",
                        "mmd2": "MMD^2",
                        "sld_inf_norm": "SLD",
                        "xgboost_auroc": "XGB AUROC",
                        "mlp_auroc": "MLP AUROC",
                        "n_cells": "Cells",
                    }
                )
            )
        )
    sections.extend(["", "## Regression Breadth", ""])
    if not regression.empty:
        sections.append(
            md_table(
                regression[["method_label", "metric_mean", "metric_std", "avg_rank", "n_cells"]].rename(
                    columns={
                        "method_label": "Method",
                        "metric_mean": "Mean R2",
                        "metric_std": "Std",
                        "avg_rank": "Avg. Rank",
                        "n_cells": "Cells",
                    }
                )
            )
        )
        if not regression_pair.empty:
            sections.extend(["", "Regression pairwise tests:", ""])
            sections.append(
                md_table(
                    regression_pair[["method_label", "comparator_label", "n_pairs", "mean_diff", "wins", "ties", "losses", "wilcoxon_p"]].rename(
                        columns={
                            "method_label": "Method",
                            "comparator_label": "Comparator",
                            "n_pairs": "Pairs",
                            "mean_diff": "Mean Diff",
                            "wilcoxon_p": "Wilcoxon p",
                        }
                    )
                )
            )
            sections.append("")
            sections.append("Regression is a mixed/weak breadth result at budgets 25/50, especially with MLP. Use tree-learner slices cautiously; do not make it the headline.")
    sections.extend(["", "## Multiclass One-vs-Rest Breadth", ""])
    if not multiclass.empty:
        sections.append(
            md_table(
                multiclass[["method_label", "metric_mean", "metric_std", "avg_rank", "n_cells"]].rename(
                    columns={
                        "method_label": "Method",
                        "metric_mean": "Mean Macro-AUROC",
                        "metric_std": "Std",
                        "avg_rank": "Avg. Rank",
                        "n_cells": "Cells",
                    }
                )
            )
        )
        if not multiclass_pair.empty:
            sections.extend(["", "Multiclass pairwise tests:", ""])
            sections.append(
                md_table(
                    multiclass_pair[["method_label", "comparator_label", "n_pairs", "mean_diff", "wins", "ties", "losses", "wilcoxon_p"]].rename(
                        columns={
                            "method_label": "Method",
                            "comparator_label": "Comparator",
                            "n_pairs": "Pairs",
                            "mean_diff": "Mean Diff",
                            "wilcoxon_p": "Wilcoxon p",
                        }
                    )
                )
            )
    sections.extend(["", "## HPO Diagnostic", ""])
    if not hpo.empty:
        sections.append(md_table(hpo))
        sections.append("")
        sections.append("HPO is currently a weak/negative application result: the average final test-AUROC gap is small, but top-3 agreement and speedup do not meet the planned success target.")
    sections.extend(["", "## Experiment Status", ""])
    if not status.empty:
        sections.append(md_table(status))
    if not eval_audit.empty:
        sections.extend(["", "## Evaluation Integrity", ""])
        counts = eval_audit.groupby("status").size().reset_index(name="count").sort_values("status")
        sections.append(md_table(counts))
        warnings = eval_audit[eval_audit["status"].eq("warn")][["area", "check", "detail"]]
        if not warnings.empty:
            sections.extend(["", "Warnings to handle in paper/re-runs:", ""])
            sections.append(md_table(warnings))
    sections.extend(
        [
            "",
            "## Paper Claim Guidance",
            "",
            "- Strongest claim: histogram-aligned SLD gives a principled, model-aligned fidelity metric and improves SLD versus random/herding while matching legacy gain-path performance closely.",
            "- Safe binary claim: HistDistill significantly beats random and herding, but does not beat the strongest legacy gain-path reference.",
            "- Safe multiclass claim: the one-vs-rest extension beats random and herding on macro-AUROC and is competitive with k-center.",
            "- Regression claim: include as breadth/appendix only; all-learner R2 is weak at 25/50 total points, though tree learners are more favorable.",
            "- Safe theory claim: within each dataset/seed budget curve, lower SLD is associated with higher AUROC, and Prop. 1 cells show 100% root agreement.",
            "- Phase 2 E18 claim: controlled split-noise dosing obeys the Delta/2 threshold and degrades split agreement/regret monotonically; downstream AUROC moves only mildly when leaves are re-estimated on full data.",
            "- Phase 2 E21 claim: regression is a documented limit case with both large structure residuals and substantial leaf-estimate recovery; leaf population TV strongly predicts the leaf-estimate term.",
            "- Unsafe claim: do not claim modern TDColER/KIP superiority, 10x HPO speedup, or a strong regression win from the current artifacts.",
            "",
            "## Important Figures",
            "",
            "- `figures/icdm_core_budget_profile.png`",
            "- `figures/icdm_core_sld_scatter.png`",
            "- `figures/icdm_ablation_delta_violins.png`",
            "- `figures/icdm_sld_law_scatter.png`",
            "- `figures/icdm_budget_collapse_sld.png`",
            "- `figures/icdm_transfer_regret_profiles.png`",
            "- `figures/icdm_density_pareto.png`",
            "- `figures/icdm_anchor_drift.png`",
            "- `figures/icdm_sensitivity.png`",
            "- `figures/icdm_hpo_speedup_agreement.png`",
            "- `figures/icdm_binary_headline_5seed_performance.png`",
            "- `figures/icdm_regression_budget_profile.png`",
            "- `figures/icdm_multiclass_ovr_budget_profile.png`",
            "- `figures/phase2_e16_error_decomposition.png`",
            "- `figures/phase2_e17_split_regret_law.png`",
            "- `figures/phase2_e18_dose_response.png`",
            "- `figures/phase2_e18_downstream_dose_response.png`",
            "- `figures/phase2_e19_universal_sld_map.png`",
            "- `figures/phase2_e20_training_free_selection.png`",
            "- `figures/phase2_e21_regression_decomposition.png`",
            "",
            "## Main Notes",
            "",
            "- `notes/PHASE2_RESULTS.md`",
            "- `notes/PHASE2_E18_E21_FULL.md`",
            "- `notes/ICDM_EXPERIMENT_AUDIT.md`",
            "- `notes/EVAL_INTEGRITY_AUDIT.md`",
            "- `notes/ICDM_CORE_UPDATE.md`",
            "- `notes/ICDM_ABLATION_UPDATE.md`",
            "- `notes/ICDM_THEORY_DIAGNOSTICS.md`",
            "- `notes/ICDM_DENSITY_PARETO.md`",
            "- `notes/ICDM_ANCHORING_DIAGNOSTIC.md`",
            "- `notes/ICDM_SENSITIVITY_DIAGNOSTIC.md`",
            "- `notes/ICDM_HPO_DIAGNOSTIC.md`",
            "- `notes/ICDM_BINARY_HEADLINE_5SEED_UPDATE.md`",
            "- `notes/ICDM_REGRESSION_UPDATE.md`",
            "- `notes/ICDM_MULTICLASS_OVR_UPDATE.md`",
            "",
        ]
    )
    (root / "FINAL_RESULTS.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {root / 'FINAL_RESULTS.md'}")


if __name__ == "__main__":
    main()
