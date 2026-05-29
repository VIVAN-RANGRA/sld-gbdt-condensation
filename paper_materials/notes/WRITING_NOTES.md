# Writing Notes

## Recommended Paper Story

Most dataset distillation objectives were designed around neural-network training signals. Gradient-boosted tabular models are built through feature-threshold split gains and leaf-wise first- and second-order statistics. The paper should argue that condensation for tree-dominated tabular learning should preserve those training mechanisms directly.

## Suggested Title

Split-Gain Landscape Condensation for Gradient-Boosted Tabular Models

Use "Distillation" only if the final paper centers synthetic optimization. The current results support "Condensation" more strongly.

## Suggested Contributions

1. A split-gain landscape view of tabular data condensation for gradient-boosted trees.
2. A gain/path-aware condensed coreset method that preserves candidate split behavior, gradient/Hessian mass, teacher confidence, and tree-path occupancy.
3. A focused binary benchmark across datasets, budgets, seeds, and downstream learners.
4. Mechanism analysis through split-fidelity, ablations, and paired statistical tests.

## Suggested Main Tables

- Table 1: Main complete-method AUROC and rank.
- Table 2: Paired Wilcoxon and win/tie/loss against random and herding.
- Table 3: Budget sensitivity.
- Table 4: Cross-model transfer.
- Table 5: Split-gain fidelity.
- Appendix: dataset inventory, method coverage, all downstream metrics, all split-fidelity metrics, ablations.

## Reviewer Risk Notes

- Add stronger 2026-era tabular condensation baselines if time permits.
- Keep binary scope explicit.
- Be transparent that the synthetic variant is not the current empirical winner.
- Avoid overclaiming fidelity as a complete explanation: the fidelity improvement is real but not perfectly aligned with downstream AUROC.
