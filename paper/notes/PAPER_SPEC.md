# PAPER_SPEC.md — Authoritative Blueprint (READ THIS FIRST)

This is the single source of truth for the paper. The writing agent and the assembly agent MUST follow it. It encodes the locked framing, the honest claim map, the math, the section plan, and the exact figure/table mapping. **Every number you write in the paper must be re-verified against the named CSV under `paper_materials/tables/`. If a number here disagrees with the CSV, the CSV wins — flag it.**

---

## 0. Meta

- **Venue:** ICDM 2026, **Applied / Applied-Data-Science style track**. IEEE 2-column conference format (`\documentclass[conference]{IEEEtran}`).
- **Hard limit:** 10 pages including references. Aim for 9 pages of body + references fitting in 10.
- **Paper type:** *measurement and analysis* paper. The deliverables are (a) a training-free diagnostic and (b) a comparative analysis / benchmark of condensation methods for tabular GBDTs. **We do NOT claim a new state-of-the-art condensation method.**
- **Title (locked):** `Coverage Is Not Accuracy: A Split-Landscape Analysis of Data Condensation for Gradient-Boosted Trees`
- **Keywords:** data condensation; gradient-boosted decision trees; coreset selection; tabular data; training-free model diagnostics
- **Abstract (locked, number-free — use verbatim, light polish only):**

> Data condensation replaces a large training set with a small surrogate so that a model retrained on it behaves like one trained on the full data. For gradient-boosted decision trees, the dominant model for tabular problems, many coreset and distillation methods now exist, but they are compared almost exclusively by the accuracy of the retrained model. Accuracy is silent about mechanism: it does not say whether a condensed set preserves the structure the booster would have learned or merely covers the input space, and it offers no way to compare methods without paying for full retraining.
>
> We take a measurement-first view. A gradient-boosted tree commits to its structure through its split-selection landscape, which can be compared between full and condensed data before any model is trained. We formalize this comparison as the Split-Landscape Discrepancy and use it to study a broad slate of condensation methods — classical coresets, gradient- and gain-based selectors, and distribution-matching distillation — under a common protocol across many tabular datasets.
>
> The analysis produces three results. A single training-free landscape score orders the methods by their eventual accuracy, enabling method selection without retraining. Decomposing error into structural and leaf-estimate components explains a counterintuitive failure: a method that provably maximizes data coverage is among the worst for accuracy, because coverage and structural fidelity are different objectives. Finally, learned tree structure tolerates landscape perturbation up to a sharp threshold, which explains why effective condensers succeed and bounds how far data can be reduced. We also delimit where the lens does not extend, including regression and hyperparameter transfer, and release the benchmark and diagnostic as reusable tools.

---

## 1. The Honest Claim Map (NON-NEGOTIABLE)

### WE CAN claim (each backed by a CSV):
- **C1 (Universal ordering / headline).** At the method level, a single training-free landscape score (split-regret / SLD) ranks the 17 condensers by their eventual downstream AUROC, slope −0.91, Spearman **−0.87** (`phase2_e19_universal_sld_map.csv`, `phase2_e19_residual_summary.csv`, 17 methods, 156 cells).
- **C2 (Within-dataset SLD law).** Within a dataset, lower SLD predicts higher AUROC: within-dataset Spearman(SLD∞, AUROC) ≈ **−0.56 to −0.64** across methods (`icdm_sld_law_summary.csv`). NOTE the *pooled* Spearman is near zero / weakly positive — this is **Simpson's paradox** from cross-dataset scale confounding; we must report and explain this honestly, it is a feature of the analysis, not hidden.
- **C3 (Coverage ≠ accuracy).** A provably (1−1/e)-optimal *coverage* selector (HistDistill-Greedy) attains the best coverage (f(S)/proxy upper bound = 1.0; coverage ratio 0.37→0.54) yet the **worst** downstream accuracy (AUROC **0.6285** vs ~0.74 for refined/density) (`icdm_selector_coverage_summary.csv`, `icdm_histdistill_ablation_performance.csv`, `phase2_e19_universal_sld_map.csv`).
- **C4 (Structure vs leaf decomposition).** Total error decomposes into a *structure* component (governed by SLD) and a *leaf-estimate* component (governed by population representativeness, measured by `leaf_population_l1`). HistDistill-Greedy fails on **structure** (structure_error 0.0945, SLD∞ 76.86 — both worst); gradient-sampling fails on **leaves** (leaf_estimate_recovery 0.374). `leaf_population_l1`→`leaf_estimate_recovery` Spearman pooled 0.41 / within 0.37 (`phase2_e16_error_decomposition_summary.csv`, `phase2_e16_decomposition_correlations.csv`).
- **C5 (Robustness threshold).** Injecting calibrated split-gain noise: structure and downstream AUROC are **flat up to a sharp threshold** (dose ≤ 0.5 → agreement 1.0, AUROC unchanged), then decline monotonically (AUROC 0.8153→0.8077 at dose 2.0; at dose 2.0 ~41% of splits corrupted yet <1% AUROC lost). This is the empirical face of Proposition 1's margin condition (`phase2_e18_downstream_dose_response_summary.csv`, 1,872 cells = 26 datasets × 3 seeds × 3 repeats × 8 doses).
- **C6 (Fidelity ranking of real methods).** HistDistill variants achieve the lowest SLD∞ (≈16.7–25) vs published baselines (Random 27.7, Herding 34.6; broader set: MVS 40.0, CRAIG 44.9, GOSS 49.7, GradMatch 68.0) (`icdm_core_fidelity.csv`, `phase2_e16_error_decomposition_summary.csv`).
- **C7 (Accuracy parity, stated as parity NOT a win).** On binary GBDT, HistDistill-Refined/Density (AUROC 0.7425) are **on par with** the legacy gain-path method (0.7430; difference p≈0.039, i.e. a statistical tie) and **above** Herding (0.7306) and Random (0.7244) (`icdm_core_performance.csv`, `icdm_core_pairwise.csv`). Frame as "matches the strongest prior selector while being the most faithful," NOT "beats SOTA."
- **C8 (Practical payoff, modest and honest).** Condensation yields ~3–4× median retrain speedup at these budgets (`icdm_runtime_summary.csv`); the diagnostic itself needs **no retraining**, so it screens methods/budgets essentially for free.

### WE MUST NOT claim:
- ❌ A new SOTA condensation method / that HistDistill "beats" everything. (It ties its own legacy variant.)
- ❌ That the theory (Lemma/Props/Theorem) is novel — it is standard GBDT/coreset folklore; we present it only to *ground* the diagnostic. Say so.
- ❌ N-independence of raw SLD∞ (it scales with N — DROP this; only normalized/within-dataset use is valid).
- ❌ That a density term fixes MLP/deep-model transfer (β sweep moves MLP only ~+0.008 — DROP).
- ❌ That condensed-set HPO recovers full-data tuning (it does not — top-3 Jaccard 0.042; report as a documented limitation).
- ❌ Strong pooled SLD→accuracy correlation (it's confounded; only the within-dataset and method-level orderings hold).

### Limitations we state plainly (this HELPS in the applied track):
- Regression: condensation is **structure-dominated** and no condenser fixes it (deployed R² 0.1624 vs leaf_refit 0.2796 vs structure_ref 0.6266; `phase2_e21_regression_decomposition_summary.csv`). The lens diagnoses the failure but does not cure it.
- HPO transfer: weak (top-3 Jaccard 0.042, top-1 0.083, Spearman 0.145, median speedup 1.57×, test-AUC gap 0.00125; `icdm_hpo_overall.csv`).
- Training-free *selection* of the single best method is still weak (top-3 0.436; `phase2_e20_training_free_selection_summary.csv`) — the score *orders* methods well but is not yet a turnkey selector.

---

## 2. Notation and Math (state formally; label as standard where standard)

Let training data be $D=\{(x_i,y_i)\}_{i=1}^N$. A GBDT fits additive trees by, at each node, choosing the split that maximizes the regularized gain. With first/second-order statistics $g_i=\partial_{\hat y}\ell$, $h_i=\partial^2_{\hat y}\ell$, the gain of splitting a node's instance set $I$ into $I_L,I_R$ is the standard XGBoost criterion:

$$\mathcal{G}(I_L,I_R)=\tfrac12\!\left[\frac{(\sum_{I_L}g_i)^2}{\sum_{I_L}h_i+\lambda}+\frac{(\sum_{I_R}g_i)^2}{\sum_{I_R}h_i+\lambda}-\frac{(\sum_{I}g_i)^2}{\sum_{I}h_i+\lambda}\right]-\gamma .$$

**Histogram sufficiency (Lemma 1, standard).** For histogram GBDTs, each feature $f$ is binned into $B$ bins; the only statistics needed to evaluate every candidate split on $f$ are the per-bin sums $\big(\sum g,\ \sum h\big)$. Hence the per-node *split-gain landscape* is fully determined by the feature×bin grid of $(g,h)$ aggregates. Define the landscape $\Gamma_D \in \mathbb{R}^{F\times B}$ as the gain each candidate (feature, bin-threshold) would receive on $D$.

**Split-Landscape Discrepancy (SLD) — the diagnostic (our framing, mechanism is standard).**
$$\mathrm{SLD}_p(D,D') \;=\; \big\lVert \Gamma_D - \Gamma_{D'} \big\rVert_p,$$
computed at the root (and optionally averaged over early nodes). We use $p=\infty$ (`sld_inf_norm`) and the rank-based *split-regret* variant (normalized loss in gain from selecting $D'$'s argmax split under $D$'s landscape). **Crucially SLD is computed from histogram statistics only — no model is trained on $D'$.**

**Proposition 1 (margin preservation, standard perturbation argument).** Let $\Delta$ be the gain margin between the best and runner-up split at a node on $D$. If $\mathrm{SLD}_\infty(D,D') < \Delta/2$, the argmax split is preserved under $D'$. *Empirics:* in margin-qualifying cells, root agreement = **1.0** (`icdm_sld_law_summary.csv`, `prop1_root_agreement`); dose-response confirms the threshold (C5).

**Proposition 2 (submodular coverage, Nemhauser et al.).** The split-coverage objective (fraction of full-data high-gain splits whose threshold is reproducible from $D'$) is monotone submodular; greedy selection achieves a $(1-1/e)$ guarantee. *This is exactly the selector (HistDistill-Greedy) that wins coverage but loses accuracy — the engine of C3.*

**Theorem 1 (budget, sketch).** $\tilde O(1/\varepsilon^2)$ samples suffice to drive $\mathrm{SLD}$ below $\varepsilon$ w.h.p. (standard concentration of the per-bin $(g,h)$ sums). Keep the proof short, in an appendix or a compressed paragraph; emphasize it only motivates *why* small condensed sets can preserve structure.

> Writing rule: explicitly say Lemma 1 / Prop 2 / Thm 1 are adaptations of known results; the novelty is using $\Gamma$ as a *measurement instrument* for condensation, not the theorems.

---

## 3. Methods compared (the 17 condensers) — Table 1 taxonomy

Group them; cite each (keys come from `references.bib`):
- **Geometry / classical coresets:** Random, Herding, K-center, Importance/sensitivity sampling.
- **GBDT-native subsampling:** GOSS (LightGBM), MVS (minimal variance sampling).
- **Gradient/score-based selection:** GraNd, EL2N, CRAIG-style, GradMatch-style, Gradient sampling.
- **Distribution matching distillation:** DistributionMatching (DM).
- **Landscape/gain-based (ours, the HistDistill family):** HistDistill-Greedy (coverage-optimal), HistDistill-Refined, HistDistill-Density, gain sketch, gain-path sketch, legacy gain-path refined.

Make clear HistDistill-Greedy is the (1−1/e) coverage selector used as the C3 foil; HistDistill-Refined/Density are the practical variants.

---

## 4. Datasets & protocol
- Binary core grid: results pooled to **3,250 cells** for headline performance (`icdm_core_performance.csv`); fidelity over 650 cells.
- Broad robustness (E18): **26 binary datasets** × 3 seeds × 3 repeats × 8 dose levels = 1,872 cells.
- Multiclass (OVR, 5 datasets), regression (5 datasets), HPO (12 datasets).
- Budgets (condensed-set sizes) swept; report budget curves.
- Dataset inventory → appendix table from `dataset_inventory.csv` / `icdm_processed_dataset_audit.csv`.
- Deployed model = histogram GBDT; transfer tested to XGBoost/LightGBM/CatBoost/RandomForest/MLP (`icdm_transfer_regret.csv`).
- Metric definitions: AUROC (binary), macro-AUROC (multiclass), R² (regression); SLD∞, split-regret, structure_error, leaf_estimate_recovery, leaf_population_l1 — define each precisely in §setup.

---

## 5. Section plan (target lengths; ~9 pp body)

1. **Introduction (1–1.25 pp).** Practitioner pain point first: condensed tabular data is used to cut retrain/HPO/sharing cost, but methods are chosen blind — accuracy after retraining is the only yardstick and it requires the very retraining you wanted to avoid, and it hides *why* a set works. Contributions as a bullet list (the diagnostic; the comparative analysis of 17 methods; the three insights C1/C3/C5; honest scope). End with "we release benchmark + diagnostic."
2. **Related work (0.75–1 pp).** From `LITERATURE_REVIEW.md`. Four short paragraphs: dataset distillation/condensation; coreset selection; GBDTs & native subsampling; data-centric eval / why tabular trees. Position: prior art optimizes/evaluates by retrained accuracy; none offers a training-free fidelity instrument for tree structure.
3. **Background & the SLD diagnostic (1.5 pp).** §2 math. Include the **TikZ pipeline flowchart (Fig. 1)**. State Lemma 1, SLD def, Prop 1, Prop 2, Thm 1 (compressed), each with one line of intuition. Algorithm box: "Compute SLD (training-free)".
4. **Experimental setup (0.5 pp).** Datasets, methods (Table 1 taxonomy), budgets, metrics, hardware, reproducibility pointer.
5. **Results (3.5–4 pp)** — one subsection per insight, each = claim → figure/table → significance → practitioner takeaway (NO table dumps):
   - 5.1 *Does the landscape predict accuracy?* → C1 universal ordering (Fig: `phase2_e19_universal_sld_map.png`) + C2 within-dataset law incl. the Simpson caveat (Fig: `icdm_sld_law_scatter.png`). Table 2 = headline performance+fidelity.
   - 5.2 *Coverage is not accuracy.* → C3 (Fig: `icdm_selector_coverage.png` or `icdm_ablation_sld_tradeoff.png`) + C4 decomposition (Fig: `phase2_e16_error_decomposition.png`, Table 3 decomposition).
   - 5.3 *How much can structure be perturbed?* → C5 dose-response (Fig: `phase2_e18_downstream_dose_response.png`, Table 4). Tie to Prop 1.
   - 5.4 *Practical payoff & transfer.* → C7 parity, C8 speedup (Fig: `icdm_runtime_speedup.png`), cross-learner transfer (`icdm_transfer_regret_profiles.png`), budget curves (`icdm_core_budget_profile.png` or `paper_budget_curves_ci.png`).
6. **Where the lens stops: scope & limitations (0.75 pp).** Regression structure-dominated failure (Fig: `phase2_e21_regression_decomposition.png`); HPO transfer weak (Fig: `icdm_hpo_speedup_agreement.png`); training-free single-best selection still weak. Table 5 = generality scope. Frame as honest scoping that *the diagnostic itself surfaced*.
7. **Conclusion (0.4 pp).** Restate: measure mechanism, not just accuracy; coverage≠accuracy; release artifacts.

---

## 6. Figures to include (use existing PNGs in `paper_materials/figures/`; copy into `paper/figures/`)

| Fig | File | Section | Caption focus / significance |
|---|---|---|---|
| 1 | **NEW TikZ flowchart** | §3 | Pipeline: full data → 17 condensers → condensed set; SLD read off histogram landscape (training-free) → predicts retrain fidelity; contrast with the expensive retrain-and-evaluate loop. |
| 2 | `phase2_e19_universal_sld_map.png` | §5.1 | Method-level: one landscape score orders all 17 condensers by AUROC (Spearman −0.87). HEADLINE. |
| 3 | `icdm_sld_law_scatter.png` | §5.1 | Within-dataset SLD↓ → AUROC↑; annotate the Simpson caveat. |
| 4 | `icdm_selector_coverage.png` | §5.2 | Coverage-optimal greedy dominates coverage but not accuracy. |
| 5 | `phase2_e16_error_decomposition.png` | §5.2 | Structure vs leaf error; greedy=structure failure, gradient-sampling=leaf failure. |
| 6 | `phase2_e18_downstream_dose_response.png` | §5.3 | Flat-then-decline threshold; ~41% splits corrupted → <1% AUROC. |
| 7 | `icdm_runtime_speedup.png` | §5.4 | Retrain speedup. |
| 8 | `icdm_transfer_regret_profiles.png` | §5.4 | Cross-learner transfer regret. |
| 9 | `phase2_e21_regression_decomposition.png` | §6 | Regression is structure-dominated; no condenser fixes it. |
| 10 | `icdm_hpo_speedup_agreement.png` | §6 | HPO transfer weak. |

Keep to ~8–10 figures given 10-page limit; if tight, fold Figs 3/8/10 into multi-panel or drop to appendix. **Do NOT shrink axis text below readable.** If any PNG's text is too small at column width, note it for regeneration rather than embedding illegibly.

## 7. Tables to build (booktabs; NOT dumps — these ~5 only)

| Tab | Source CSV | Content |
|---|---|---|
| 1 | (authored) | Method taxonomy: 17 methods × {family, selection signal, citation}. |
| 2 | `icdm_core_performance.csv` + `icdm_core_fidelity.csv` | Headline: AUROC mean±std, avg rank, SLD∞, median speedup — for the core methods. |
| 3 | `phase2_e16_error_decomposition_summary.csv` | Decomposition: deployed/leaf_refit/structure_ref AUROC, structure_error, leaf_estimate_recovery, leaf_population_l1 (condense to ~8 rows). |
| 4 | `phase2_e18_downstream_dose_response_summary.csv` | Dose vs AUROC, agreement, split_regret (8 rows). |
| 5 | `phase2_e21_generality_scope.csv` | Multiclass / regression / HPO scope with honest notes. |

Appendix table: dataset inventory.

## 8. Plots/graphics to GENERATE (tell the user)
- **Only one new graphic is required: the TikZ pipeline flowchart (Fig. 1).** Build it in TikZ directly in LaTeX (no external tool).
- All other figures already exist as PNGs. If, when compiled at column width, any existing PNG is illegible, list exactly which ones need regeneration and why — do not silently embed unreadable figures.

## 9. Reproducibility (for the ICDM checklist, which we answered Yes to)
- Add a short "Reproducibility" paragraph: datasets are public (OpenML); code/diagnostic released; report seeds/repeats; hardware line; HP grid + selection rule. Tables/figures map to commands in the released README.

## 10. Style rules (the WRITING_STYLE_GUIDE.md will refine these)
- Problem-first, practitioner voice; "we/our"; present tense for findings.
- Every experiment: claim → evidence → **significance** → **takeaway**. No naked tables.
- Use `\eqref`, `cleveref` (`\Cref`), `booktabs`, `siunitx` for numbers, `subcaption` for panels.
- Define every symbol/metric on first use. Use math for the gain, SLD, Prop 1 threshold.
- Be explicit and unembarrassed about limitations; they are part of the contribution.

---

## 11. Shared LABEL REGISTRY (both writer and assembler MUST use these exact labels)

The **writer** references everything via `\Cref{...}`/`\eqref{...}` and writes the prose, equations, and the algorithm box ONLY. The **assembler** creates ALL floats (figures, tables, TikZ flowchart) with exactly these `\label`s and writes their captions from the caption-focus notes in §6/§7.

**Figures:** `fig:pipeline` (TikZ, §3) · `fig:universal` (phase2_e19_universal_sld_map.png, §5.1) · `fig:sldlaw` (icdm_sld_law_scatter.png, §5.1) · `fig:coverage` (icdm_selector_coverage.png, §5.2) · `fig:decomp` (phase2_e16_error_decomposition.png, §5.2) · `fig:dose` (phase2_e18_downstream_dose_response.png, §5.3) · `fig:runtime` (icdm_runtime_speedup.png, §5.4) · `fig:transfer` (icdm_transfer_regret_profiles.png, §5.4) · `fig:regression` (phase2_e21_regression_decomposition.png, §6) · `fig:hpo` (icdm_hpo_speedup_agreement.png, §6)

**Tables:** `tab:methods` (taxonomy, §4) · `tab:headline` (perf+fidelity, §5.1) · `tab:decomp` (E16, §5.2) · `tab:dose` (E18, §5.3) · `tab:scope` (E21 generality, §6) · `tab:datasets` (appendix)

**Equations:** `eq:gain` (split gain) · `eq:sld` (SLD definition) · `eq:prop1` (margin threshold)

**Algorithm:** `alg:sld` (training-free SLD computation).

## 12. C2TC positioning (IMPORTANT — closest related work)

`xu2025c2tc` (C²TC, ICDE 2026, Xu et al.) is the **first training-free tabular dataset *condensation* method** (class-adaptive clustering). It is the nearest neighbor to our work and MUST be discussed in Related Work and contrasted clearly:
- C²TC is a **method that produces a condensed set without training**; ours is a **training-free measurement instrument that audits/predicts the fidelity of *any* condensed set** and a comparative analysis of 17 condensers.
- C²TC is model-agnostic clustering; SLD is **GBDT-mechanism-specific** (the split-gain landscape) — it explains *why* a condensed set preserves tree structure, not just how to build one.
- They are complementary: SLD could be used to audit C²TC's output. Say this; do not overclaim competition.

## 13. Minor bib note for assembler
- `kang2025tabdistill` is typed `@inproceedings` but TMLR is a journal — change to `@article` with `journal = {Transactions on Machine Learning Research}` during assembly. Otherwise the bib is verified and clean.
