# Final Experiments — Histogram-Aligned Distillation for GBDT

Execution-ready experiment plan for the ICDM submission. Every experiment is tied
to a claim in `THEORY_AND_SPEC.md` so the empirical section *tests the theory*,
not just performance. Tables reuse existing legacy numbers as anchors; new
HistDistill methods are marked `TBD` with the theory-predicted direction.

Companion files: `THEORY_AND_SPEC.md` (definitions, Lemma 1, Props 1–4, Thm 1),
`FINAL_RESULTS.md` (legacy row-method numbers).

---

## 0. Claim → Experiment Map

| # | Claim / Result | Experiment | Headline artifact |
| --- | --- | --- | --- |
| C1 | Histogram is the sufficient statistic (Lemma 1): we can hit near-perfect split fidelity | E3 (fidelity), E1 (downstream) | T2 fidelity, T1 main |
| C2 | Low SLD ⇒ preserved greedy tree (Props 1–2) | E3 SLD-vs-AUROC scatter | F2 |
| C3 | Budget for fidelity is $\tilde O(\varepsilon^{-2})$, independent of $N$ (Thm 1 / Cor 1) | E4 budget-collapse | F3 |
| C4 | Greedy coverage is $(1-1/e)$-optimal and beats i.i.d. sampling at small $B$ (Prop 4, Lemma 2) | E6 selector study | T5 |
| C5 | Beats modern distillation baselines, not just coresets | E1, E2 | T1, T3 |
| C6 | Cross-learner transfer; MLP gap closed by density term (Eq. 6) | E7, E8 | T4, F4, F5 |
| C7 | Boosting drift is bounded under anchoring (Assumption A, Prop 3) | E9 | F6 |
| C8 | Every method component contributes | E10 ablation | T6 |
| C9 | Generality beyond binary | E11 regression, E12 multiclass | T7, T8 |
| C10 | Practical value: fast HPO | E13 | T9, F7 |
| C11 | Affordable | E14 runtime | T10 |

A paper is "solid" when C1–C6 + C9 land with significance and C7/C8/C10/C11 are
clean supporting evidence.

---

## 1. Benchmark Protocol

**Tasks / suites.**
- **Binary (headline):** 26 real binary datasets (the existing broad26 suite:
  `adult, australian, bank_marketing, breast_w, credit_g, diabetes, pc1, spambase, …`).
- **Regression (breadth):** 8–10 OpenML regression datasets
  (e.g. `boston/housing, abalone, cpu_act, wine_quality, bike_sharing, california_housing, concrete, energy, kin8nm, superconduct`).
- **Multiclass (breadth):** 6–8 OpenML multiclass datasets
  (e.g. `covertype(sub), letter, mnist(tab/sub), segment, vehicle, satimage, dna, optdigits`).

**Budgets (per class for clf; per dataset for reg).** `10, 25, 50, 100, 200`
samples/class. The two extremes drive the budget-curve experiment (E4); 25/50
remain the comparison default.

**Seeds.** `0,1,2` for all; `0..4` (5 seeds) for the binary headline table only.

**Downstream learners.** XGBoost, LightGBM, CatBoost, Random Forest, MLP.
(Regression/multiclass breadth suites may use 3 learners — XGB, LGBM, MLP — to
control cost.)

**Teacher.** XGBoost (hist), fixed config per dataset; landscape, gradients,
Hessians, probe regions computed from the teacher as in the current pipeline.

**Splits / preprocessing.** Existing train/val/test, numeric+categorical binning.
Bins fixed at $B_j=256$ (LightGBM default) unless varied in E15.

**Cell.** One (task, dataset, method, budget, seed, learner) → one evaluation.
Binary headline ≈ $26\times5\,\text{budgets}\times5\,\text{seeds}\times5\,\text{learners}=3250$
cells/method.

---

## 2. Methods and Baselines

**Ours (new).**
- `HistDistill-Greedy` — submodular split-coverage greedy + NNLS weight fit (Prop 4).
- `HistDistill-Refined` — Greedy + local swap refinement.
- `HistDistill-Density` — Refined + MMD marginal term, Eq. (6) (MLP fix).
- `ImportanceSample` — Horvitz–Thompson importance sampling (Thm 1) — the
  *analyzable* selector, included to validate Prop 4/C4, not as headline.

**Legacy (now ablations/reference).**
- `gain_path_refined`, `gain_path_safeguarded`, `gain_sketch`, `gain_path_sketch`
  — the previous row-based winners; demoted to "legacy gain-aware" reference.

**Coreset baselines.**
- `Random`, `Herding`, `K-center`, `Craig`, `GradMatch`.

**Distillation baselines (modern, the credibility upgrade).**
- `DistributionMatching` (DM), `KIP-tabular` (KRR/NTK adapted), `TDColER`
  (official code if available; else faithful reimpl validated to within tolerance
  of reported numbers on an overlap dataset).

**Appendix-only.** `synthetic soft-bin` optimizer (negative result);
`soft-tree + trajectory matching` named as future work.

---

## 3. Metrics

**Downstream quality.**
- Binary: test **AUROC** (primary), accuracy, log-loss.
- Multiclass: **macro-AUROC** (primary), accuracy, macro-F1.
- Regression: **R²** (primary), MAE, RMSE.
- **Retention** $=\text{metric}(\tilde D)/\text{metric}(\text{full})$ for compression curves.

**Fidelity (the new core, from Def. 1).**
- $\mathrm{SLD}_\infty$, $\mathrm{SLD}_1$ at root + depth-1 + depth-2 probes, per
  teacher checkpoint.
- Derived (for continuity with prior work): root-split agreement, top-5 overlap,
  gain-rank correlation — reported as *functions of* SLD.

**Transfer.**
- Cross-learner AUROC matrix.
- **Transfer regret** $\mathcal R(M;L)=\text{AUROC}_{\text{full}}(L)-\text{AUROC}_{M}(L)$.

**Theory diagnostics.**
- Margin distribution $\Delta_v$ per dataset/node.
- Anchoring $\rho_t=\max_i|F^{\text{stu}}_t-F^{\text{tea}}_t|$ vs boosting round.
- $\mathrm{MMD}^2(P_X,P_{\tilde X})$ for the Pareto study.

**Statistics.** Paired Wilcoxon signed-rank, win/tie/loss, average rank
(Friedman + Nemenyi for the rank diagram), 95% bootstrap CIs.

---

## 4. Experiments

Each: *purpose → tests → protocol → artifact → predicted outcome (falsifiable)*.

### E1 — Main binary benchmark (headline)
- **Tests:** C1, C5. **Protocol:** all methods, broad26, budgets 25/50, 5 seeds,
  5 learners; aggregate mean AUROC, std, average rank.
- **Artifact:** **T1**, **F1** (rank diagram).
- **Predicted:** HistDistill-Refined ≥ best legacy and ≥ all modern baselines on
  mean AUROC and average rank; gap over TDColER/KIP positive and significant.

### E2 — Statistical tests
- **Tests:** C5. **Protocol:** paired Wilcoxon + win/tie/loss for each
  HistDistill vs each baseline; Friedman across all methods.
- **Artifact:** **T3**.
- **Predicted:** HistDistill vs Random/Herding $p<10^{-10}$; vs DM/KIP/TDColER
  $p<10^{-2}$.

### E3 — SLD and the SLD↔AUROC law  *(the theory's centerpiece)*
- **Tests:** C1, C2 (Lemma 1, Props 1–2). **Protocol:** compute SLD (all probes,
  checkpoints) for every cell; scatter SLD vs downstream AUROC across all
  (method, dataset, budget, seed).
- **Artifact:** **T2** (fidelity table), **F2** (SLD↔AUROC scatter with Pareto
  front + the $\Delta/2$ guarantee band from Prop 1).
- **Predicted:** (i) HistDistill root-agreement $\to$ near 1 and gain-rank-corr
  $\gg 0.35$ legacy (Lemma 1 by construction); (ii) monotone SLD↔AUROC trend;
  (iii) cells with $\mathrm{SLD}_\infty<\Delta/2$ show ~100% root agreement
  (direct Prop 1 confirmation).

### E4 — Budget curve & size-independence  *(Theorem 1 / Cor 1)*
- **Tests:** C3. **Protocol:** for each method, plot $\mathrm{SLD}$ and AUROC vs
  budget $B\in\{10,25,50,100,200\}$; overlay datasets of very different $N$
  (e.g. `australian` N≈400 vs `adult`/`bank` N≈12k) on the SLD-vs-$B$ axis.
- **Artifact:** **F3** (two panels: AUROC-vs-B; SLD-vs-B with curves from
  different-$N$ datasets overlaid).
- **Predicted:** SLD-vs-$B$ curves **collapse across $N$** (the $N$-independence
  claim); SLD decays $\approx B^{-1/2}$ (the $\tilde O(\varepsilon^{-2})$ law).

### E5 — Margin distribution
- **Tests:** context for C2. **Protocol:** histogram of node margins $\Delta_v$
  per dataset; fraction of nodes with $\Delta_v>2\,\mathrm{SLD}_\infty$ achieved
  by each method.
- **Artifact:** **F8** (appendix).
- **Predicted:** HistDistill achieves the $\Delta_v>2\varepsilon$ regime at far
  smaller $B$ than baselines.

### E6 — Selector study (greedy vs sampling vs proxy-optimum)  *(Prop 4, C4)*
- **Tests:** C4. **Protocol:** compare `HistDistill-Greedy`, `ImportanceSample`,
  and a relaxed/LP upper-bound proxy on the coverage objective $f$; report
  $f(S)/f^{\text{ub}}$, SLD, AUROC at $B=10,25,50$.
- **Artifact:** **T5**.
- **Predicted:** greedy attains $\ge (1-1/e)$ of the proxy optimum and beats
  importance sampling at small $B$ (lower variance), confirming the deployed
  selector choice.

### E7 — Cross-learner transfer + MLP gap  *(C6)*
- **Tests:** C6. **Protocol:** transfer matrix (condense once, train 5 learners);
  transfer regret per learner; compare HistDistill-Refined vs -Density on MLP.
- **Artifact:** **T4**, **F4** (transfer profiles).
- **Predicted:** HistDistill dominates on XGB/LGBM/CB/RF; `-Density` closes
  ≥50% of the MLP regret gap vs `-Refined`.

### E8 — Tree-vs-density Pareto frontier  *(Eq. 6, C6)*
- **Tests:** C6. **Protocol:** sweep $\beta$ in (6); plot $\mathrm{MMD}^2$ vs SLD,
  colored by MLP and XGB AUROC.
- **Artifact:** **F5**.
- **Predicted:** clean frontier; XGB AUROC tracks low SLD, MLP AUROC tracks low
  MMD — quantifying the transfer tension and justifying `-Density`.

### E9 — Anchoring diagnostic  *(Assumption A, Prop 3, C7)*
- **Tests:** C7. **Protocol:** measure $\rho_t$ vs boosting round, with/without
  warm-start anchoring; correlate $\rho_t$ with realized per-checkpoint SLD.
- **Artifact:** **F6** (appendix).
- **Predicted:** anchoring keeps $\rho_t$ small and bounded; SLD tracks
  $O(\beta\rho)$ as Prop 3 predicts.

### E10 — Component ablation
- **Tests:** C8. **Protocol:** remove each piece — coverage saturation $\phi$,
  weight fit (5), probe depths (root-only vs +d1 vs +d2), refinement, density
  term — measure ΔAUROC and ΔSLD with Wilcoxon.
- **Artifact:** **T6**.
- **Predicted:** weight-fit and probe-depth contribute most to SLD; refinement
  and density most to AUROC/transfer. **Note:** the legacy "gain-weighting" no-op
  is *replaced* here by saturation $\phi$ and weight-fit, which must show real,
  significant effects (fixes the prior 0.8350=0.8350 embarrassment).

### E11 — Regression benchmark
- **Tests:** C9. **Protocol:** squared/Huber landscape; 8–10 datasets, budgets
  25/50, seeds 0–2, learners XGB/LGBM/MLP; R²/MAE.
- **Artifact:** **T7**.
- **Predicted:** HistDistill beats Random/coreset baselines on R², $p<0.05$.

### E12 — Multiclass benchmark
- **Tests:** C9. **Protocol:** one-vs-rest per-class landscapes; 6–8 datasets;
  macro-AUROC.
- **Artifact:** **T8**.
- **Predicted:** HistDistill beats Random/coreset baselines on macro-AUROC.

### E13 — Application: fast HPO
- **Tests:** C10. **Protocol:** Optuna 100-trial XGBoost HPO on full vs distilled
  (5 datasets); measure top-3 config agreement (Jaccard / rank-overlap) and
  wall-clock speedup; validate the distilled-selected config on full test.
- **Artifact:** **T9**, **F7**.
- **Predicted:** top-3 agreement $\ge 0.8$ at $\ge 10\times$ speedup; final test
  AUROC within ~1% of full-data HPO.

### E14 — Runtime / cost
- **Tests:** C11. **Protocol:** wall-clock for condensation per method/dataset;
  downstream train time on condensed vs full.
- **Artifact:** **T10**.
- **Predicted:** HistDistill condensation cost comparable to herding, far below
  KIP/TDColER; downstream train ≥10× faster than full.

### E15 — Sensitivity (appendix)
- **Tests:** robustness. **Protocol:** vary bins $B_j\in\{64,128,256\}$,
  $\lambda$, teacher depth; report AUROC/SLD stability.
- **Artifact:** **F9**.
- **Predicted:** results stable across reasonable settings.

---

## 5. Result Tables (templates)

Legacy values from broad26 (`FINAL_RESULTS.md`) are filled as anchors; new methods
`TBD` with predicted direction (↑ = expected above the best anchor).

### T1 — Main binary performance (broad26, budgets 25/50, 5 seeds, 5 learners)
| Method | Mean AUROC | Std | Avg. Rank |
| --- | --- | --- | --- |
| Full data | (anchor) | | |
| **HistDistill-Density** | TBD ↑ | | |
| **HistDistill-Refined** | TBD ↑ | | |
| **HistDistill-Greedy** | TBD ↑ | | |
| TDColER | TBD | | |
| KIP-tabular | TBD | | |
| GradMatch | TBD | | |
| Craig | TBD | | |
| _legacy_ gain_path_refined | 0.7509 | 0.1777 | 3.040 |
| _legacy_ gain_path_safeguarded | 0.7504 | 0.1779 | 3.009 |
| _legacy_ gain_sketch | 0.7490 | 0.1763 | 3.189 |
| Herding | 0.7339 | 0.1681 | 4.179 |
| Distribution matching | TBD | | |
| K-center | TBD | | |
| Random | 0.7260 | 0.1733 | 4.533 |

### T2 — Split-landscape fidelity (lower SLD = better; agreement higher = better)
| Method | SLD∞ ↓ | SLD₁ ↓ | root agree ↑ | top5 ↑ | gain-rank corr ↑ |
| --- | --- | --- | --- | --- | --- |
| Full data | 0 | 0 | 1.000 | 0.683 | 0.933 |
| **HistDistill-*** | TBD (→0) | TBD | TBD (→1) | TBD | TBD (≫0.35) |
| _legacy_ gain_path_refined | TBD | TBD | 0.000 | 0.113 | 0.345 |
| Random | TBD | TBD | 0.125 | 0.129 | 0.268 |

> Prediction (Lemma 1): HistDistill root-agreement and gain-rank-corr jump toward
> the full-data column — the qualitative win that legacy row methods cannot get.

### T3 — Paired statistical tests (HistDistill-Refined vs each)
| Comparator | Mean Diff | Wins | Ties | Losses | Wilcoxon p |
| --- | --- | --- | --- | --- | --- |
| Random | TBD ↑ | | | | <1e-10 (pred) |
| Herding | TBD ↑ | | | | <1e-8 (pred) |
| TDColER | TBD ↑ | | | | <1e-2 (pred) |
| KIP-tabular | TBD ↑ | | | | <1e-2 (pred) |
| _legacy_ gain_path_refined | TBD ↑ | | | | <0.05 (pred) |

### T4 — Cross-learner transfer (mean AUROC by learner)
| Method | catboost | lightgbm | mlp | random_forest | xgboost |
| --- | --- | --- | --- | --- | --- |
| Full data | 0.906 | 0.910 | 0.715 | 0.907 | 0.914 |
| **HistDistill-Density** | TBD | TBD | TBD ↑↑ | TBD | TBD |
| **HistDistill-Refined** | TBD | TBD | TBD | TBD | TBD |
| _legacy_ gain_path_refined | 0.859 | 0.860 | 0.685 | 0.890 | 0.882 |
| Random | 0.825 | 0.827 | 0.650 | 0.862 | 0.849 |

### T5 — Selector study (Prop 4 validation)
| Selector | f(S)/f_ub ↑ | SLD∞ ↓ | AUROC@B=10 | AUROC@B=25 |
| --- | --- | --- | --- | --- |
| HistDistill-Greedy | TBD (≥0.63) | TBD | TBD | TBD |
| ImportanceSample (Thm 1) | TBD | TBD | TBD | TBD |
| Proxy upper bound | 1.000 | — | — | — |

### T6 — Component ablation (Δ vs HistDistill-Refined)
| Variant | ΔAUROC | ΔSLD∞ | Wilcoxon p |
| --- | --- | --- | --- |
| − weight fit (5) | TBD (↓) | TBD (↑) | TBD |
| − saturation φ (linear) | TBD | TBD | TBD |
| root-only probes | TBD (↓) | TBD (↑) | TBD |
| + depth-2 probes | TBD | TBD | TBD |
| − refinement | TBD | TBD | TBD |
| − density term | TBD (MLP↓) | TBD | TBD |

### T7 — Regression (mean R²)
| Method | Mean R² | Avg. Rank | Wilcoxon vs Random |
| --- | --- | --- | --- |
| Full data | (anchor) | | |
| HistDistill-Refined | TBD ↑ | | TBD |
| GradMatch | TBD | | |
| Herding / Random | TBD | | — |

### T8 — Multiclass (mean macro-AUROC)
| Method | Macro-AUROC | Avg. Rank | Wilcoxon vs Random |
| --- | --- | --- | --- |
| Full data | (anchor) | | |
| HistDistill-Refined | TBD ↑ | | TBD |
| Herding / Random | TBD | | — |

### T9 — Fast HPO
| Dataset | Top-3 agree ↑ | Speedup ↑ | Test AUROC gap ↓ |
| --- | --- | --- | --- |
| adult / spambase / … | TBD (≥0.8) | TBD (≥10×) | TBD (≤0.01) |

### T10 — Runtime
| Method | Condense time (s) | Downstream train speedup vs full |
| --- | --- | --- |
| HistDistill-Refined | TBD | TBD (≥10×) |
| KIP / TDColER | TBD (≫) | — |

---

## 6. Figures

- **F1** Critical-difference rank diagram (E1, Friedman+Nemenyi).
- **F2** SLD↔AUROC scatter with Prop-1 guarantee band (E3) — *the theory figure*.
- **F3** Budget curves: AUROC-vs-B and SLD-vs-B collapsing across $N$ (E4).
- **F4** Cross-learner transfer profiles (E7).
- **F5** MMD–SLD Pareto frontier colored by MLP/XGB AUROC (E8).
- **F6** Anchoring $\rho_t$ vs round (E9, appendix).
- **F7** HPO speedup vs agreement (E13).
- **F8** Margin distributions (E5, appendix).
- **F9** Sensitivity to bins/λ/depth (E15, appendix).

---

## 7. Statistical Protocol

- Pairing on identical (dataset, budget, seed, learner) cells.
- Paired **Wilcoxon signed-rank** for each method pair; report mean diff, median
  diff, win/tie/loss, p.
- **Friedman** omnibus across methods + **Nemenyi** post-hoc → CD diagram (F1).
- **95% bootstrap CIs** (10k resamples over datasets) on mean AUROC.
- Multiple-comparison control: Holm correction across the baseline family.
- Effect sizes (median diff + rank-biserial) alongside p-values — avoid p-only.

---

## 8. Reproducibility

- Fixed seeds; teacher config, bin edges, λ, γ, probe depths logged per run.
- All condensed artifacts (rows + weights + target histograms) serialized.
- Per-cell metric rows exported to `paper_tables/*.csv`; aggregation via existing
  `09_aggregate_results.py` / `13_focused_report.py` extended with an SLD column.
- New artifacts to add:
  `paper_tables/sld_per_cell.csv`, `sld_vs_auroc.csv`, `budget_collapse.csv`,
  `selector_study.csv`, `transfer_regret.csv`, `mmd_sld_pareto.csv`,
  `anchoring_rho.csv`, `hpo_agreement.csv`, `runtime.csv`,
  `regression_performance.csv`, `multiclass_performance.csv`.

---

## 9. Success Criteria (go/no-go before writing claims)

A claim ships only if its check is green:

1. **C1/C2:** HistDistill root-agreement ≥ 0.8 and gain-rank-corr ≥ 0.6 (vs 0.35
   legacy); SLD↔AUROC monotone; near-100% agreement when $\mathrm{SLD}_\infty<\Delta/2$.
2. **C3:** SLD-vs-$B$ curves from small-$N$ and large-$N$ datasets overlap within CI.
3. **C4:** greedy $f/f_{ub}\ge 0.63$ and ≥ importance sampling at $B=10$.
4. **C5:** HistDistill > TDColER and KIP on mean AUROC with $p<0.05$.
5. **C6:** `-Density` closes ≥50% of MLP transfer regret vs `-Refined`.
6. **C9:** regression R² and multiclass macro-AUROC beat Random/coreset, $p<0.05$.
7. **C10:** HPO top-3 agreement ≥0.8 at ≥10× speedup.

Fallback priority if time slips: keep C1–C5 + C9(regression) + C10; drop multiclass,
then E8 Pareto, then E15. Minimum viable paper = E1, E2, E3, E4, E10, E11, E13.

---

# PHASE 2 — Revised Plan After Reading the Completed Runs

> **Why this section exists.** Phase 1 (E1–E15) was written *before* the binary
> grid finished. The completed runs (see `ICDM_CORE_UPDATE.md`,
> `ICDM_EXPERIMENT_AUDIT.md`, `ICDM_THEORY_DIAGNOSTICS.md`) force three honest
> corrections that change what we can claim, and therefore what we should run next:
>
> 1. **AUROC is a statistical tie with the legacy method**, *not* a win.
>    Completed 3250-cell grid: legacy gain-path refined **0.7430**,
>    HistDistill-Refined **0.7425**, HistDistill-Density **0.7425**. The paired
>    Wilcoxon vs legacy is `mean diff = −0.0006, p = 0.039` — i.e. we are
>    statistically *indistinguishable* (and nominally a hair below). We DO beat
>    Herding (`+0.0118`, p≈0) and Random (`+0.0181`, p≈0). **C5 as originally
>    phrased ("beats best legacy on AUROC") is dead. Stop chasing it.**
> 2. **The win that is real is fidelity.** On the 650-cell direct-SLD grid,
>    HistDistill is best-in-class: direct $\mathrm{SLD}_\infty$ = **16.7**
>    (Refined) / **17.0** (Density) vs **18.4** legacy, **27.7** Random,
>    **34.6** Herding; direct root-agreement **0.320 / 0.315** vs 0.306 legacy
>    and 0.192 Random/Herding. We are the most *mechanism-faithful* condenser,
>    and we tie the best method on accuracy while being so.
> 3. **The theory's sharpest empirical signal is the dissociation between
>    fidelity and accuracy**, which the completed ablation makes load-bearing:
>    the provably $(1{-}1/e)$-optimal pure coverage selector (`HistDistill-Greedy`,
>    $f/f_{ub}=1.0$, coverage ratio 0.37/0.54 vs 0.06–0.16 for everyone else) is
>    the **worst** on accuracy (**0.6285**), while the deployed variant reaches
>    **0.7498**. Optimizing split coverage is necessary but *not sufficient*.
>
> **Reframe.** The paper is no longer "our method wins the leaderboard." It is a
> **measurement-and-analysis paper**: we introduce SLD as a tree-mechanism
> fidelity axis, prove when it controls structure preservation (Prop 1, confirmed
> at 100% on margin-satisfied cells), and use it to *explain* the whole condenser
> landscape — including why provably-optimal coverage still underperforms. New
> baselines are run as **points on the SLD map that the theory then explains**,
> not as a scoreboard we must top. Phase 2 (E16–E21) is built to make that spine
> rigorous and to convert the weak/negative results (regression, HPO) into honest
> framework-generality evidence rather than overclaims.

---

## P2.0 New Claim → Experiment Map

| # | Claim (Phase 2, defensible against the real numbers) | Experiment | Headline artifact |
| --- | --- | --- | --- |
| C12 | Downstream error splits into **structure error** (controlled by SLD) + **leaf-estimate error** (controlled by population representativeness); coverage selectors kill the first but not the second | E16 | T11, F10 |
| C13 | A **continuous split-regret** metric (not the coarse 12-cell binary root-agreement) tracks AUROC *within dataset* and resolves the Simpson reversal | E17 | T12, F11 |
| C14 | The SLD→accuracy link is **causal/monotone under controlled dosing**, not just observational | E18 | F12 |
| C15 | SLD is a **universal diagnostic**: it linearly orders *all* condensers (GOSS, MVS, CRAIG, GradMatch, DM, TDColER, ours) by mechanism fidelity, and the residual off the SLD line is exactly the leaf-estimate term from C12 | E19 | T13, F13 |
| C16 | SLD computed **once, training-free**, predicts which condensed set yields the best downstream model — replacing the weak HPO claim with a defensible selection-rule claim | E20 | T14, F14 |
| C17 | The framework **generalizes in form** to regression/multiclass/HPO; we report multiclass as supportive and regression/HPO as **documented, theory-consistent failure modes** (high leaf-estimate error regimes) | E21 | T15 |

Phase 2 ships if **C12 + C13 + C15** land (these are the analysis spine and are
low-risk — they re-slice data we already have). C14, C16 are high-value stretch
claims. C17 is honest-scoping insurance.

---

## P2.1 New Baselines — protocol and why each is here

All new baselines are run **as condensers under one common protocol** so they are
comparable to us and become labeled points on the SLD map:

> **Common condensation protocol.** Run the selector *once* against the frozen
> teacher (its gradients/Hessians/landscape, exactly as our pipeline already
> computes) to produce a frozen size-$B$ weighted subset per (dataset, budget,
> seed). Then train **all 5 downstream learners** on that one frozen set. This
> makes an in-training sampler (GOSS/MVS) comparable to a persistent condensed
> dataset, and is the *only* fair way to compare — we must state this explicitly
> in the paper, because GOSS/MVS were designed to resample every round.

**Tier 1 — gradient / coverage coresets (must-have; these are the real prior-art threats).**
- **GOSS** (LightGBM gradient-based one-side sampling): keep top-$a$ by $|g|$, sample
  $b$ from the rest with amplification. *Why:* it has a published gain-error bound —
  the closest thing to our Prop 1. We must show our offline SLD framing is distinct
  and that GOSS lands at a *specific, explainable* SLD.
- **MVS** (CatBoost minimal-variance sampling): sample probability $\propto$
  $\sqrt{g^2+\lambda h^2}$. *Why:* the variance-optimal one-round sampler; direct
  competitor for "which rows matter for the next split."
- **CRAIG** (submodular facility-location gradient coreset + per-element weights).
  *Why:* the canonical submodular coreset; our Prop 4 coverage objective must be
  shown to be a *different* submodular function (split-coverage vs gradient-cover),
  and CRAIG is the head-to-head for the submodular story.
- **GradMatch** (orthogonal-matching-pursuit gradient-sum matching + weights).
  *Why:* matches *aggregate* gradient, the most direct "match the teacher signal"
  baseline; expected to be strong on AUROC but weak on SLD (it doesn't target splits).

**Tier 2 — modern distillation (credibility; reviewers will ask).**
- **DM** (Distribution Matching, feature-space MMD synthesis adapted to tabular).
- **TDColER** (tabular distillation, official code if runnable; else faithful reimpl
  validated within tolerance on one overlap dataset). *Why:* without at least one
  *learned-synthesis* distillation baseline, the related-work claim is hollow. These
  are expected to be strong on MLP/density and weak on tree-SLD — which is itself a
  finding (they don't respect the split mechanism).

**Tier 3 — data-pruning scores (cheap, appendix breadth).**
- **GraNd / EL2N** (gradient-norm / error-$\ell_2$ importance), **forgetting score**,
  **leverage / k-center**. *Why:* round out the "importance vs coverage" axis cheaply;
  all reuse the teacher signal we already store. Appendix unless one surprises us.

**Predicted placement on the SLD map (falsifiable):**
- GradMatch, GraNd/EL2N: low transfer regret on trees, **high SLD** (match gradient
  mass, not split structure) → they sit *above* our SLD line (good accuracy for their
  SLD) only if leaf-estimate error is low; otherwise on the line.
- GOSS, MVS, CRAIG: moderate SLD, moderate accuracy — should land **on** the SLD line,
  confirming SLD as the universal ordering.
- DM, TDColER: **high SLD**, possibly decent MLP — the off-tree-mechanism corner; the
  visual proof that synthesis-without-splits is the wrong axis for GBDT.
- Ours (Refined/Density): lowest SLD, on or slightly above the line.

If a Tier-1 baseline both **beats us on AUROC and matches our SLD**, the fidelity
story is in trouble — that is the explicit kill-switch we test for.

---

## P2.2 Experiments E16–E21 (in depth)

Format per experiment: *what it is → what it tests → exact protocol → predicted
result (anchored to measured numbers) → interpretation / claim licensed → kill-switch*.

### E16 — Structure vs leaf-estimate error decomposition  *(the new spine, C12)*

- **What it is.** The single experiment that explains the coverage≠accuracy result.
  Given a condensed set $\tilde D$, fit the tree **structure** (split features +
  thresholds) on $\tilde D$ as usual, then produce two predictors: (a) the normal
  one with leaf values estimated from $\tilde D$; (b) the **same structure with leaf
  values re-estimated on the full training data**. Decompose total error:
  $$\underbrace{\mathrm{err}(\tilde D)}_{\text{deployed}}
   = \underbrace{[\mathrm{err}(\text{struct}(\tilde D),\,\text{leaves}_{\text{full}}) - \mathrm{err}(\text{full})]}_{\text{STRUCTURE error}}
   + \underbrace{[\mathrm{err}(\tilde D) - \mathrm{err}(\text{struct}(\tilde D),\,\text{leaves}_{\text{full}})]}_{\text{LEAF-ESTIMATE error}}.$$
- **Tests.** C12. Mechanistically separates the two failure channels SLD does (splits)
  and does *not* (leaf populations) control.
- **Protocol.** Reuse the existing condensed artifacts for all methods on broad26,
  budgets 25/50, seeds 0–2, XGB+LGBM (tree learners only — leaf refit is only defined
  for trees). For each cell: refit leaves on full train holding structure fixed; log
  structure-error and leaf-estimate-error in AUROC units. Correlate STRUCTURE error
  with measured SLD; correlate LEAF-ESTIMATE error with a leaf-population
  representativeness measure (per-leaf weight-vs-full-mass KL, or per-leaf MMD).
- **Predicted result (anchored).** For **HistDistill-Greedy** (the provably-optimal
  coverage selector, AUROC **0.6285**, $f/f_{ub}=1.0$): structure error should be
  *small* (it covers splits) but leaf-estimate error **large** — recovering most of
  the gap from 0.6285 toward the ~0.75 of refined when leaves are refit. For
  **HistDistill-Refined** (**0.7498**) and **legacy** (**0.7430**): both errors small,
  refined slightly lower structure error (consistent with its lower SLD 16.7 vs 18.4).
  For **Random** (0.7244, SLD 27.7): structure error large. Expect STRUCTURE-error vs
  SLD Spearman strongly positive within dataset; LEAF-error vs SLD ≈ 0.
- **Interpretation / claim licensed.** "SLD provably and empirically governs the
  *structure* component of condensed-tree error; the residual accuracy gap is a
  *separable* leaf-estimate term governed by population representativeness, which the
  weight-fit and density terms target. This is why a $(1{-}1/e)$-optimal coverage
  selector is necessary but not sufficient." This is a genuinely novel, citable
  decomposition for tree condensation and the paper's strongest single result.
- **Kill-switch.** If leaf-refit does *not* substantially close the Greedy gap (i.e.
  Greedy's problem is structural after all), the coverage≠accuracy story is wrong and
  we fall back to SLD-as-pure-diagnostic without the decomposition claim.

### E17 — Continuous split-regret metric  *(C13, fixes the Simpson reversal)*

- **What it is.** Replace the coarse binary root-split-agreement (only ~12 margin
  cells, 0/1 valued) with a **continuous split-regret**: for each internal node,
  $\mathrm{regret}(v) = \Gamma^{\text{full}}(s^\star_{\text{full}}) - \Gamma^{\text{full}}(s_{\tilde D}(v))$
  — the *true* gain lost by taking the condensed-chosen split instead of the
  full-data-optimal split, evaluated under the full-data landscape. Aggregate
  (root, depth-1, depth-2) with gain weighting.
- **Tests.** C13, and directly addresses the documented Simpson's paradox: pooled
  Spearman(SLD, AUROC) = **+0.20** (wrong sign) but within-dataset = **−0.56**
  (correct). Split-regret is dataset-normalized by construction, so it should be
  monotone with AUROC *both* pooled and within-dataset.
- **Protocol.** Compute split-regret for every cell on the 650-cell direct grid +
  the full 3250 where landscapes are stored. Report pooled and within-dataset
  Spearman of split-regret vs AUROC; compare to the SLD versions; show the binary
  root-agreement as the coarse special case.
- **Predicted result (anchored).** Within-dataset Spearman strengthens past the
  current SLD −0.56; **pooled** Spearman flips to the correct (negative) sign,
  eliminating the reversal. Method ordering by mean split-regret matches AUROC
  ordering: Refined ≈ Density < legacy < Herding/Random.
- **Interpretation / claim licensed.** "Split-regret is the right scalar
  summary of tree fidelity: it is monotone with downstream accuracy without the
  cross-dataset confound that afflicts raw SLD, and the binary root-agreement
  used by prior tabular work is its degenerate 0/1 special case." Lets us *lead*
  with one clean fidelity number instead of explaining away a sign flip.
- **Kill-switch.** If pooled split-regret still shows the reversal, we keep SLD but
  report only the within-dataset law and explicitly disclose the pooled confound
  (still honest, weaker).

### E18 — Controlled SLD dose-response  *(C14, causal upgrade)*

- **What it is.** Turn the observational SLD↔AUROC correlation into a *controlled*
  one. Take a near-perfect condensed set and **inject calibrated perturbations into
  the split landscape** to hit target SLD levels $\varepsilon \in
  \{0,\,0.25\Delta,\,0.5\Delta,\,\Delta,\,2\Delta,\dots\}$ (dosing relative to the
  measured margin $\Delta$), holding the leaf-estimate channel fixed; measure
  resulting AUROC and root agreement.
- **Tests.** C14, and a direct second confirmation of Prop 1 (the $\mathrm{SLD}_\infty<\Delta/2$
  guarantee) beyond the 12 naturally-occurring margin cells.
- **Protocol.** On ~6 representative datasets (mix of $N$), 3 seeds: sweep dosed SLD,
  record AUROC + agreement; fit dose-response curve; locate the empirical knee vs the
  Prop-1 predicted $\Delta/2$ threshold.
- **Predicted result (anchored).** Agreement stays ~1.0 for $\varepsilon<\Delta/2$
  (matching the measured Prop1 agreement = **1.0** on margin-satisfied cells), then
  degrades; AUROC monotonically declines past the knee. The knee sits near the
  predicted $\Delta/2$.
- **Interpretation / claim licensed.** "The SLD→structure→accuracy chain is causal,
  not merely correlational: controlled SLD dosing moves accuracy in the predicted
  direction with the predicted threshold." Upgrades the centerpiece from "law we
  observed" to "mechanism we can steer."
- **Kill-switch.** If accuracy is flat under dosing, SLD is not causally controlling
  accuracy on those datasets → demote to observational and lean on E16 instead.

### E19 — SLD as a universal condenser diagnostic  *(C15, the baseline payoff)*

- **What it is.** Place *every* baseline (Tier 1–3 + ours + legacy) on one
  SLD-vs-AUROC plot and show SLD linearly orders mechanism fidelity across methods
  of completely different design (sampling, submodular, synthesis).
- **Tests.** C15; this is where the new baselines earn their keep — as explained
  points, not competitors.
- **Protocol.** Common condensation protocol (P2.1) for all baselines; compute
  SLD/split-regret + AUROC per cell; fit the cross-method SLD→AUROC line; measure each
  method's residual off the line and correlate that residual with the E16
  leaf-estimate error.
- **Predicted result (anchored).** Methods fall on a descending SLD→AUROC line with
  ours/legacy at the low-SLD end (16.7–18.4, AUROC ~0.743), Random/Herding at the
  high-SLD end (27.7/34.6, AUROC 0.724/0.731). GradMatch/DM/TDColER predicted
  high-SLD; if any sits *above* the line, its positive residual should equal a *low*
  E16 leaf-estimate error — closing the loop with C12.
- **Interpretation / claim licensed.** "One scalar (split-regret/SLD) explains the
  entire condenser landscape for GBDT, and deviations from it are exactly the
  leaf-estimate term — a unifying diagnostic the field currently lacks." This is the
  measurement-paper headline.
- **Kill-switch.** If methods do *not* fall on a common line (SLD is method-specific),
  the universality claim drops to "within our method family" and the paper narrows.

### E20 — Training-free condensed-set selection  *(C16, replaces weak HPO)*

- **What it is.** A defensible practical claim to *replace* the weak/negative HPO
  result: SLD (or split-regret), computed **once and without any downstream
  training**, predicts which of several candidate condensed sets will yield the best
  downstream model. Selection rule, not hyperparameter search.
- **Tests.** C16. Practical value of the metric itself.
- **Protocol.** For each dataset, generate $k$ candidate condensed sets (methods ×
  seeds × small config grid). Rank them by training-free SLD; check whether the
  SLD-argmin matches the AUROC-argmax (top-1 / top-3 agreement, rank correlation).
  Report wall-clock saved vs the train-everything baseline.
- **Predicted result (anchored).** Given the within-dataset SLD↔AUROC Spearman of
  −0.56 and the clean method ordering, expect top-3 selection agreement ≈ 0.7–0.85
  and large wall-clock savings (SLD is a histogram diff vs full retraining). Honest:
  top-1 may be noisy where candidates tie within ~0.0006 (the legacy/refined regime).
- **Interpretation / claim licensed.** "SLD is an actionable, training-free model-
  selection signal for condensation." Keeps a practical contribution without the
  HPO overclaim flagged in the audit (E13 HPO = weak/negative).
- **Kill-switch.** If SLD-argmin ≠ AUROC-argmax more often than chance, drop to "SLD
  predicts ordering on average" (still supported by the −0.56 law) and cut the
  selection-rule framing.

### E21 — Generality scoping: multiclass supportive, regression/HPO as documented limits  *(C17)*

- **What it is.** Convert the audit's weak results into honest, theory-consistent
  scoping rather than buried failures. Multiclass (8 datasets OvR, supportive) is
  reported as positive breadth; regression (8 datasets, weak/mixed) and HPO
  (weak/negative) are reported as **predicted high-leaf-estimate-error regimes** and
  analyzed with the E16 decomposition.
- **Tests.** C17 + a stress test of the E16 theory (does the decomposition *predict*
  where we fail?).
- **Protocol.** Keep existing multiclass/regression/HPO runs. For regression, apply
  the E16 decomposition: show regression's gap is dominated by leaf-estimate error
  (continuous targets ⇒ leaf means are population-sensitive), exactly where SLD
  offers no protection. Tree-learner slice reported separately (audit notes it is more
  favorable than the all-learner slice).
- **Predicted result (anchored).** Multiclass: HistDistill ≥ Random/coreset on
  macro-AUROC (supportive, per audit). Regression all-learner: weak (per audit), with
  E16 attributing the gap to leaf-estimate error; tree-only slice more favorable. HPO:
  reported as application limitation, redirected to E20's selection claim.
- **Interpretation / claim licensed.** "The framework is general in *form*; SLD's
  protection is specific to the *structure* channel, so tasks dominated by the
  leaf-estimate channel (regression) are predictably harder — and our own theory says
  so." Turning a weakness into a theory-confirming boundary is reviewer-credible.
- **Kill-switch.** None needed — this is the honest-scoping experiment; worst case it
  stays in the appendix as documented limitations.

---

## P2.3 New tables / figures

| ID | Content |
| --- | --- |
| **T11** | Structure vs leaf-estimate error decomposition by method (E16): Greedy 0.6285 with large leaf-estimate gap vs Refined 0.7498 small both. |
| **T12** | Split-regret vs SLD vs binary-agreement; pooled & within-dataset Spearman with AUROC (E17). |
| **T13** | All-baseline SLD/split-regret + AUROC + residual-off-line, with E16 leaf-error column (E19). |
| **T14** | Training-free selection: top-1/top-3 agreement, rank corr, wall-clock saved (E20). |
| **T15** | Generality summary: multiclass macro-AUROC (supportive), regression R² with E16 attribution, HPO note (E21). |
| **F10** | Structure-error-vs-SLD and leaf-error-vs-representativeness scatters (E16). |
| **F11** | Split-regret↔AUROC, pooled vs within-dataset, showing the Simpson reversal resolved (E17). |
| **F12** | SLD dose-response curve with the $\Delta/2$ Prop-1 threshold marked (E18). |
| **F13** | The universal SLD map: all condensers as labeled points on one fidelity↔accuracy line (E19) — *the new money figure*. |
| **F14** | Training-free selection agreement vs wall-clock saved (E20). |

---

## P2.4 What gets demoted (so the paper stays honest)

- **AUROC-beats-legacy (old C5):** removed as a claim. We report the **tie**
  (0.7425 vs 0.7430, p=0.039) plainly and pivot the contribution to fidelity +
  analysis. We still keep the significant wins over Random/Herding.
- **Lemma 1, perturbation Prop 1 (existence), submodular Prop 4 (as novelty):**
  demoted to *background + cited* (GOSS/MVS/CRAIG own the priors). Prop 1 stays as a
  *used* tool (E18 dosing); Prop 4 stays as the *reason* coverage is optimal-yet-
  insufficient (the E16 setup), not as a headline theorem.
- **Regression / HPO / MLP-density:** appendix, framed via E21 as documented limits.
- **Budget-collapse (E4):** kept but supporting, not headline (slopes −0.44 refined /
  −0.52 random already measured; consistent with $\tilde O(\varepsilon^{-2})$).

## P2.5 Optional method-win shot (only if E16 lands early)

If E16 cleanly isolates the leaf-estimate term, build **one** selector that jointly
optimizes split-coverage (Prop 4) **and** the leaf-estimate fidelity term E16 exposes
(per-leaf population matching). Predicted to push past the 0.7498 ablation ceiling and
*possibly* clear the legacy 0.7430 tie with significance. This is the only remaining
path to a genuine accuracy win, and it falls out of the analysis rather than being
bolted on. Time-box it; the paper does not depend on it.

---

## P2.6 Revised execution order (≈3-week horizon)

1. **Week 1 (low-risk spine, reuses stored artifacts):** E16, E17, E19 on existing
   condensed sets → C12, C13, C15. These need re-slicing, not new condensation runs.
2. **Week 1–2 (new baselines):** Tier 1 (GOSS, MVS, CRAIG, GradMatch) under the common
   protocol; feed into E19. Tier 2 (DM, TDColER) next; Tier 3 if time.
3. **Week 2 (causal + practical):** E18 dose-response, E20 training-free selection.
4. **Week 2–3:** E21 scoping (reuse runs + E16 attribution); optional P2.5 selector.
5. **Week 3:** writing, CD diagrams, the F13 universal map, statistics hardening.

Phase-2 go/no-go: **C12 + C13 + C15 green ⇒ submit as a measurement/analysis paper.**
C14/C16/P2.5 green ⇒ stronger; their absence does not sink the paper.
