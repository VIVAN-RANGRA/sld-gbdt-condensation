# Writing Style Guide: Applied/ADS Track Papers
## For ICDM 2026 Applied Track — Measurement + Diagnostic Paper on Data Condensation for GBDTs

*Research basis: KDD ADS Call for Papers 2018–2026, ICDM 2026 Applied Track CFP, ECML-PKDD ADS track, and analysis of accepted applied-track paper conventions.*

---

## 1. Canonical Section Structure of an Accepted ADS/Applied Paper

The following structure is the consensus template across KDD ADS, ICDM Applied, and ECML-PKDD ADS tracks. It differs from the pure-research structure in where emphasis falls.

| # | Section | One-line purpose |
|---|---------|-----------------|
| 1 | **Abstract** | Lead with the practitioner problem and the concrete finding; end with the actionable takeaway, not the method name. |
| 2 | **Introduction** | Establish the real-world cost/pain first (1–2 paragraphs), then state what the paper *found*, then list contributions as bullet claims — not as a method tour. |
| 3 | **Problem Formulation / Background** | Define only what is needed to make the analysis intelligible; keep notation minimal. Contrast with pure-research papers that spend 2 pages on theory. |
| 4 | **Related Work** | Short (0.5–1 page), positioned *after* problem formulation; frame prior work as "what practitioners have tried" or "what the field assumed" that this paper challenges or extends. |
| 5 | **Methods / Experimental Setup** | Describe the 17 condensation methods, datasets, and the diagnostic tool as a *measurement apparatus*, not a novel algorithm. Justify design choices in terms of practical realism. |
| 6 | **The Diagnostic / Metric** | Dedicate one section to the diagnostic itself: what it measures, why it predicts practitioner-relevant fidelity, and its computational cost. This is the primary novel contribution. |
| 7 | **Experiments & Results** | Organized around *questions a practitioner would ask*, not around ablation tables. Each subsection = one finding, stated as a boldface claim, supported by evidence, ended with a practitioner takeaway box or italicized sentence. |
| 8 | **Discussion / Implications** | Synthesize the non-obvious cross-cutting findings (e.g., "coverage ≠ accuracy"). Map each insight to a decision a practitioner must make. This section is what ADS reviewers remember. |
| 9 | **Limitations** | Honest, specific, forward-pointing — not defensive. One focused paragraph or short sub-section. |
| 10 | **Conclusion** | Re-state the 3–4 practitioner takeaways, not the method summary. End with what a practitioner should do differently after reading this paper. |

### How this differs from a pure-research-track paper

- **Research track**: Abstract/Intro leads with a theoretical gap; Related Work is 2+ pages; Results center on state-of-the-art tables; Discussion is short.
- **ADS/Applied track**: Intro leads with practitioner cost; Related Work is short and problem-anchored; Results center on *findings and their meaning*; Discussion is the climax, not the afterthought.
- **Research track**: Novelty = new algorithm or theorem.
- **ADS/Applied track**: Novelty = new understanding of a real problem, new diagnostic, new empirical finding with practical consequence. Explicitly stated by KDD: "novelty may lie in the choice of application domain, engineering design, usability, or business use case."

---

## 2. How ADS Papers Motivate the Problem

**Rule: Start from a concrete practitioner pain point with a cost, not from a theoretical observation.**

### The pattern used in accepted ADS papers:

1. **Sentence 1–2**: State the practitioner situation in concrete terms — who does what, at what scale, with what resource constraint.
   - *Bad (research-track style)*: "Dataset condensation has attracted significant theoretical interest as a data-efficient learning paradigm."
   - *Good (ADS style)*: "ML teams at mid-to-large organizations routinely re-train gradient-boosted models on datasets of millions of rows; each retraining pipeline consumes hours of compute and requires storing the full dataset, often across regulatory boundaries."

2. **Sentence 3–4**: Identify the specific friction — the action practitioners already take or want to take — and the cost of doing it poorly.
   - "Practitioners already condense training data informally (e.g., random subsampling, stratified sampling); the question is whether they can trust a condensed dataset to preserve model behavior — and currently they have no principled way to answer this before training."

3. **Sentence 5+**: State the gap as a *decision problem*: "There is no diagnostic that tells a practitioner, before committing to a condensed dataset, how faithfully it will reproduce the model's predictions."

4. **Do NOT start from**: survey counts, citation counts, conferences where the topic appeared, abstract theoretical desiderata.

### What motivates reviewers:
KDD ADS explicitly requires papers to "specify an audience or group of users that have benefited or will benefit." Name the audience in the first paragraph. For this paper: ML engineers and data scientists who train and deploy GBDT models on tabular data.

---

## 3. How to Frame Novelty When the Contribution Is Measurement/Analysis/Diagnostic (Not a New SOTA Model)

### The KDD ADS "Evidential" category was designed for exactly this paper.

The KDD ADS Evidential category accepts papers that "provide significant gains in the understanding of an applied area/domain." The paper does not need a deployed system. What it needs is:
- A clear real-world problem that motivated the study.
- A description of the milestones reached (the diagnostic, the comparative analysis).
- A statement of practical impact (what practitioners can now do that they couldn't before).
- An honest account of obstacles to broader deployment (limitations).

### The framing formula for a measurement/diagnostic contribution:

> "We do not propose a new condensation method. Our contribution is a training-free diagnostic [NAME] that predicts condensation fidelity for GBDT models, and a systematic comparative analysis of 17 methods — the first such study for tree-based models on tabular data. The diagnostic enables practitioners to select condensation methods without running full retraining experiments."

### Analogous accepted papers in this mold (study/benchmark/diagnostic, no new SOTA):

- **DC-BENCH (NeurIPS 2022 Datasets & Benchmarks)**: No new condensation method — introduced a standardized benchmark, ran a large-scale comparison, reported non-obvious findings about evaluation methodology. Contribution = the measurement infrastructure and empirical findings.
- **"Why do tree-based models still outperform deep learning on tabular data?" (NeurIPS 2022)**: No new model — systematic comparison across 45 datasets, finding that GBDTs remain superior and analyzing *why*. Published as a research contribution purely on the basis of empirical insight.
- **"Tabular Data Distillation: An Extensive Comparison" (MDPI Machine Learning and Knowledge Extraction, 2024)**: Evaluation of distillation methods for non-neural models — 17 classification and 9 regression problems, 5 methods. Contribution = comparative analysis and practitioner guidance.
- **TabReD (2024)**: No new model — identified gaps in tabular benchmarks by analyzing properties of industrial vs. academic datasets. Contribution = understanding + new datasets with better real-world realism.
- **"A Closer Look at Deep Learning Methods on Tabular Datasets" (2024)**: 300-dataset benchmark, no new method, contribution = the empirical findings themselves.

### How to state measurement novelty without overclaiming:

Use this structure in the Introduction contributions list:
```
(1) We introduce [DIAGNOSTIC NAME], a training-free metric that predicts how
    faithfully a condensed dataset preserves a GBDT's predictive behavior on
    tabular data (Section X).
(2) We present the first systematic comparative evaluation of 17 condensation
    methods specifically for GBDT-on-tabular settings, across N datasets and
    M configurations (Section Y).
(3) We identify several non-obvious empirical findings — including that
    maximizing data coverage does not maximize prediction accuracy — and
    provide practitioner guidance on method selection (Section Z).
```

Do NOT use: "state-of-the-art," "outperforms," "novel algorithm." Do USE: "first systematic," "diagnostic," "empirical findings," "practitioner guidance."

---

## 4. Tone and Voice Conventions

### Person
- Use **first-person plural ("we")** throughout. This is the norm for ADS papers (which are single-blind — author names visible). Avoid passive voice constructions that obscure agency ("it was found that…").

### Tense
- Present tense for claims, findings, and contributions: "The diagnostic *predicts*…"; "Coverage *does not* correlate with accuracy."
- Past tense for what was done in experiments: "We *trained* 17 methods on…"; "We *measured* fidelity using…"
- Do not mix tenses within a finding statement.

### Concreteness
ADS reviewers penalize vagueness. Every claim must be anchored to a number, a dataset count, a method name, or an observable outcome.
- *Vague*: "Our diagnostic performs well across a variety of settings."
- *Concrete*: "The diagnostic achieves a Spearman rank correlation of 0.87 with post-training accuracy across 15 datasets and all 17 methods."

### Hedging
- Hedge *scope*, not *findings*. State what you found confidently; hedge the generalization domain.
- *Over-hedged (bad)*: "Results may potentially suggest that coverage could possibly be less important than previously assumed."
- *Appropriately scoped (good)*: "For the 15 tabular datasets and 17 methods we studied, coverage-maximizing methods did not yield higher GBDT accuracy — a finding that contradicts common practitioner intuition."

### Vocabulary register
- Write for a senior ML engineer, not for a theoretician. Avoid measure-theory notation, convergence proofs, and jargon that does not appear in practitioner tooling documentation.
- Define every acronym on first use. Spell out "gradient-boosted decision tree" before using GBDT.

---

## 5. How Results Are Presented in Accepted ADS Papers

### Core rule: every experiment paired with its significance and a practitioner takeaway.

Never end a results paragraph with a number. End it with what the number *means for someone who has to make a decision*.

### The finding-first pattern (used in ADS papers):

```
[Boldface claim / finding header]
[1–2 sentences of setup: what we measured and how]
[Key numbers / evidence — often a figure reference]
[1–2 sentences interpreting why this matters]
[Italicized or boxed PRACTITIONER TAKEAWAY: what to do differently]
```

Example:
> **Coverage-maximizing methods do not improve GBDT accuracy.**
> We ranked all 17 methods by their coverage of the original feature space and compared this ranking to post-training accuracy ranking. Figure 3 shows no positive correlation (Spearman ρ = −0.12, p = 0.61 across 15 datasets). This contradicts the common intuition — prevalent in the dataset distillation literature — that broader coverage yields better condensed datasets for downstream models.
> *Practitioner takeaway: Do not select a condensation method based on coverage metrics alone when the downstream model is a GBDT.*

### Narrative over tables

- Figures are primary; tables are supplementary.
- Use a figure when you want to show a pattern, trend, or comparison that a practitioner can immediately interpret visually (scatter plots, bar charts, rank plots).
- Use a table when you need exact numbers for reference or reproducibility.
- Every figure must be self-contained: axis labels, a legend, and a one-sentence caption that states the finding (not just what is plotted).
- **Figure 1 / the teaser figure**: Should convey the core insight of the paper in one visual. Many ADS reviewers form their impression from the teaser before reading the text.

### When to use a figure vs. a table:

| Use a FIGURE when... | Use a TABLE when... |
|---------------------|---------------------|
| Showing a trend or correlation | Reporting exact numbers needed for reproducibility |
| Comparing ranks or orderings across methods | Listing method properties or hyperparameters |
| Visualizing the diagnostic's predictive power | Showing ablation breakdowns with many configurations |
| Conveying a counter-intuitive finding (the "aha moment") | Providing a reference catalog of 17 methods |

### Avoid "table dumps"
A common ADS rejection reason is presenting a large comparison table with no narrative synthesis. Each table with more than 4 rows needs a companion paragraph that names the winner, explains why, and states the practical implication.

---

## 6. How Accepted ADS Papers Handle Limitations / Negative Results Without Undermining Themselves

### The principle: frame limitations as *scope*, not as *failures*.

The KDD ADS track explicitly accepts papers where "a conclusion has been reached that the problem is unsolvable" in a domain. Honest limitations are a feature, not a bug — they increase credibility with experienced reviewers.

### The three-part limitations formula:

1. **State the specific scope boundary**: "Our findings apply to gradient-boosted decision tree models on tabular classification and regression tasks. We have not evaluated condensation fidelity for neural networks or graph-structured data."
2. **Explain why this boundary exists** (briefly): "The diagnostic is designed around the inductive bias of tree-based models; extending it to neural networks would require a different fidelity metric."
3. **Point forward without over-promising**: "Future work could adapt the diagnostic to neural models; preliminary experiments suggest [X]."

### What not to do:
- Do not start the limitations section with "Despite these limitations..." — it signals defensiveness.
- Do not enumerate 8+ limitations — it signals lack of focus. Three to four specific, honest limitations are better than a long hedging list.
- Do not bury limitations in footnotes or the conclusion — put them in a named subsection.

### Handling the "no SOTA" issue honestly:
If reviewers might ask "why not compare to the new method X published last month?" — preempt this in the paper:
> "We evaluate the 17 methods that were publicly available and reproducible as of [DATE]; newer distillation methods for tree models are outside our current scope but are natural candidates for future evaluation using our diagnostic framework."

### Turning negative results into positive findings:
Every negative result is a positive claim about what *does not work*. State it as a positive finding:
- *Negative framing (weak)*: "Unfortunately, coverage-based methods did not perform well."
- *Positive framing (strong)*: "Coverage-based methods are systematically outperformed by density-based methods on GBDT tasks — a finding that challenges the dominant motivation in the condensation literature."

---

## 7. Top Reasons ADS/Applied Submissions Get Accepted vs. Rejected

### Acceptance signals (what reviewers reward):

1. **Named audience**: The paper states clearly who benefits and how. Vague "practitioners" is weaker than "ML engineers who retrain GBDT models on datasets exceeding 500K rows."
2. **Concrete, novel empirical finding**: Not "we studied X" but "we found Y, which contradicts the assumption that Z."
3. **Practical decision support**: The paper helps a practitioner make a specific decision they couldn't make before (e.g., which condensation method to use, when to trust a condensed dataset).
4. **Honest scope**: Limitations are specific and forward-pointing.
5. **Visual storytelling**: The teaser figure conveys the core insight immediately.
6. **Non-trivial comparison**: Baselines include reasonable competitive methods, not just random sampling.
7. **Evidential or deployment grounding**: For Evidential papers — a real-world problem context (not purely academic benchmark) is clearly stated.

### Desk-rejection triggers (automatic failure):

1. **No real-world grounding**: A paper "describing an algorithm or system tested solely on academic benchmark data" is explicitly called out as grounds for rejection without review (KDD ADS CFP, 2024).
2. **No practitioner audience specified**: Failure to identify who benefits.
3. **Purely research framing**: Introduction motivates from theoretical gaps, not practitioner costs.
4. **No quantified practical impact or fidelity metric**: For Deployed papers, no post-deployment performance numbers. For Evidential papers, no milestones or measurable insights.
5. **Formatting violations**: Page limit, template, and OpenReview profile requirements.

### Review-stage rejection triggers:

1. **Table dump without synthesis**: Large comparison table with no narrative explanation of what the results mean.
2. **Overclaiming**: Stating SOTA results when the contribution is analysis. Reviewers penalize mismatch between claims and evidence.
3. **No diagnostic value**: A study that describes what happens but does not help the reader predict or act differently.
4. **Weak motivation**: "This has not been studied before" is not sufficient; the paper must explain *why it matters that it hasn't been studied*.
5. **Missing practitioner takeaways**: Experiment sections that end with numbers but no guidance on what practitioners should do.
6. **Excessive hedging**: Findings stated so tentatively that reviewers cannot determine what was actually found.

---

## 8. Final Checklist: DO / DON'T Rules for THIS Paper

*(Measurement + diagnostic + honest limits, no SOTA claim — ICDM 2026 Applied Track)*

### DO

- **DO** open the Introduction with a concrete practitioner pain point: the cost of training full GBDT models, the existing informal practice of subsampling, and the absence of a principled pre-training fidelity check.
- **DO** label the paper explicitly as Evidential-category applied work (or its ICDM equivalent): "fundamental insights derived from addressing a significant real-world problem."
- **DO** name the target audience in the first paragraph: "ML engineers and data scientists who train gradient-boosted models (XGBoost, LightGBM, CatBoost) on tabular data."
- **DO** frame the diagnostic as the primary contribution: "a training-free metric that predicts post-training fidelity" — not as "a new algorithm."
- **DO** state every experiment result as a named finding (boldface claim header), followed immediately by what it means for a practitioner.
- **DO** lead Figure 1 with the diagnostic's predictive power or the most counter-intuitive finding (coverage ≠ accuracy) — this is what reviewers see first.
- **DO** include a dedicated "Practitioner Guidance" or "Implications" section that directly maps findings to decisions (e.g., "Given dataset size N and method class M, use the diagnostic score threshold T to decide whether to proceed").
- **DO** scope limitations specifically: "applies to GBDT models on tabular data; not evaluated on neural networks or time-series."

### DON'T

- **DON'T** frame the paper as a research-track contribution: never claim "state-of-the-art," never lead with a theoretical gap, and never position this as an algorithm paper.
- **DON'T** present a large comparison table (17 methods × 15 datasets) without a synthesis paragraph that names the key finding and its practitioner meaning. A table dump is the single most-cited ADS rejection reason.
- **DON'T** use the phrase "to the best of our knowledge, this is the first work to..." without immediately following it with *why it matters that no one has done it*. The gap must be motivated, not just asserted.
- **DON'T** bury the diagnostic in the methodology section as if it were just another experimental detail — it deserves its own section.
- **DON'T** hedge findings excessively. "Coverage does not predict GBDT accuracy in our study" is correct scoping; "coverage may not necessarily be the most reliable indicator of potential prediction quality" is over-hedged and signals uncertainty about the finding itself.
- **DON'T** start the abstract with "Dataset condensation is an important topic in machine learning..." — start with the practitioner situation.
- **DON'T** omit a Limitations section or fold it into the conclusion — ADS reviewers specifically look for honest scoping; absence reads as overclaiming.
- **DON'T** test methods only on standard academic benchmark datasets without explaining their connection to real-world tabular ML practice (regulatory constraints, data size, feature heterogeneity). Ground dataset selection in practitioner relevance.

---

## Sources Consulted

- KDD 2024 ADS Track Call for Papers: https://kdd2024.kdd.org/applied-data-science-ads-track-call-for-papers/
- KDD 2026 ADS Track Call for Papers: https://kdd2026.kdd.org/applied-data-science-ads-track-call-for-papers/
- KDD 2025 ADS Track Call for Papers: https://kdd2025.kdd.org/applied-data-science-ads-track-call-for-papers/
- KDD 2023 ADS Track Call for Papers: https://kdd.org/kdd2023/call-for-applied-data-science-ads-track-papers/
- KDD 2022 ADS Track Call for Papers: https://kdd.org/kdd2022/cfpAppliedDS.html
- ICDM 2026 Applied Track Call for Papers: http://icdm2026.neu.edu.cn/CallforAppliedTrackPapers/list.htm
- ECML-PKDD 2024 ADS Track: https://ecmlpkdd.org/2024/submissions-ads-track/
- DC-BENCH: Dataset Condensation Benchmark (NeurIPS 2022): https://arxiv.org/abs/2207.09639
- "Why do tree-based models still outperform deep learning on tabular data?" (NeurIPS 2022): https://arxiv.org/pdf/2207.08815
- "Highly Opinionated Advice on How to Write ML Papers" (Farquhar 2024): https://sebastianfarquhar.com/on-research/2024/11/04/how_to_write_ml_papers/
- KDD 2025 ADS Statistics, Paper Copilot: https://papercopilot.com/statistics/kdd-statistics/kdd-2025-statistics-applied-data-science-track/
- "Tabular Data Distillation: An Extensive Comparison" (2024): https://www.mdpi.com/2504-4990/8/4/84
- "On Learning Representations for Tabular Data Distillation" (2025): https://arxiv.org/pdf/2501.13905
- "TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks" (2024): https://arxiv.org/pdf/2406.19380
