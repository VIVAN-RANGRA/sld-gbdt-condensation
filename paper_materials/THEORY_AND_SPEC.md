# Histogram-Aligned Distillation for Gradient-Boosted Tabular Models
## Theory and Specification

This document develops the formal core of the paper: the condensed object, the
Split Landscape Discrepancy (SLD) metric, the perturbation guarantee, the
submodular selection guarantee, the size-independent sample-complexity corollary,
and the method specification. It is written so the experiments and writeup can be
built directly on top of it.

All math is inline/display markdown. Notation is fixed in Section 0 and reused
throughout.

---

## 0. Notation and Setup

We study gradient-boosted decision trees (GBDT) of the XGBoost/LightGBM/CatBoost
family on tabular data.

- Dataset $D=\{(x_i,y_i)\}_{i=1}^{N}$, $x_i\in\mathbb{R}^{d}$, label $y_i$.
- Loss $\ell(y,\hat{y})$, twice differentiable in the margin $\hat{y}$.
- At a boosting state $t$ (current ensemble $F_t$), each row has
  - **gradient** $g_i^{(t)} = \partial_{\hat{y}}\,\ell\big(y_i,F_t(x_i)\big)$,
  - **Hessian** $h_i^{(t)} = \partial^2_{\hat{y}}\,\ell\big(y_i,F_t(x_i)\big)\ge 0$.
- Regularization $\lambda>0$ (leaf L2), $\gamma\ge 0$ (split penalty).

For a node holding instance set $I$, the optimal leaf weight and structure score
(the standard XGBoost second-order objective) are

$$
w^{*}(I) \;=\; -\frac{\sum_{i\in I} g_i}{\sum_{i\in I} h_i + \lambda},
\qquad
\mathrm{score}(I)\;=\;-\tfrac12\,\frac{\big(\sum_{i\in I} g_i\big)^2}{\sum_{i\in I} h_i+\lambda}.
$$

A **candidate split** $s=(j,\tau)$ tests feature $j$ at threshold $\tau$ and
partitions $I$ into $I_L(s)=\{i\in I: x_{ij}\le\tau\}$ and $I_R(s)=I\setminus I_L(s)$.
Writing $G_L=\sum_{i\in I_L}g_i$, $H_L=\sum_{i\in I_L}h_i$ (and $G_R,H_R$ likewise),
the **split gain** is

$$
\boxed{\;
\Gamma(s)\;=\;\tfrac12\!\left[\frac{G_L^{2}}{H_L+\lambda}
+\frac{G_R^{2}}{H_R+\lambda}
-\frac{(G_L+G_R)^{2}}{H_L+H_R+\lambda}\right]-\gamma .
\;}
\tag{1}
$$

GBDT induction is greedy: at each node it selects $s^\star=\arg\max_{s}\Gamma(s)$,
recurses on $I_L(s^\star),I_R(s^\star)$ up to depth $k$, then boosts.

---

## 1. The Core Observation: Histograms are a Sufficient Statistic for Split Gain

Discretize each feature $j$ into bins $b\in\{1,\dots,B_j\}$ (the binning a
histogram-based GBDT already performs). Define the per-node **gradient/Hessian
histogram**

$$
\mathcal{H}(I)=\Big\{\, \big(\,\mathrm{G}_{j,b},\,\mathrm{H}_{j,b}\,\big)\ :\
\mathrm{G}_{j,b}=\!\!\sum_{i\in I:\,x_{ij}\in\mathrm{bin}_{j,b}}\!\!g_i,\quad
\mathrm{H}_{j,b}=\!\!\sum_{i\in I:\,x_{ij}\in\mathrm{bin}_{j,b}}\!\!h_i \,\Big\}.
\tag{2}
$$

For any threshold placed at the boundary after bin $b$, the partition sums are
prefix sums of the histogram:

$$
G_L=\sum_{b'\le b}\mathrm{G}_{j,b'},\quad
H_L=\sum_{b'\le b}\mathrm{H}_{j,b'},\quad
G_R=G-G_L,\quad H_R=H-H_L,
\tag{3}
$$

with totals $G=\sum_b \mathrm{G}_{j,b}$, $H=\sum_b \mathrm{H}_{j,b}$. Substituting
(3) into (1) shows that **$\Gamma(s)$ depends on the data only through the
histogram $\mathcal{H}(I)$.** Hence:

> **Lemma 1 (Histogram sufficiency for split gain).**
> Two instance sets with identical gradient/Hessian histograms (2) induce
> identical split gains (1) for every bin-aligned candidate split. Consequently
> they induce the same greedy split, the same leaf weights, and the same one-step
> structure score.

*Proof.* Immediate: by (3) every quantity entering (1) is a prefix sum of
$\mathcal{H}(I)$; equal histograms give equal $(G_L,H_L,G_R,H_R)$ for all $s$,
hence equal $\Gamma(s)$, hence equal $\arg\max$ and equal $w^\star$. $\qquad\blacksquare$

**Why this matters (novelty).** Every prior tabular distillation/coreset method
condenses *rows* and then *hopes* the learner behaves similarly. Lemma 1 says the
histogram (2) is the **exact** object a tree consumes during split finding. So we
propose to distill the histogram directly. This is the first dataset-condensation
method whose compressed object is the sufficient statistic of the target learner
rather than a proxy. It also means split-fidelity is achievable *by construction*,
not merely measured after the fact.

---

## 2. The Split Landscape and Its Discrepancy

Let $\mathcal{S}$ be the full candidate-split set (all bin-aligned $(j,\tau)$).
The **split landscape** of a dataset $D$ at state $t$ and node $I$ is the map

$$
\Gamma_{D}^{(t,I)}:\ \mathcal{S}\to\mathbb{R},\qquad s\mapsto \Gamma_{D}^{(t,I)}(s).
$$

Because absolute gains scale with sample mass, we compare landscapes under a
**mass-normalized** convention: the condensed dataset $\tilde D$ carries
nonnegative weights so that its total Hessian mass matches the reference,
$\sum_i \tilde h_i = \sum_i h_i$ (Section 6 makes this exact). All gains below are
computed under this convention so the two landscapes live on the same scale.

> **Definition 1 (Split Landscape Discrepancy, SLD).**
> For datasets $D,\tilde D$ at a fixed state/node, and $p\in[1,\infty]$,
> $$
> \mathrm{SLD}_p\big(D,\tilde D\big)\;=\;\big\|\,\Gamma_{D}-\Gamma_{\tilde D}\,\big\|_p
> \;=\;\Big(\sum_{s\in\mathcal{S}}\big|\Gamma_{D}(s)-\Gamma_{\tilde D}(s)\big|^{p}\Big)^{1/p},
> $$
> with $\mathrm{SLD}_\infty=\max_{s\in\mathcal{S}}|\Gamma_D(s)-\Gamma_{\tilde D}(s)|$.

SLD is a pseudometric on datasets induced by the learner's own decision functional.
It is the quantity the paper proposes the community measure and minimize, replacing
ad-hoc fidelity proxies (root-agreement, top-k overlap, rank correlation) with one
principled axis. Those proxies become **derived** quantities: e.g. root agreement
$=\mathbb{1}[\arg\max\Gamma_D=\arg\max\Gamma_{\tilde D}]$, which Proposition 1 ties
directly to $\mathrm{SLD}_\infty$.

---

## 3. Perturbation Guarantee: Low SLD Preserves the Greedy Tree

### 3.1 Single node

> **Proposition 1 (Greedy split preservation at a node).**
> Fix a node with full-data best split $s^\star=\arg\max_{s}\Gamma_{D}(s)$ and
> **gain margin**
> $$
> \Delta \;=\; \Gamma_{D}(s^\star)-\max_{s\neq s^\star}\Gamma_{D}(s)\;>\;0.
> $$
> If $\mathrm{SLD}_\infty(D,\tilde D)<\Delta/2$, then
> $\arg\max_{s}\Gamma_{\tilde D}(s)=s^\star$: the condensed data selects the same
> split, with the same partition.

*Proof.* Let $\varepsilon=\mathrm{SLD}_\infty<\Delta/2$, so
$|\Gamma_{\tilde D}(s)-\Gamma_D(s)|<\varepsilon$ for all $s$. Then
$\Gamma_{\tilde D}(s^\star)>\Gamma_D(s^\star)-\varepsilon$, and for any
$s\neq s^\star$,
$$
\Gamma_{\tilde D}(s)<\Gamma_D(s)+\varepsilon\le\big(\Gamma_D(s^\star)-\Delta\big)+\varepsilon
=\Gamma_D(s^\star)-(\Delta-\varepsilon)<\Gamma_D(s^\star)-\varepsilon<\Gamma_{\tilde D}(s^\star).
$$
Hence $s^\star$ is the unique maximizer of $\Gamma_{\tilde D}$. $\qquad\blacksquare$

This converts the awkward empirical fidelity numbers into a *predicted* phenomenon:
agreement should hold exactly once SLD drops below half the margin, and degrade
gracefully (in $\arg\max$ probability) above it. The SLD-vs-AUROC scatter in the
experiments is the empirical test of this prediction.

### 3.2 Depth-$k$ trees

Deeper nodes see instances filtered by ancestor splits, so the relevant
comparison is the **conditional** landscape on the routed subset. Let
$\Gamma^{(v)}_{D}$ be the landscape restricted to instances reaching node $v$,
and $\Delta_v$ its margin.

> **Proposition 2 (Structural identity to depth $k$).**
> Suppose for every internal node $v$ of the full-data tree down to depth $k$,
> $$
> \mathrm{SLD}_\infty^{(v)}(D,\tilde D)<\Delta_v/2.
> $$
> Then greedy induction on $\tilde D$ produces a tree structurally identical to the
> full-data tree to depth $k$ (the same feature/threshold at every node).

*Proof.* Induction on depth. Depth 0 is Proposition 1 at the root. Suppose all
nodes to depth $m<k$ split identically; then the instance partition at every
depth-$(m{+}1)$ node is defined by the *same* thresholds on both datasets, so the
conditional landscapes $\Gamma^{(v)}$ are the correct objects to compare. Applying
Proposition 1 at each such $v$ gives an identical split. There are at most
$2^{k}-1$ internal nodes, completing the induction. $\qquad\blacksquare$

A high-probability variant replaces each premise by
$\Pr[\mathrm{SLD}^{(v)}_\infty<\Delta_v/2]\ge 1-\delta_v$ and concludes structural
identity with probability $\ge 1-\sum_v\delta_v$ (union bound). **This is exactly
why the method preserves conditional histograms at root / depth-1 / depth-2 probe
regions**: the probes are the nodes whose margins must be protected, giving the
previously heuristic "path/probe" machinery a precise theoretical role.

### 3.3 Honest scope: the boosting feedback loop

Propositions 1–2 govern a single tree at a fixed gradient state. Across boosting
rounds there is genuine feedback: the *student* trained on $\tilde D$ produces its
own predictions, so its gradients drift from the teacher's. We handle this with an
explicit, checkable assumption rather than hiding it.

> **Assumption A (teacher-anchored trajectory).**
> The student ensemble stays within margin $\rho$ of the teacher on the condensed
> support: $|F^{\mathrm{stu}}_t(\tilde x_i)-F^{\mathrm{tea}}_t(\tilde x_i)|\le\rho$
> for all $i$ and checkpoints $t\in\mathcal T$.

> **Proposition 3 (Per-checkpoint stability under Assumption A).**
> If $\ell$ has $\beta$-Lipschitz first/second margin-derivatives, then under
> Assumption A the gradient/Hessian perturbation is $O(\beta\rho)$ per row, hence
> the histogram perturbation is $O(\beta\rho)\cdot(\text{node mass})$ and, by the
> Lipschitz bound of Section 5, $\mathrm{SLD}_\infty^{(t)}=O(\beta\rho)$ at every
> checkpoint. Combined with Proposition 2 this preserves the per-checkpoint greedy
> tree whenever $O(\beta\rho)<\Delta_v/2$.

Warm-start anchoring (already in the method) is the mechanism that makes
Assumption A hold empirically; we report measured $\rho$ as a diagnostic. The
honest claim is therefore: **exact structural guarantees per tree/checkpoint, plus
a bounded-drift trajectory guarantee under a stated, measurable anchoring
condition.** Reviewers reward this over an overclaimed end-to-end theorem.

---

## 4. Submodular Selection with a $(1-1/e)$ Guarantee

Lemma 1 says *what* to preserve (the histogram). We now justify *how we select
rows* to preserve it, with an approximation guarantee for the greedy selector that
replaces the heuristic `gain_path_refined`.

Let $V$ be the pool of candidate real rows. For a row $i$ and split-side pair
$a=(s,\text{side})$, let $m_a(i)\ge 0$ be the (gradient+Hessian) **mass** row $i$
contributes to side $a$ of split $s$ — a nonnegative, additive (modular) quantity.
The aggregate mass of a selected set $S$ on side $a$ is the modular
$M_a(S)=\sum_{i\in S}m_a(i)$. Choose a per-side importance $\omega_a\ge 0$ (e.g.
the full-data gain of $s$) and a concave, nondecreasing **saturation**
$\phi:\mathbb{R}_{\ge0}\to\mathbb{R}_{\ge0}$ with $\phi(0)=0$ — canonically the
cap $\phi_a(z)=\min(z,\,c_a)$ where $c_a=M_a(V)$ is the full-data mass on side $a$.
Define the **split-coverage objective**

$$
f(S)\;=\;\sum_{a}\ \omega_a\,\phi_a\!\Big(\sum_{i\in S} m_a(i)\Big).
\tag{4}
$$

> **Proposition 4 (Submodularity and greedy guarantee).**
> $f$ in (4) is monotone submodular with $f(\emptyset)=0$. Therefore cardinality-
> constrained greedy selection of a budget-$B$ set $S_{\mathrm{gr}}$ satisfies
> $$
> f(S_{\mathrm{gr}})\;\ge\;\big(1-\tfrac1e\big)\,\max_{|S|\le B} f(S).
> $$

*Proof.* For fixed $a$, $z\mapsto\phi_a(z)$ is concave nondecreasing and
$M_a$ is modular nonnegative; the composition $S\mapsto\phi_a(M_a(S))$ is monotone
(both stages monotone) and submodular: for $A\subseteq B$, $i\notin B$,
$$
\phi_a(M_a(A)+m_a(i))-\phi_a(M_a(A))\ \ge\ \phi_a(M_a(B)+m_a(i))-\phi_a(M_a(B)),
$$
because $M_a(A)\le M_a(B)$ and $\phi_a$ has nonincreasing increments (concavity).
A nonnegative weighted sum of monotone submodular functions is monotone
submodular, so $f$ is. With $\phi_a(0)=0$ we get $f(\emptyset)=0$, and the
Nemhauser–Wolsey–Fisher theorem yields the $(1-1/e)$ bound. $\qquad\blacksquare$

### 4.1 Coverage controls SLD

The choice $\phi_a=\min(\cdot,c_a)$ ties (4) back to Definition 1. The **coverage
gap** is total weighted uncovered mass,
$$
\mathrm{gap}(S)=f(V)-f(S)=\sum_a \omega_a\,\max\!\big(0,\;c_a-M_a(S)\big),
$$
i.e. how much gradient/Hessian mass each split-side is under-represented by $S$.

> **Lemma 2 (Coverage gap bounds SLD).**
> On the domain where total gradient is bounded by $G_{\max}$ and $H+\lambda\ge\lambda>0$,
> the gain map (1) is $L$-Lipschitz in $(G_L,H_L,G_R,H_R)$ with
> $L=O\!\big(G_{\max}/\lambda + G_{\max}^2/\lambda^2\big)$. Hence
> $$
> \mathrm{SLD}_1(D,\tilde D)\ \le\ L\cdot \mathrm{gap}(S)\big/\min_a\omega_a .
> $$

*Proof sketch.* The partials of (1) are
$\partial\Gamma/\partial G_L = \tfrac{G_L}{H_L+\lambda}-\tfrac{G_L+G_R}{H+\lambda}$
and
$\partial\Gamma/\partial H_L = -\tfrac12\tfrac{G_L^2}{(H_L+\lambda)^2}+\tfrac12\tfrac{(G_L+G_R)^2}{(H+\lambda)^2}$
(symmetrically for $R$), each bounded on the stated domain, giving the Lipschitz
constant $L$. Each split's error in $(G_L,H_L,\dots)$ is at most its uncovered
mass; summing over $\mathcal S$ and dividing by the minimal importance gives the
bound. $\qquad\blacksquare$

**Consequence.** Greedy maximization of the submodular coverage $f$ is a
$(1-1/e)$-optimal *surrogate for SLD minimization*. Selection (Proposition 4) gets
the support right; the convex weight fit of Section 6 then drives SLD to (near)
zero. The previously isolated tricks — gain-sketch selection, weight fitting,
probe regions — assemble into one chain: **coverage $\Rightarrow$ low SLD
(Lemma 2) $\Rightarrow$ preserved greedy tree (Props 1–2).**

---

## 5. Size-Independent Sample Complexity

How small can the budget be? The headline message: it depends on the target
fidelity, **not** on $N$.

> **Theorem 1 (Budget for $\varepsilon$-fidelity).**
> Draw $B$ rows i.i.d. with importance $p_i\propto |g_i|+h_i$ and assign Horvitz–
> Thompson weights $w_i=1/(B p_i)$. Suppose per-row gradients/Hessians are bounded
> and $H+\lambda\ge\lambda>0$. Then for any $\delta\in(0,1)$, with
> $$
> B \;=\; O\!\left(\frac{V_{\max}}{\varepsilon^{2}}\,\log\frac{|\mathcal S|}{\delta}\right)
> $$
> (where $V_{\max}$ bounds the per-split mass variance), the weighted sample
> $\tilde D$ satisfies $\mathrm{SLD}_\infty(D,\tilde D)\le \varepsilon$ with
> probability $\ge 1-\delta$. The bound is **independent of the dataset size $N$.**

*Proof sketch.* For each split-side, $\hat G_L(s)=\sum_{\text{sampled }i\in I_L(s)}w_i g_i$
is an unbiased estimator of $G_L(s)$ (and likewise $\hat H_L$). Bernstein's
inequality gives $|\hat G_L(s)-G_L(s)|\le\varepsilon'$ w.p. $\ge1-\delta'$ for
$B=O(V_{\max}\varepsilon'^{-2}\log(1/\delta'))$. A union bound over the $O(|\mathcal S|)$
sums (the $R$ side is the complement, hence determined) controls all partition
sums simultaneously. Lemma 2's Lipschitz constant transfers the sum error to the
gain error: $\mathrm{SLD}_\infty\le L\varepsilon'$. Set $\varepsilon'=\varepsilon/L$
and $\delta'=\delta/|\mathcal S|$. $\qquad\blacksquare$

> **Corollary 1 (Compression statement).**
> To preserve the split landscape to fidelity $\varepsilon$ — and therefore, by
> Propositions 1–2, the greedy tree structure wherever the margin exceeds
> $2\varepsilon$ — it suffices to keep $\tilde O(\varepsilon^{-2})$ weighted rows,
> regardless of how large the original dataset is.

This is the quotable result: **fidelity, not data volume, sets the budget.** It
explains why tiny condensed sets (25–50/class) already recover most of the signal,
and predicts diminishing returns past the $\varepsilon^{-2}$ regime — directly
testable on the budget curve.

Importance sampling (Theorem 1) is the *theoretical* selector; the submodular
greedy of Section 4 is the *deployed* selector that additionally enforces
explicit, deterministic coverage (better at very small $B$, where i.i.d. sampling
is high-variance). We present importance sampling as the analyzable baseline and
greedy+weights as the practical method, with the same fidelity target.

---

## 6. The Condensed Object and Materialization

### 6.1 Object

The condensed artifact is a **weighted histogram bundle** over checkpoints
$t\in\mathcal T$ and probe nodes $v$:

$$
\tilde{\mathcal{C}}=\Big\{\,(\tilde x_i,\,\tilde y_i,\,w_i)\,\Big\}_{i=1}^{B}
\quad\text{such that}\quad
\mathcal{H}_{\tilde{\mathcal C}}^{(t,v)}\approx \mathcal{H}_{D}^{(t,v)}\ \ \forall (t,v),
$$

with weights $w_i\ge0$ and the **mass-matching constraint**
$\sum_i w_i\,\tilde h_i^{(t)}=\sum_i h_i^{(t)}$ that makes landscapes comparable
(Section 2). Storing rows (not raw histograms) keeps the artifact usable by
**any** downstream learner; the histograms are the optimization target.

### 6.2 Exact matching by convex weight fit

Given a selected support, fitting weights to match histograms is a **nonnegative
least squares** problem,
$$
\min_{w\ge0}\ \sum_{t,v}\big\|\,A^{(t,v)}w-h^{(t,v)}_{D}\,\big\|_2^2
\;+\;\eta\,\|w\|_2^2,
\tag{5}
$$
where $A^{(t,v)}$ maps row weights to the bundle's gradient/Hessian bin sums and
$h^{(t,v)}_D$ stacks the full-data targets. Problem (5) is convex; its optimum is
the SLD-minimizing reweighting of the chosen support and supplies the
$\mathrm{gap}(S)\to 0$ regime of Lemma 2.

### 6.3 Materialization for row-based learners

Trees can consume the bundle directly; RF/MLP need rows. We materialize by
emitting each support row $\tilde x_i$ with sample weight $w_i$ (or by integer
expansion / weighted resampling to $B'$ unweighted rows for learners that ignore
weights). Section 7.3 corrects the marginal density these learners depend on.

---

## 7. Methods

Three deployed methods, all producing the object of Section 6.

### 7.1 HistDistill-Greedy
Greedy maximization of the split-coverage objective (4) to budget $B$, then convex
weight fit (5). Carries the $(1-1/e)$ guarantee (Prop 4) and drives SLD via Lemma 2.
Replaces `gain_path_refined` as the primary method.

### 7.2 HistDistill-Refined
HistDistill-Greedy plus local support refinement: swap moves that reduce the
NNLS residual (5) while preserving feasibility. Monotone in the SLD surrogate;
inherits the guarantee as a warm start.

### 7.3 HistDistill-Density (cross-learner / MLP correction)
Tree fidelity does not constrain the joint marginal $P(x)$ that row-based learners
(MLP) rely on. We make this an explicit **multi-objective**:
$$
\min_{\tilde{\mathcal C}}\ \underbrace{\mathrm{SLD}_p(D,\tilde D)}_{\text{tree fidelity}}
\;+\;\beta\,\underbrace{\mathrm{MMD}^2\!\big(P_X,\,P_{\tilde X}\big)}_{\text{density fidelity}}.
\tag{6}
$$
$\beta=0$ recovers pure histogram distillation; $\beta>0$ trades a little tree
fidelity for marginal coverage and is the variant that closes the MLP transfer
gap. The Pareto frontier of (6) is itself a paper figure: it quantifies the
tree-vs-density tension that makes cross-learner transfer hard.

### 7.4 Deprecated / appendix
The purely synthetic soft-bin optimizer is reported only as an ablation (it
underperforms selection). The differentiable soft-tree surrogate + trajectory
matching is named as future work (it is the route to a fully synthetic method but
is out of scope for this cycle).

---

## 8. What Each Theoretical Result Buys the Paper

| Result | Plugs which hole in the current draft |
| --- | --- |
| Lemma 1 (sufficiency) | Turns "we hope to preserve gains" into "the histogram **is** the gain functional"; makes split-fidelity a design target, not a weak measurement |
| Definition 1 (SLD) | Replaces ad-hoc fidelity proxies with one principled, reusable metric |
| Prop 1–2 (perturbation) | Formal link "low SLD ⇒ same tree"; recontextualizes the weak 0.35 rank-corr numbers as points on a predicted curve |
| Prop 3 + Assumption A | Honest, measurable treatment of the boosting feedback loop instead of an overclaim |
| Prop 4 + Lemma 2 (submodular) | $(1-1/e)$ guarantee for the deployed greedy selector; unifies selection + weighting + probes into one chain |
| Theorem 1 + Cor 1 (sample complexity) | Quotable headline: budget is $\tilde O(\varepsilon^{-2})$, **independent of $N$** |
| Eq. (6) (multi-objective) | Principled fix + figure for the MLP transfer gap |

---

## 9. Open Theoretical Risks (to state in the paper, not hide)

1. **Bin alignment.** Guarantees are for bin-aligned thresholds (the
   histogram-GBDT regime). Exact-greedy thresholds between bins are approximated;
   note the standard hist-mode assumption.
2. **Categorical splits.** Subset splits are not threshold splits; we reduce them
   via one-hot/target encoding so (1) applies, and flag native categorical
   handling as future work.
3. **Margin assumption.** Prop 1 needs $\Delta>0$ (unique best split). Near-ties
   degrade gracefully but are not covered by the exact statement; report the
   empirical margin distribution.
4. **Trajectory drift.** Prop 3 is conditional on Assumption A; we report measured
   $\rho$ and show anchoring keeps it small, but it is an assumption, not a
   theorem about arbitrary students.

---

## 10. Minimal Experimental Hooks Implied by the Theory

These are the measurements the theory demands (so the experiments test the theory,
not just performance):

1. **SLD computation** at root + depth-1/2 probes, per checkpoint, for every
   method — the new fidelity column.
2. **SLD-vs-AUROC scatter** across all (method, dataset, budget, seed): tests
   Props 1–2 (expect monotone trend; our methods on the Pareto front).
3. **Margin distribution** $\Delta_v$ per dataset: contextualizes the $\Delta/2$
   threshold in Prop 1.
4. **Budget curve vs $\varepsilon$** ($\mathrm{SLD}$ as a function of $B$): tests
   the $\tilde O(\varepsilon^{-2})$, $N$-independent claim of Theorem 1 (overlay
   datasets of very different $N$ — curves should collapse).
5. **Anchoring diagnostic** $\rho$ vs boosting round: supports Assumption A.
6. **MMD–SLD Pareto frontier** (Eq. 6): supports the HistDistill-Density story and
   the MLP gap closure.
