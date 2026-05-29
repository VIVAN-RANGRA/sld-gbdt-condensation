# GainDistill: Split-Gain Landscape Distillation for Gradient-Boosted Tabular Models


> **Math rendering note:** This version uses GitHub/KaTeX-style math delimiters: inline math is written as `$...$` and display equations are written as `$$...$$`. If a Markdown viewer still does not render math, open the file in VS Code with a Markdown+Math extension, GitHub, Obsidian, Typora, or any KaTeX/MathJax-enabled viewer.

**Paper type:** CIKM/ICDM-style full research paper candidate  
**Primary area:** tabular data mining, data condensation, dataset distillation, gradient-boosted decision trees  
**Core claim:** small synthetic tabular datasets should preserve the *training mechanics* of gradient-boosted decision trees, not only feature distributions, neural gradients, or teacher predictions.

---

## 0. One-line thesis

Gradient-boosted decision trees construct models by comparing feature-threshold split gains and Newton leaf updates. **GainDistill** learns a compact synthetic tabular dataset whose split-gain landscape, split margins, and gradient/Hessian statistics approximate those of the full training data, so that tree ensembles trained on the distilled data make similar split decisions and retain downstream utility.

---

## 1. Title, abstract, and keywords

### Title

**GainDistill: Split-Gain Landscape Distillation for Gradient-Boosted Tabular Models**

### Abstract draft

Gradient-boosted decision trees remain among the strongest models for tabular prediction, yet most dataset distillation methods are designed around neural learners and optimize synthetic data through gradient, representation, trajectory, or distribution matching. This creates a mismatch between the distillation objective and the training mechanism of tree ensembles, whose construction is governed by feature-threshold split gains and leaf-wise first- and second-order statistics. We propose **GainDistill**, a model-aligned data condensation method for tabular learning. GainDistill learns a compact synthetic dataset by matching the split-gain landscape induced by the full data across selected boosting states and tree regions. The objective combines gain-distribution matching, split-margin preservation, Newton leaf-value matching, and gradient/Hessian histogram matching using a differentiable soft-bin representation of synthetic rows.

The method is designed to preserve the quantities that directly influence how gradient-boosted trees are built, rather than only preserving marginal feature distributions or neural surrogate behavior. We evaluate GainDistill on public OpenML and PMLB tabular benchmarks under multiple compression budgets and downstream learners, including XGBoost, LightGBM, CatBoost, Random Forests, and neural tabular models. Evaluation includes downstream predictive performance, split-structure fidelity, feature-importance agreement, cross-learner transfer, runtime, storage reduction, and ablation studies over each loss component. The experiments test whether preserving the split-gain landscape provides a more model-aligned condensation objective for tree-based tabular learning than random subset selection, coreset methods, generic distribution matching, and existing tabular distillation baselines.

### Keywords

Tabular data mining; dataset distillation; data condensation; gradient-boosted decision trees; split-gain matching; XGBoost; LightGBM; CatBoost; coreset selection; synthetic data; tree ensembles; model-aligned distillation.

---

## 2. Strict novelty boundary

This paper must **not** claim any of the following:

1. First dataset distillation method.
2. First tabular dataset distillation method.
3. First method to use trees for data compression.
4. First method to synthesize tabular data.
5. First method to preserve tree leaves or rule regions.

The safe and precise claim is:

> Existing tabular distillation methods usually preserve distributions, embeddings, selected examples, teacher logits, or neural gradients. GainDistill preserves the **candidate split-gain landscape**, **split-margin ordering**, and **Newton update statistics** used by gradient-boosted tree training.

The paper should be written as a **model-aligned distillation** paper, not as a generic synthetic-data paper.

---

## 3. Research questions

Use these RQs in the experiment section.

### RQ1: Downstream utility

Does GainDistill produce compact datasets that yield better downstream performance than random subsets, coreset selection, generic distribution matching, and tabular distillation baselines?

### RQ2: Split-structure fidelity

Does GainDistill preserve the split decisions, split rankings, and feature-importance structure of full-data gradient-boosted trees better than competing methods?

### RQ3: Compression behavior

How does performance vary as the synthetic budget changes from extremely small budgets to moderate compression ratios?

### RQ4: Cross-learner transfer

If GainDistill is optimized using an XGBoost-style teacher, does the distilled dataset transfer to LightGBM, CatBoost, Random Forests, and neural tabular learners?

### RQ5: Ablation and mechanism validation

Which terms in the objective are necessary: gain distribution matching, margin preservation, Newton leaf matching, histogram matching, marginal regularization, pairwise regularization, and soft-bin annealing?

### RQ6: Efficiency and practicality

What are the runtime, memory, and storage tradeoffs of distilling a dataset versus training directly on the full dataset?

---

## 4. Problem formulation

Let the full training dataset be:

$$
D = \{(x_i, y_i)\}_{i=1}^{n},
\quad
x_i \in \mathcal{X},
\quad
y_i \in \mathcal{Y}.
$$

The features may be numerical, categorical, binary, or missing-valued. We want to learn a much smaller weighted synthetic dataset:

$$
\tilde{D} = \{(\tilde{x}_a, \tilde{y}_a, \tilde{w}_a)\}_{a=1}^{m},
\quad
m \ll n,
\quad
\tilde{w}_a > 0.
$$

The target learner is a gradient-boosted decision-tree algorithm:

$$
A \in \{\text{XGBoost}, \text{LightGBM}, \text{CatBoost}\}.
$$

A generic distillation objective would be:

$$
\min_{\tilde{D}} \; \mathcal{L}_{test}(A(\tilde{D}), D_{test}),
$$

but this is not directly usable because the training procedure is discrete, non-differentiable, and expensive to unroll. GainDistill instead constructs a differentiable proxy around the quantities used by boosted trees during split construction:

$$
\min_{\tilde{D}}
\; d\big(\mathcal{S}(D), \mathcal{S}(\tilde{D})\big),
$$

where:

$$
\mathcal{S}(D)
=
\text{split-gain landscape induced by }D.
$$

The central hypothesis is:

> If $\tilde{D}$ approximates the split-gain landscape and Newton statistics of $D$, then a tree ensemble trained on $\tilde{D}$ will choose similar high-gain splits and learn similar leaf values.

---

## 5. Gradient-boosted tree mechanics to preserve

At boosting round $t$, the current ensemble prediction is:

$$
F_t(x).
$$

For differentiable loss $\ell(y, F(x))$, define first- and second-order derivatives:

$$
g_i^{(t)} =
\frac{\partial \ell(y_i, F_t(x_i))}{\partial F_t(x_i)},
$$

$$
h_i^{(t)} =
\frac{\partial^2 \ell(y_i, F_t(x_i))}{\partial F_t(x_i)^2}.
$$

For a tree node or region $R$, define aggregated Newton statistics:

$$
G_R^{(t)} = \sum_{i \in R} g_i^{(t)},
\quad
H_R^{(t)} = \sum_{i \in R} h_i^{(t)}.
$$

For candidate split $c=(j,b)$, where feature $j$ is split at bin or threshold $b$, define:

$$
R_L(c) = \{i \in R : x_{ij} \leq b\},
\quad
R_R(c) = R \setminus R_L(c).
$$

The regularized split gain is:

$$
\Gamma_D^{(t,R)}(c)
=
\frac{1}{2}
\left[
\frac{(G_L^{(t)})^2}{H_L^{(t)} + \lambda}
+
\frac{(G_R^{(t)})^2}{H_R^{(t)} + \lambda}
-
\frac{(G_0^{(t)})^2}{H_0^{(t)} + \lambda}
\right]
-
\gamma,
$$

where:

- $\lambda$ is leaf regularization,
- $\gamma$ is split penalty,
- $G_0,H_0$ are the parent-node statistics,
- $G_L,H_L$ are left-child statistics,
- $G_R,H_R$ are right-child statistics.

The vector over all candidate splits is the **split-gain landscape**:

$$
\Gamma_D^{(t,R)}
=
[\Gamma_D^{(t,R)}(c_1),\dots,\Gamma_D^{(t,R)}(c_M)].
$$

GainDistill tries to learn $\tilde{D}$ such that:

$$
\Gamma_{\tilde{D}}^{(t,R)} \approx \Gamma_D^{(t,R)}
$$

across selected boosting rounds $t$, tree regions $R$, and candidate splits $c$.

---

## 6. Method overview

GainDistill consists of eight stages:

1. Split data into train/validation/test.
2. Fit train-only feature bins.
3. Train a full-data teacher GBDT.
4. Select probe states $(t,R)$ from the teacher.
5. Compute full-data split-gain landscapes and gradient/Hessian sketches.
6. Parameterize a small synthetic dataset with differentiable soft-bin assignments.
7. Optimize the synthetic data to match full-data gain distributions, split margins, and Newton statistics.
8. Decode synthetic rows and train fresh downstream models on the distilled dataset.

---

## 7. Feature binning

For each numerical feature $j$, create quantile bins using only the training split:

$$
\mathcal{B}_j = \{b_{j,1},\dots,b_{j,B_j}\}.
$$

For categorical features, create category bins:

$$
\mathcal{B}_j = \{\text{cat}_{j,1},\dots,\text{cat}_{j,B_j}\}.
$$

Missing values are treated as a dedicated bin:

$$
b_{j,\text{miss}}.
$$

Each real sample is represented by bin assignments:

$$
z_{ij} \in \{1,\dots,B_j\}.
$$

Recommended defaults:

| Quantity | Default |
|---|---:|
| Numerical bins | 32 |
| Maximum categorical bins | 64 |
| Missing-value handling | separate bin |
| Rare categories | grouped into `__RARE__` |
| Very high-cardinality categorical feature | hash or top-k categories + rare bin |

---

## 8. Teacher model and probe states

Train a full-data teacher:

$$
T_D = \text{GBDT}(D_{train}).
$$

The teacher is used only to identify relevant boosting states and tree regions. It is **not** enough to match teacher predictions; the method needs tree-construction statistics.

Define probe set:

$$
\mathcal{P} = \{(t,R)\},
$$

where:

- $t$ is a boosting checkpoint,
- $R$ is a teacher node/region.

Recommended probe rounds:

$$
t \in \{0,1,5,10,25,50\}.
$$

Recommended probe regions:

1. root nodes,
2. depth-1 nodes,
3. depth-2 nodes,
4. high-gain internal nodes,
5. random internal nodes for coverage.

Do **not** match all nodes from all trees at first. Start with a compact probe set.

---

## 9. Full-data landscape computation

For each $(t,R) \in \mathcal{P}$, compute candidate split gains:

$$
\Gamma_D^{(t,R)}(j,b).
$$

Also compute gradient/Hessian cumulative histograms:

$$
\mathcal{H}_D^{(t,R)}(j,b)
=
[N_D^{(t,R)}(j,b), G_D^{(t,R)}(j,b), H_D^{(t,R)}(j,b)].
$$

where:

$$
N_D^{(t,R)}(j,b)
=
\sum_{i \in R} \mathbf{1}[x_{ij} \leq b],
$$

$$
G_D^{(t,R)}(j,b)
=
\sum_{i \in R} g_i^{(t)}\mathbf{1}[x_{ij} \leq b],
$$

$$
H_D^{(t,R)}(j,b)
=
\sum_{i \in R} h_i^{(t)}\mathbf{1}[x_{ij} \leq b].
$$

Use candidate split subset:

$$
\mathcal{C}_{t,R}
=
\mathcal{C}_{top}
\cup
\mathcal{C}_{hard}
\cup
\mathcal{C}_{rand}.
$$

Recommended:

| Candidate type | Count |
|---|---:|
| top-gain candidates | 32 |
| hard challenger candidates | 32 |
| random candidates | 64 |
| total per region | 128 |

Hard challengers are candidate splits whose gain is close to the best full-data split.

---

## 10. Synthetic data parameterization

Each synthetic row $a$ and feature $j$ is represented by a soft distribution over bins:

$$
p_{a,j}
=
\text{softmax}\left(\frac{\theta_{a,j}}{\tau}\right)
\in
\Delta^{B_j},
$$

where:

- $\theta_{a,j}$ are learnable bin logits,
- $\tau$ is a temperature,
- $\Delta^{B_j}$ is a probability simplex.

The probability that synthetic row $a$ goes left under threshold split $(j,b)$ is:

$$
A_{a,j,b}
=
\sum_{r \leq b} p_{a,j,r}.
$$

For categorical one-vs-rest split:

$$
A_{a,j,c}=p_{a,j,c}.
$$

Synthetic labels are soft during optimization.

For classification:

$$
\tilde{y}_a = \text{softmax}(\phi_a).
$$

Synthetic weights are positive:

$$
\tilde{w}_a = \text{softplus}(\rho_a) + \epsilon_w.
$$

The learnable parameters are:

$$
\Theta = \{\theta_{a,j}, \phi_a, \rho_a\}_{a=1}^{m}.
$$

At the end, decode:

$$
\tilde{z}_{a,j} = \arg\max_r p_{a,j,r}.
$$

Then map bins back to values:

- numerical: use bin midpoint or random value inside bin,
- categorical: use decoded category,
- missing bin: set value to missing.

---

## 11. Soft teacher routing

Tree routing is discrete. To make the objective differentiable, use soft routing through the frozen teacher.

For a teacher leaf $\ell$, define path probability for synthetic row $a$:

$$
q_{a,\ell}
=
\prod_{(j,b,s) \in \text{path}(\ell)} P_a(j,b,s),
$$

where:

$$
P_a(j,b,\text{left})=A_{a,j,b},
$$

$$
P_a(j,b,\text{right})=1-A_{a,j,b}.
$$

The soft teacher prediction at checkpoint $t$ is:

$$
\tilde{F}_t(\tilde{x}_a)
=
\sum_{k \leq t}
\eta
\sum_{\ell \in \text{leaves}(k)}
q_{a,\ell}v_{\ell},
$$

where:

- $v_\ell$ is the teacher leaf value,
- $\eta$ is the learning rate.

Then synthetic gradients and Hessians are:

$$
\tilde{g}_a^{(t)}
=
\frac{\partial \ell(\tilde{y}_a, \tilde{F}_t(\tilde{x}_a))}{\partial \tilde{F}_t(\tilde{x}_a)},
$$

$$
\tilde{h}_a^{(t)}
=
\frac{\partial^2 \ell(\tilde{y}_a, \tilde{F}_t(\tilde{x}_a))}{\partial \tilde{F}_t(\tilde{x}_a)^2}.
$$

For binary logistic loss, if $p_a^{(t)} = \sigma(\tilde{F}_t(\tilde{x}_a))$:

$$
\tilde{g}_a^{(t)} = p_a^{(t)} - \tilde{y}_a,
$$

$$
\tilde{h}_a^{(t)} = p_a^{(t)}(1-p_a^{(t)}).
$$

For multiclass tasks, use per-class softmax gradients and Hessian approximations.

---

## 12. Synthetic split-gain computation

Let $Q_{a,R}^{(t)}\in[0,1]$ be soft membership of synthetic row $a$ in teacher region $R$ at checkpoint $t$.

For candidate split $(j,b)$, synthetic left statistics:

$$
\tilde{G}_{L}^{(t,R)}(j,b)
=
\sum_{a=1}^{m}
\tilde{w}_a
Q_{a,R}^{(t)}
A_{a,j,b}
\tilde{g}_a^{(t)},
$$

$$
\tilde{H}_{L}^{(t,R)}(j,b)
=
\sum_{a=1}^{m}
\tilde{w}_a
Q_{a,R}^{(t)}
A_{a,j,b}
\tilde{h}_a^{(t)}.
$$

Right statistics:

$$
\tilde{G}_{R}^{(t,R)}(j,b)
=
\sum_{a=1}^{m}
\tilde{w}_a
Q_{a,R}^{(t)}
(1-A_{a,j,b})
\tilde{g}_a^{(t)},
$$

$$
\tilde{H}_{R}^{(t,R)}(j,b)
=
\sum_{a=1}^{m}
\tilde{w}_a
Q_{a,R}^{(t)}
(1-A_{a,j,b})
\tilde{h}_a^{(t)}.
$$

Parent statistics:

$$
\tilde{G}_{0}^{(t,R)}
=
\sum_{a=1}^{m}
\tilde{w}_a Q_{a,R}^{(t)}\tilde{g}_a^{(t)},
$$

$$
\tilde{H}_{0}^{(t,R)}
=
\sum_{a=1}^{m}
\tilde{w}_a Q_{a,R}^{(t)}\tilde{h}_a^{(t)}.
$$

Synthetic gain:

$$
\Gamma_{\tilde{D}}^{(t,R)}(j,b)
=
\frac{1}{2}
\left[
\frac{(\tilde{G}_{L})^2}{\tilde{H}_{L}+\lambda}
+
\frac{(\tilde{G}_{R})^2}{\tilde{H}_{R}+\lambda}
-
\frac{(\tilde{G}_{0})^2}{\tilde{H}_{0}+\lambda}
\right]
-
\gamma.
$$

---

## 13. GainDistill objective

The final loss is:

$$
\min_{\Theta}
\mathcal{L}(\Theta)
=
\mathcal{L}_{gain}
+
\lambda_1\mathcal{L}_{margin}
+
\lambda_2\mathcal{L}_{newton}
+
\lambda_3\mathcal{L}_{child}
+
\lambda_4\mathcal{L}_{hist}
+
\lambda_5\mathcal{L}_{class}
+
\lambda_6\mathcal{L}_{marg}
+
\lambda_7\mathcal{L}_{pair}
+
\lambda_8\mathcal{L}_{ent}.
$$

### 13.1 Gain distribution matching

Raw gains vary by dataset, round, and node. Convert each gain vector into a soft distribution:

$$
\pi_D^{(t,R)}(c)
=
\frac{\exp(\Gamma_D^{(t,R)}(c)/\tau_g)}
{\sum_{c'} \exp(\Gamma_D^{(t,R)}(c')/\tau_g)}.
$$

$$
\pi_{\tilde{D}}^{(t,R)}(c)
=
\frac{\exp(\Gamma_{\tilde{D}}^{(t,R)}(c)/\tau_g)}
{\sum_{c'} \exp(\Gamma_{\tilde{D}}^{(t,R)}(c')/\tau_g)}.
$$

Loss:

$$
\mathcal{L}_{gain}
=
\sum_{(t,R)\in\mathcal{P}}
\text{JS}\left(
\pi_D^{(t,R)} \;\|\; \pi_{\tilde{D}}^{(t,R)}
\right).
$$

### 13.2 Split-margin preservation

Let:

$$
c^* = \arg\max_c \Gamma_D^{(t,R)}(c).
$$

Full-data margin:

$$
\Delta_D(c) = \Gamma_D^{(t,R)}(c^*) - \Gamma_D^{(t,R)}(c).
$$

Synthetic margin:

$$
\Delta_{\tilde{D}}(c) = \Gamma_{\tilde{D}}^{(t,R)}(c^*) - \Gamma_{\tilde{D}}^{(t,R)}(c).
$$

Loss:

$$
\mathcal{L}_{margin}
=
\sum_{(t,R)}
\sum_{c\neq c^*}
\omega_c
\left[
\max\left(0,\alpha\Delta_D(c)-\Delta_{\tilde{D}}(c)\right)
\right]^2.
$$

This protects the top split from being displaced by close competitors.

### 13.3 Newton leaf-value matching

The regularized optimal leaf value is:

$$
v_R = -\frac{G_R}{H_R+\lambda}.
$$

Parent-region loss:

$$
\mathcal{L}_{newton}
=
\sum_{(t,R)}
\left(
\frac{G_R}{H_R+\lambda}
-
\frac{\tilde{G}_R}{\tilde{H}_R+\lambda}
\right)^2.
$$

Child-region loss for important candidate splits:

$$
\mathcal{L}_{child}
=
\sum_{(t,R)}
\sum_{c\in\mathcal{C}_{top}}
\left[
\left(
\frac{G_L(c)}{H_L(c)+\lambda}
-
\frac{\tilde{G}_L(c)}{\tilde{H}_L(c)+\lambda}
\right)^2
+
\left(
\frac{G_R(c)}{H_R(c)+\lambda}
-
\frac{\tilde{G}_R(c)}{\tilde{H}_R(c)+\lambda}
\right)^2
\right].
$$

### 13.4 Gradient/Hessian histogram matching

For each feature-bin candidate:

$$
\mathcal{H}_D^{(t,R)}(j,b)=[N,G,H],
\quad
\mathcal{H}_{\tilde{D}}^{(t,R)}(j,b)=[\tilde{N},\tilde{G},\tilde{H}].
$$

Loss:

$$
\mathcal{L}_{hist}
=
\sum_{(t,R)}
\sum_{j,b}
\left\|
\text{norm}(\mathcal{H}_D^{(t,R)}(j,b))
-
\text{norm}(\mathcal{H}_{\tilde{D}}^{(t,R)}(j,b))
\right\|_1.
$$

### 13.5 Class prior matching

$$
\mathcal{L}_{class}
=
\left\|
p_D(y)-p_{\tilde{D}}(\tilde{y})
\right\|_2^2.
$$

### 13.6 Feature marginal matching

$$
\mathcal{L}_{marg}
=
\sum_j
\text{JS}\left(
p_D(z_j) \;\|\; p_{\tilde{D}}(z_j)
\right).
$$

### 13.7 Pairwise interaction matching

Let $\mathcal{I}_{top}$ be top feature pairs selected by teacher gain, mutual information, or feature importance.

$$
\mathcal{L}_{pair}
=
\sum_{(j,k)\in\mathcal{I}_{top}}
\text{JS}\left(
p_D(z_j,z_k) \;\|\; p_{\tilde{D}}(z_j,z_k)
\right).
$$

### 13.8 Entropy annealing

$$
\mathcal{L}_{ent}
=
\sum_{a,j} H(p_{a,j}).
$$

Training starts with soft assignments and gradually makes them sharper.

---

## 14. Theoretical justification

### Proposition 1: Split preservation under gain approximation

Let $c^*$ be the best full-data split at a fixed probe state $(t,R)$:

$$
c^* = \arg\max_c \Gamma_D(c).
$$

Let the full-data split margin be:

$$
\delta = \Gamma_D(c^*) - \max_{c\neq c^*}\Gamma_D(c).
$$

Assume synthetic gain approximation satisfies:

$$
\sup_c |\Gamma_D(c)-\Gamma_{\tilde{D}}(c)| \leq \epsilon.
$$

If:

$$
\delta > 2\epsilon,
$$

then:

$$
\arg\max_c \Gamma_{\tilde{D}}(c)=\arg\max_c \Gamma_D(c).
$$

#### Proof sketch

For any $c\neq c^*$:

$$
\Gamma_{\tilde{D}}(c^*) \geq \Gamma_D(c^*)-\epsilon,
$$

$$
\Gamma_{\tilde{D}}(c) \leq \Gamma_D(c)+\epsilon.
$$

Therefore:

$$
\Gamma_{\tilde{D}}(c^*)-\Gamma_{\tilde{D}}(c)
\geq
\Gamma_D(c^*)-\Gamma_D(c)-2\epsilon.
$$

Since $\Gamma_D(c^*)-\Gamma_D(c)\geq\delta>2\epsilon$, the synthetic gain for $c^*$ is greater than for every alternative split.

### Proposition 2: Leaf-value stability under Newton-statistic matching

The regularized leaf value is:

$$
v(G,H)=-\frac{G}{H+\lambda}.
$$

Suppose:

$$
|G-\tilde{G}|\leq\epsilon_G,
\quad
|H-\tilde{H}|\leq\epsilon_H,
\quad
H+\lambda\geq\kappa>0.
$$

Then approximately:

$$
|v(G,H)-v(\tilde{G},\tilde{H})|
\leq
\frac{\epsilon_G}{\kappa}
+
\frac{|G|\epsilon_H}{\kappa^2}
+O(\epsilon_G\epsilon_H).
$$

This motivates matching gradient and Hessian statistics, not only split identity.

---

## 15. Algorithm pseudocode

### Algorithm 1: GainDistill

```text
Input:
  D_train, D_val, D_test
  synthetic budget m
  bin count B
  teacher learner A_teacher
  downstream learners A_eval
  probe rounds T_probe
  candidate split counts K_top, K_hard, K_rand
  optimization steps S

Output:
  distilled synthetic dataset D_tilde
  downstream results
  split-fidelity results

1. Fit feature binning scheme on D_train only.
2. Convert D_train into binned representation Z_train.
3. Train full-data teacher T_D = A_teacher(D_train).
4. Select probe states P = {(t, R)}:
      for t in T_probe:
          collect root, shallow nodes, high-gain nodes, and random nodes.
5. For each probe state (t, R):
      compute full-data gradients g_i^(t) and Hessians h_i^(t).
      compute candidate split gains Gamma_D^(t,R)(j,b).
      select C_(t,R) = top candidates + hard challengers + random candidates.
      store full-data N/G/H histograms and split-gain vectors.
6. Initialize synthetic parameters Theta = {theta, phi, rho}.
7. For step = 1,...,S:
      compute soft bin probabilities p_a,j = softmax(theta_a,j / tau).
      compute synthetic labels y_tilde_a = softmax(phi_a).
      compute synthetic weights w_tilde_a = softplus(rho_a).
      soft-route synthetic rows through frozen teacher to compute F_t(x_tilde_a).
      compute synthetic gradients and Hessians.
      compute synthetic split-gain landscape Gamma_tilde.
      compute L_gain, L_margin, L_newton, L_child, L_hist,
              L_class, L_marg, L_pair, L_ent.
      update Theta by Adam.
      anneal tau.
8. Decode hard synthetic rows from optimized soft-bin assignments.
9. Train fresh downstream learners on D_tilde.
10. Evaluate on D_test.
11. Compute split-fidelity metrics against full-data teacher.
12. Return D_tilde, metrics, logs, and tables.
```

---

## 16. Repository structure Codex should generate

```text
gaindistill/
  README.md
  requirements.txt
  pyproject.toml
  configs/
    default.yaml
    openml_cc18.yaml
    pmlb.yaml
    ablations.yaml
  data/
    raw/
    processed/
    splits/
    distilled/
  results/
    main/
    ablations/
    split_fidelity/
    transfer/
    runtime/
  src/
    gaindistill/
      __init__.py
      data/
        download_openml.py
        download_pmlb.py
        filters.py
        preprocess.py
        splits.py
      binning/
        quantile_binner.py
        categorical_binner.py
        decoder.py
      teacher/
        train_gbdt.py
        parse_trees.py
        probe_states.py
        gradients.py
      landscape/
        full_landscape.py
        candidate_splits.py
        histograms.py
        gain.py
      distill/
        synthetic_params.py
        soft_routing.py
        losses.py
        optimizer.py
        decode.py
      baselines/
        random_subset.py
        kcenter.py
        herding.py
        gradient_sampling.py
        distribution_matching.py
        tree_region_sampling.py
      eval/
        downstream.py
        metrics.py
        split_fidelity.py
        transfer.py
        runtime.py
      utils/
        seed.py
        io.py
        logging.py
        tables.py
  scripts/
    00_download_datasets.py
    01_preprocess.py
    02_train_teacher.py
    03_compute_landscape.py
    04_distill.py
    05_run_baselines.py
    06_train_downstream.py
    07_eval_split_fidelity.py
    08_run_ablations.py
    09_aggregate_results.py
  notebooks/
    sanity_check_one_dataset.ipynb
    visualize_gain_landscape.ipynb
  paper_tables/
    table_main_performance.csv
    table_split_fidelity.csv
    table_ablation.csv
    table_transfer.csv
    table_runtime.csv
```

---

## 17. Dependencies

Generate `requirements.txt` with:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
tqdm
pyyaml
openml
pmlb
xgboost
lightgbm
catboost
torch
joblib
```

Optional:

```text
wandb
optuna
tabpfn
rtdl-revisiting-models
```

Do not make optional packages required for the first working version.

---

## 18. Configuration template

Generate `configs/default.yaml`:

```yaml
seed: 42

paths:
  raw_data_dir: data/raw
  processed_data_dir: data/processed
  split_dir: data/splits
  distilled_dir: data/distilled
  results_dir: results

datasets:
  source: openml_cc18
  openml_suite_id: 99
  max_datasets: 30
  filters:
    min_rows: 1000
    max_rows: 200000
    max_features: 500
    max_classes: 10
    max_missing_fraction: 0.40
    classification_only: true

splits:
  method: stratified
  train_size: 0.70
  val_size: 0.15
  test_size: 0.15
  n_folds_small_data: 5
  small_data_threshold: 3000

binning:
  numerical_bins: 32
  max_categorical_bins: 64
  rare_category_min_count: 20
  missing_bin: true

teacher:
  learner: xgboost
  objective: auto
  n_estimators: 100
  max_depth: 4
  learning_rate: 0.10
  reg_lambda: 1.0
  gamma: 0.0
  subsample: 1.0
  colsample_bytree: 1.0

probe:
  rounds: [0, 1, 5, 10, 25, 50]
  include_root: true
  include_depths: [1, 2]
  high_gain_nodes_per_round: 8
  random_nodes_per_round: 8
  candidate_splits:
    top_k: 32
    hard_k: 32
    random_k: 64

synthetic:
  budget_type: per_class
  samples_per_class: [10, 25, 50, 100]
  init: class_balanced_kmeans
  optimize_labels: true
  optimize_weights: true
  min_weight: 1.0e-6

distill:
  steps: 3000
  optimizer: adam
  lr: 0.01
  weight_decay: 0.0
  initial_temperature: 2.0
  final_temperature: 0.10
  gain_temperature: 1.0
  anneal: cosine
  loss_weights:
    gain: 1.0
    margin: 1.0
    newton: 0.5
    child: 0.5
    hist: 0.5
    class: 0.1
    marginal: 0.1
    pairwise: 0.05
    entropy: 0.01

downstream:
  learners: [xgboost, lightgbm, catboost, random_forest, mlp]
  seeds: [0, 1, 2]

experiments:
  compression_budgets_per_class: [5, 10, 25, 50, 100]
  compression_ratios: [0.001, 0.005, 0.01, 0.05]
  run_main: true
  run_split_fidelity: true
  run_transfer: true
  run_ablations: true
  run_runtime: true
```

---

## 19. Dataset plan: open-source and downloadable

Use two public dataset sources.

### 19.1 Primary benchmark: OpenML-CC18

Use OpenML-CC18 as the main benchmark suite.

Recommended access route:

```python
import openml

suite = openml.study.get_suite(99)  # OpenML-CC18
for task_id in suite.tasks:
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y(dataset_format="dataframe")
```

Codex should implement robust fallback logic:

1. Try `openml.study.get_suite(99)`.
2. If this fails, search OpenML studies by name containing `OpenML-CC18`.
3. If task-based loading fails, use `openml.datasets.get_dataset(dataset_id)` from task metadata.
4. Cache every loaded dataset locally in Parquet format.

Filtering rules:

| Filter | Value |
|---|---:|
| minimum rows | 1,000 |
| maximum rows | 200,000 |
| maximum features | 500 |
| maximum classes | 10 |
| missing fraction | <= 40% |
| task type | classification |

The final paper should report exactly which datasets survived filtering.

### 19.2 Secondary benchmark: PMLB

Use PMLB as robustness benchmark.

Recommended access route:

```python
from pmlb import fetch_data, classification_dataset_names

for name in classification_dataset_names:
    X, y = fetch_data(name, return_X_y=True, local_cache_dir="data/raw/pmlb")
```

Filtering rules same as above.

### 19.3 Optional regression extension

Regression can be included if classification results are already strong.

Regression losses:

- squared error,
- gradient $g_i = F_t(x_i)-y_i$,
- Hessian $h_i = 1$.

Metrics:

- RMSE,
- MAE,
- $R^2$.

Do not include regression if it delays the classification paper.

---

## 20. Dataset download script specification

Codex should generate `scripts/00_download_datasets.py`.

CLI:

```bash
python scripts/00_download_datasets.py \
  --source openml_cc18 \
  --suite-id 99 \
  --max-datasets 30 \
  --out data/raw/openml_cc18

python scripts/00_download_datasets.py \
  --source pmlb \
  --max-datasets 20 \
  --out data/raw/pmlb
```

Expected outputs:

```text
data/raw/openml_cc18/
  metadata.csv
  <dataset_name>__<task_id>/
    X.parquet
    y.parquet
    meta.json

data/raw/pmlb/
  metadata.csv
  <dataset_name>/
    X.parquet
    y.parquet
    meta.json
```

`metadata.csv` columns:

```text
source,dataset_name,task_id,dataset_id,n_rows,n_features,n_classes,
missing_fraction,n_numeric,n_categorical,status,reason_if_skipped
```

Dataset loading must be deterministic and cached. Never redownload if files already exist unless `--force` is passed.

---

## 21. Preprocessing script specification

Codex should generate `scripts/01_preprocess.py`.

CLI:

```bash
python scripts/01_preprocess.py \
  --config configs/default.yaml \
  --dataset adult
```

Tasks:

1. Load raw Parquet dataset.
2. Detect numerical and categorical columns.
3. Remove constant columns.
4. Impute missing numerical values only for non-tree baselines; retain missing-bin representation for GainDistill.
5. Encode labels.
6. Create train/validation/test split.
7. Fit quantile bins on train only.
8. Save binned representation.
9. Save split indices.

Outputs:

```text
data/processed/<dataset>/
  X_train.parquet
  X_val.parquet
  X_test.parquet
  y_train.npy
  y_val.npy
  y_test.npy
  bins.json
  Z_train.npy
  Z_val.npy
  Z_test.npy
  feature_types.json
  split_indices.json
```

---

## 22. Teacher training script specification

Codex should generate `scripts/02_train_teacher.py`.

CLI:

```bash
python scripts/02_train_teacher.py \
  --config configs/default.yaml \
  --dataset adult \
  --teacher xgboost
```

Tasks:

1. Train full-data teacher on `D_train`.
2. Evaluate on validation and test.
3. Save teacher model.
4. Save raw margin predictions at selected boosting rounds.
5. Save gradients and Hessians at selected rounds.
6. Save tree dump in JSON format if available.

Outputs:

```text
results/teachers/<dataset>/
  teacher_xgboost.json
  teacher_metrics.json
  margins_round_<t>.npy
  gradients_round_<t>.npy
  hessians_round_<t>.npy
  tree_dump.json
```

Teacher defaults:

| Parameter | Value |
|---|---:|
| n_estimators | 100 |
| max_depth | 4 |
| learning_rate | 0.1 |
| reg_lambda | 1.0 |
| gamma | 0.0 |
| subsample | 1.0 |
| colsample_bytree | 1.0 |

---

## 23. Landscape computation script specification

Codex should generate `scripts/03_compute_landscape.py`.

CLI:

```bash
python scripts/03_compute_landscape.py \
  --config configs/default.yaml \
  --dataset adult \
  --teacher xgboost
```

Tasks:

1. Parse teacher tree dump.
2. Select probe states $(t,R)$.
3. Compute real sample membership in each region.
4. Compute candidate split gains for each probe state.
5. Select top, hard, and random candidate splits.
6. Store full-data gain vectors and N/G/H histograms.

Outputs:

```text
results/landscape/<dataset>/
  probe_states.json
  candidates.json
  full_gain_landscape.npz
  full_histograms.npz
  full_newton_stats.npz
```

Implementation detail:

Use binned features `Z_train` to compute cumulative statistics efficiently. For every feature $j$, sort or bucket by bin index once, then use cumulative sums of $g$ and $h$.

---

## 24. Distillation script specification

Codex should generate `scripts/04_distill.py`.

CLI:

```bash
python scripts/04_distill.py \
  --config configs/default.yaml \
  --dataset adult \
  --budget-per-class 50 \
  --seed 0
```

Tasks:

1. Load processed dataset.
2. Load teacher and landscapes.
3. Initialize synthetic parameters.
4. Optimize GainDistill objective.
5. Decode hard synthetic dataset.
6. Save synthetic rows, labels, weights, and logs.

Outputs:

```text
data/distilled/<dataset>/gaindistill/budget_<m>/seed_<s>/
  X_tilde.parquet
  y_tilde.npy
  w_tilde.npy
  Z_tilde.npy
  train_log.csv
  loss_curves.json
  config_resolved.yaml
```

Train log columns:

```text
step,total_loss,gain_loss,margin_loss,newton_loss,child_loss,hist_loss,
class_loss,marginal_loss,pair_loss,entropy_loss,temperature
```

---

## 25. Baseline generation script specification

Codex should generate `scripts/05_run_baselines.py`.

CLI:

```bash
python scripts/05_run_baselines.py \
  --config configs/default.yaml \
  --dataset adult \
  --budget-per-class 50 \
  --methods random_stratified,kcenter,herding,gradient_sampling,distribution_matching,tree_region
```

### Mandatory baselines

| Baseline | Description |
|---|---|
| `random_stratified` | class-balanced random subset |
| `kcenter` | diversity-based subset in standardized feature space |
| `herding` | class-wise moment matching |
| `gradient_sampling` | sample high-gradient examples from teacher |
| `goss_sampling` | high-gradient + sampled low-gradient examples |
| `distribution_matching` | synthetic data matching bin marginals and selected pairwise distributions |
| `tree_region_sampling` | sample rows/prototypes from teacher leaves or rule regions |
| `path_occupancy_matching` | match only tree path/leaf frequencies |
| `gaindistill_real` | select/reweight real rows using GainDistill objective where possible |
| `gaindistill_synthetic` | full proposed method |

### Optional baselines

| Baseline | Include when |
|---|---|
| TDColER-style baseline | if code is available and reproducible quickly |
| TabPFN in-context distillation | only for small datasets |
| MLP gradient matching | if generic dataset distillation implementation is stable |
| MLP trajectory matching | optional, not required for first submission |

---

## 26. Downstream evaluation script specification

Codex should generate `scripts/06_train_downstream.py`.

CLI:

```bash
python scripts/06_train_downstream.py \
  --config configs/default.yaml \
  --dataset adult \
  --distilled-root data/distilled/adult \
  --learners xgboost,lightgbm,catboost,random_forest,mlp
```

Downstream learners:

| Learner | Role |
|---|---|
| XGBoost | primary target learner |
| LightGBM | GBDT transfer |
| CatBoost | categorical and ordered-boosting transfer |
| Random Forest | non-boosted tree transfer |
| MLP | non-tree transfer sanity check |
| FT-Transformer | optional neural tabular transfer |

Metrics for classification:

| Metric | Use |
|---|---|
| Accuracy | balanced multiclass/small tasks |
| AUROC | binary and multiclass OVR |
| Macro-F1 | class imbalance |
| LogLoss | probabilistic quality |
| Balanced Accuracy | imbalanced datasets |

Metrics for regression, if included:

| Metric | Use |
|---|---|
| RMSE | main regression metric |
| MAE | robustness |
| R2 | normalized comparison |

Outputs:

```text
results/main/<dataset>/<method>/<budget>/<seed>/
  metrics.json
  predictions.npy
  model_summary.json
```

---

## 27. Split-fidelity evaluation script specification

Codex should generate `scripts/07_eval_split_fidelity.py`.

CLI:

```bash
python scripts/07_eval_split_fidelity.py \
  --config configs/default.yaml \
  --dataset adult \
  --method gaindistill \
  --budget-per-class 50
```

Train a fresh GBDT on the distilled dataset, then compare with the full-data teacher.

### Metrics

#### Root split agreement

$$
\text{RSA}
=
\mathbf{1}[c_D^{root}=c_{\tilde{D}}^{root}].
$$

#### Node-level split agreement

$$
\text{NSA}
=
\frac{1}{|\mathcal{R}|}
\sum_{R\in\mathcal{R}}
\mathbf{1}[c_D^R=c_{\tilde{D}}^R].
$$

#### Top-k split overlap

$$
\text{TopK}
=
\frac{1}{|\mathcal{R}|}
\sum_R
\frac{|\text{TopK}_D(R)\cap\text{TopK}_{\tilde{D}}(R)|}{K}.
$$

#### Gain-rank correlation

$$
\rho_{gain}
=
\text{Spearman}(\Gamma_D^{(t,R)},\Gamma_{\tilde{D}}^{(t,R)}).
$$

#### Feature-importance correlation

$$
\rho_{FI}
=
\text{Spearman}(FI(T_D),FI(T_{\tilde{D}})).
$$

Outputs:

```text
results/split_fidelity/<dataset>/<method>/<budget>/<seed>/
  split_fidelity.json
  gain_rank_correlations.csv
  feature_importance.csv
```

---

## 28. Ablation script specification

Codex should generate `scripts/08_run_ablations.py`.

CLI:

```bash
python scripts/08_run_ablations.py \
  --config configs/ablations.yaml \
  --dataset adult \
  --budget-per-class 50
```

### Ablation group A: loss components

| Variant | Change |
|---|---|
| full | all terms |
| no_gain | remove $\mathcal{L}_{gain}$ |
| no_margin | remove $\mathcal{L}_{margin}$ |
| no_newton | remove $\mathcal{L}_{newton}$ |
| no_child | remove $\mathcal{L}_{child}$ |
| no_hist | remove $\mathcal{L}_{hist}$ |
| no_marginal | remove $\mathcal{L}_{marg}$ |
| no_pairwise | remove $\mathcal{L}_{pair}$ |
| no_entropy_anneal | fixed high temperature or no entropy pressure |

### Ablation group B: probe states

| Variant | Meaning |
|---|---|
| root_only | only root nodes |
| early_only | rounds $0,1,5$ |
| late_only | rounds $25,50$ |
| high_gain_only | high-gain nodes only |
| random_nodes_only | random teacher nodes |
| full_probe | proposed |

### Ablation group C: candidate split coverage

| Variant | Meaning |
|---|---|
| top_only | top-gain candidates only |
| random_only | random candidates only |
| top_hard | top + hard challengers |
| top_hard_random | proposed |
| all_splits_small | all splits for small datasets |

### Ablation group D: synthetic parameterization

| Variant | Meaning |
|---|---|
| hard_bins | hard assignments from start |
| soft_annealed | proposed |
| real_coreset | select/reweight real rows |
| continuous_values | optimize continuous values directly |
| synthetic_real_hybrid | mix synthetic rows and selected real rows |

---

## 29. Result aggregation script specification

Codex should generate `scripts/09_aggregate_results.py`.

CLI:

```bash
python scripts/09_aggregate_results.py \
  --results-dir results \
  --out paper_tables
```

Tasks:

1. Read all `metrics.json` and split-fidelity files.
2. Aggregate over seeds.
3. Compute mean ± standard deviation.
4. Compute average rank across datasets.
5. Compute win/tie/loss counts.
6. Export CSV tables for paper.
7. Generate plots.

Outputs:

```text
paper_tables/
  table_1_main_performance.csv
  table_2_split_fidelity.csv
  table_3_low_budget.csv
  table_4_transfer.csv
  table_5_ablation.csv
  table_6_runtime.csv
  figure_1_compression_curves.png
  figure_2_split_fidelity_vs_accuracy.png
  figure_3_ablation_barplot.png
```

---

## 30. Main experiment matrix

### Datasets

Minimum:

- 20 OpenML-CC18 classification datasets.
- 10 PMLB classification datasets.

Better:

- 30 OpenML-CC18 classification datasets.
- 15 PMLB classification datasets.

Do not use too many datasets if it prevents ablations.

### Budgets

Use both absolute and ratio budgets.

Absolute budgets:

$$
5,10,25,50,100 \text{ samples per class}.
$$

Ratio budgets:

$$
0.1\%,0.5\%,1\%,5\%.
$$

For main results, use:

$$
10,25,50,100 \text{ samples per class}.
$$

### Seeds

Minimum:

$$
3 \text{ seeds}.
$$

If time is limited, run 3 seeds for main results and 1 seed for broad ablations.

---

## 31. Paper result tables

### Table 1: Main downstream performance

Use average rank plus normalized score.

| Method | XGBoost | LightGBM | CatBoost | RF | MLP | Avg Rank |
|---|---:|---:|---:|---:|---:|---:|
| Full data upper bound | TBD | TBD | TBD | TBD | TBD | TBD |
| Random subset | TBD | TBD | TBD | TBD | TBD | TBD |
| K-center | TBD | TBD | TBD | TBD | TBD | TBD |
| Herding | TBD | TBD | TBD | TBD | TBD | TBD |
| Gradient sampling | TBD | TBD | TBD | TBD | TBD | TBD |
| Distribution matching | TBD | TBD | TBD | TBD | TBD | TBD |
| Tree-region sampling | TBD | TBD | TBD | TBD | TBD | TBD |
| Path-occupancy matching | TBD | TBD | TBD | TBD | TBD | TBD |
| **GainDistill** | TBD | TBD | TBD | TBD | TBD | TBD |

### Table 2: Split-fidelity metrics

| Method | Root Agreement ↑ | Node Agreement ↑ | Top-5 Split Overlap ↑ | Gain Rank Corr ↑ | Feature Importance Corr ↑ |
|---|---:|---:|---:|---:|---:|
| Random subset | TBD | TBD | TBD | TBD | TBD |
| K-center | TBD | TBD | TBD | TBD | TBD |
| Distribution matching | TBD | TBD | TBD | TBD | TBD |
| Tree-region sampling | TBD | TBD | TBD | TBD | TBD |
| Path-occupancy matching | TBD | TBD | TBD | TBD | TBD |
| **GainDistill** | TBD | TBD | TBD | TBD | TBD |

### Table 3: Low-budget performance

| Budget per class | Random | K-center | Distribution matching | Tree-region | GainDistill |
|---:|---:|---:|---:|---:|---:|
| 5 | TBD | TBD | TBD | TBD | TBD |
| 10 | TBD | TBD | TBD | TBD | TBD |
| 25 | TBD | TBD | TBD | TBD | TBD |
| 50 | TBD | TBD | TBD | TBD | TBD |
| 100 | TBD | TBD | TBD | TBD | TBD |

### Table 4: Cross-learner transfer

| Distillation teacher | XGBoost | LightGBM | CatBoost | RF | MLP |
|---|---:|---:|---:|---:|---:|
| XGBoost teacher | TBD | TBD | TBD | TBD | TBD |
| LightGBM teacher | TBD | TBD | TBD | TBD | TBD |
| CatBoost teacher | TBD | TBD | TBD | TBD | TBD |

If only XGBoost teacher is implemented, table should be:

| Method | XGBoost | LightGBM | CatBoost | RF | MLP |
|---|---:|---:|---:|---:|---:|
| Random subset | TBD | TBD | TBD | TBD | TBD |
| Distribution matching | TBD | TBD | TBD | TBD | TBD |
| Tree-region | TBD | TBD | TBD | TBD | TBD |
| **GainDistill-XGB** | TBD | TBD | TBD | TBD | TBD |

### Table 5: Ablation

| Variant | Downstream Score ↑ | Split Agreement ↑ | Gain Corr ↑ | Feature FI Corr ↑ |
|---|---:|---:|---:|---:|
| Full GainDistill | TBD | TBD | TBD | TBD |
| No gain loss | TBD | TBD | TBD | TBD |
| No margin loss | TBD | TBD | TBD | TBD |
| No Newton loss | TBD | TBD | TBD | TBD |
| No child loss | TBD | TBD | TBD | TBD |
| No histogram loss | TBD | TBD | TBD | TBD |
| No marginal regularization | TBD | TBD | TBD | TBD |
| Hard bins from start | TBD | TBD | TBD | TBD |

### Table 6: Runtime and storage

| Dataset | Method | Compression | Distillation Time | Downstream Train Speedup | Performance Retention |
|---|---|---:|---:|---:|---:|
| TBD | Random | TBD | TBD | TBD | TBD |
| TBD | K-center | TBD | TBD | TBD | TBD |
| TBD | Tree-region | TBD | TBD | TBD | TBD |
| TBD | GainDistill | TBD | TBD | TBD | TBD |

---

## 32. Plots

Generate these plots:

1. **Compression curve:** x-axis = samples per class, y-axis = AUROC or accuracy.
2. **Split fidelity vs downstream utility:** x-axis = gain-rank correlation, y-axis = downstream score.
3. **Ablation bar chart:** method variants vs average rank.
4. **Runtime plot:** compression ratio vs train speedup.
5. **Feature-importance correlation plot:** GainDistill vs baselines.

---

## 33. Statistical reporting

For each dataset and method:

- run at least 3 seeds,
- report mean ± standard deviation,
- report average rank across datasets,
- report win/tie/loss count against strongest baseline,
- optionally use Wilcoxon signed-rank test across datasets.

Do not overclaim based on one dataset.

Paper language should be:

- “GainDistill improves average rank across the evaluated datasets.”
- “GainDistill shows stronger split-fidelity metrics.”
- “The gains are most pronounced under extreme compression.”

Avoid:

- “universally better,”
- “state-of-the-art” unless strongly justified,
- “first tabular distillation method.”

---

## 34. Expected result patterns

The method is worth continuing if these patterns appear:

### Strong signal

1. GainDistill has best or second-best average rank across many datasets.
2. GainDistill has clearly higher split agreement and gain-rank correlation.
3. Removing margin/Newton/histogram terms hurts split fidelity.
4. GainDistill performs especially well at 5–50 samples per class.
5. XGBoost-optimized distilled data transfers reasonably to LightGBM/CatBoost.

### Weak but salvageable signal

1. GainDistill mainly improves split fidelity but downstream utility is similar.
2. Paper can be reframed as structure-faithful tabular distillation, but this is weaker.

### Kill condition

Drop or radically revise the paper if:

1. random stratified subset beats GainDistill consistently,
2. k-center or herding beats GainDistill consistently,
3. split-fidelity metrics do not improve,
4. ablations do not show the gain/margin/Newton terms matter.

---

## 35. Minimal viable implementation order

Codex should implement in this order.

### Phase 1: One-dataset sanity pipeline

1. Download one OpenML dataset.
2. Preprocess and bin.
3. Train XGBoost teacher.
4. Compute root-level split gains.
5. Optimize synthetic data for root-only landscape.
6. Decode synthetic data.
7. Train downstream XGBoost on synthetic data.
8. Compare with random subset.

Success criterion:

- script runs end to end,
- synthetic rows decode correctly,
- no NaNs,
- downstream model trains.

### Phase 2: Full probe states

1. Parse teacher tree nodes.
2. Add depth-1 and depth-2 regions.
3. Add top/hard/random candidate splits.
4. Add margin/Newton/histogram losses.

Success criterion:

- gain-rank correlation improves over random subset on one dataset.

### Phase 3: Baselines

Implement:

1. random subset,
2. k-center,
3. herding,
4. gradient sampling,
5. distribution matching,
6. tree-region/path-occupancy baseline.

Success criterion:

- table generated for 3 datasets.

### Phase 4: Benchmark scaling

Run on:

- 10 OpenML datasets,
- then 20+ OpenML datasets,
- then PMLB subset.

### Phase 5: Final paper experiments

Run main, split fidelity, transfer, ablations, and runtime experiments.

---

## 36. Important implementation details

### 36.1 Binary classification gradients

For binary logistic loss:

$$
p_i=\sigma(F_t(x_i)),
\quad
g_i=p_i-y_i,
\quad
h_i=p_i(1-p_i).
$$

Clip Hessian:

$$
h_i \leftarrow \max(h_i, 10^{-6}).
$$

### 36.2 Multiclass handling

Initial implementation can either:

1. restrict to binary classification first, or
2. implement one-vs-rest gain landscapes for multiclass.

For speed, start with binary datasets. Expand to multiclass after the method works.

### 36.3 Synthetic weights

Normalize synthetic weights to match training size:

$$
\sum_a \tilde{w}_a = n_{train}.
$$

After each optimizer step:

```python
w = softplus(rho) + eps
w = w * (n_train / w.sum())
```

### 36.4 Numerical stability

Use:

```python
eps = 1e-8
hessian = torch.clamp(hessian, min=1e-6)
denominator = hessian_sum + lambda_reg + eps
```

### 36.5 Candidate split subsampling

Do not compute all splits for every dataset initially. Use candidate subset.

For each probe:

```text
candidates = top_32_by_gain + hard_32_close_to_top + random_64
```

### 36.6 Categorical features

Initial version:

- one-hot encode low-cardinality categoricals for downstream non-tree models,
- bin representation for GainDistill,
- category split handled as one-vs-rest candidate.

Advanced version:

- support category subsets like CatBoost/LightGBM.

Do not implement advanced categorical split subsets initially.

---

## 37. Evaluation details

### 37.1 Normalized score

For each dataset, define:

$$
\text{NormScore}(M)
=
\frac{S(M)-S(\text{random})}{S(\text{full})-S(\text{random})+\epsilon}.
$$

This measures how much of the full-data performance gap is recovered beyond random subset.

### 37.2 Performance retention

$$
\text{Retention}(M)
=
\frac{S(M)}{S(\text{full})}.
$$

Use only when metric scale makes sense.

### 37.3 Average rank

For each dataset and budget, rank methods by metric. Report average rank.

### 37.4 Win/tie/loss

Compare GainDistill against strongest baseline per dataset:

- win if score difference > one standard error,
- tie if within one standard error,
- loss otherwise.

---

## 38. Baseline implementation details

### 38.1 Random stratified subset

For each class $c$, sample $m_c$ examples uniformly.

### 38.2 K-center

Use standardized numerical features and one-hot categoricals. Select centers greedily per class.

### 38.3 Herding

For each class, select examples whose running mean approximates the class mean.

### 38.4 Gradient sampling

Compute teacher gradient magnitude:

$$
s_i=|g_i|.
$$

Sample highest-scoring examples per class.

### 38.5 GOSS-style sampling

Keep top $a\%$ gradient examples and randomly sample from lower-gradient examples.

### 38.6 Distribution matching

Optimize synthetic bin assignments to match:

- class prior,
- feature marginals,
- top pairwise feature distributions.

No split-gain, no Newton statistics.

### 38.7 Tree-region sampling

Train Random Forest or GBDT teacher. For high-support leaves:

- choose representative real row nearest to leaf centroid, or
- sample synthetic row from feature ranges observed in leaf.

This is a strong tree-specific baseline.

### 38.8 Path-occupancy matching

Match only:

$$
\Pr_D(\text{leaf}=\ell)
\approx
\Pr_{\tilde{D}}(\text{leaf}=\ell).
$$

This baseline tests whether split-gain matching is better than merely preserving leaf frequencies.

---

## 39. Paper writing plan

### Introduction

Key points:

1. Tabular data remains central in data mining.
2. GBDTs remain strong for tabular tasks.
3. Dataset distillation is mostly neural-aligned.
4. Existing tabular distillation is not explicitly aligned with split-gain construction.
5. GainDistill preserves the training statistics that tree ensembles actually use.

### Related Work

Organize into:

1. Dataset distillation and condensation.
2. Tabular dataset distillation.
3. Coresets and data reduction for tabular data.
4. Gradient-boosted decision trees.
5. Tree-based compression and rule/leaf-region methods.

### Method

Include:

1. GBDT split-gain background.
2. Split-gain landscape definition.
3. Soft-bin synthetic parameterization.
4. Soft teacher routing.
5. Loss function.
6. Theoretical propositions.
7. Algorithm.

### Experiments

Use RQs.

### Limitations

Be honest:

1. Optimization is slower than subset selection.
2. Distilled examples may not be naturally interpretable.
3. Method is model-aligned to tree ensembles, not universal.
4. High-cardinality categoricals require careful handling.
5. No formal privacy guarantee.

---

## 40. Codex execution commands

After scripts are generated, intended workflow:

```bash
# install
pip install -r requirements.txt

# download
python scripts/00_download_datasets.py --source openml_cc18 --suite-id 99 --max-datasets 30 --out data/raw/openml_cc18
python scripts/00_download_datasets.py --source pmlb --max-datasets 20 --out data/raw/pmlb

# preprocess all datasets
python scripts/01_preprocess.py --config configs/default.yaml --all

# train teachers
python scripts/02_train_teacher.py --config configs/default.yaml --all --teacher xgboost

# compute landscapes
python scripts/03_compute_landscape.py --config configs/default.yaml --all --teacher xgboost

# run proposed method
python scripts/04_distill.py --config configs/default.yaml --all --budget-per-class 50 --seed 0
python scripts/04_distill.py --config configs/default.yaml --all --budget-per-class 50 --seed 1
python scripts/04_distill.py --config configs/default.yaml --all --budget-per-class 50 --seed 2

# run baselines
python scripts/05_run_baselines.py --config configs/default.yaml --all --budget-per-class 50

# downstream evaluation
python scripts/06_train_downstream.py --config configs/default.yaml --all

# split fidelity
python scripts/07_eval_split_fidelity.py --config configs/default.yaml --all

# ablations
python scripts/08_run_ablations.py --config configs/ablations.yaml --datasets selected_5 --budget-per-class 50

# aggregate
python scripts/09_aggregate_results.py --results-dir results --out paper_tables
```

---

## 41. First-week go/no-go experiment

Before running the full benchmark, run a one-week go/no-go test.

Datasets:

- 5 OpenML binary classification datasets.

Methods:

1. Random stratified subset.
2. K-center.
3. Distribution matching.
4. Tree-region sampling.
5. GainDistill.

Budgets:

$$
10,25,50 \text{ samples per class}.
$$

Downstream:

- XGBoost,
- LightGBM.

Must measure:

1. AUROC/accuracy,
2. root split agreement,
3. top-5 split overlap,
4. gain-rank correlation.

Continue the paper only if GainDistill improves split fidelity and is competitive or better in downstream performance.

---

## 42. Final contribution statement

Use exactly these contributions in the paper:

1. **Split-gain landscape as a distillation target.** We define a model-aligned representation of tabular datasets based on the feature-threshold gain surface used by gradient-boosted tree construction.

2. **GainDistill algorithm.** We introduce a differentiable synthetic tabular data optimization procedure that matches gain distributions, split margins, Newton leaf statistics, and gradient/Hessian histograms using soft-bin assignments and soft teacher routing.

3. **Mechanism-focused evaluation.** We evaluate not only downstream accuracy, but also split-structure fidelity, gain-rank agreement, feature-importance correlation, cross-learner transfer, compression behavior, and ablations.

---

## 43. Do-not-overclaim checklist

Before final paper submission, check:

- [ ] Does the paper avoid claiming first tabular dataset distillation?
- [ ] Does it compare to tabular distillation or at least discuss TDColER/TDBench?
- [ ] Does it include a tree-region/path-occupancy baseline?
- [ ] Does it report split-fidelity metrics, not only accuracy?
- [ ] Does it include ablations showing gain/margin/Newton terms matter?
- [ ] Does it report all datasets and filtering rules?
- [ ] Does it avoid fake or cherry-picked results?
- [ ] Does it include limitations?

---

## 44. References to cite in paper draft

Add BibTeX later. Minimum related work categories:

1. XGBoost: scalable tree boosting and second-order split construction.
2. LightGBM: histogram-based GBDT.
3. CatBoost: ordered boosting and categorical handling.
4. Dataset distillation and condensation.
5. Tabular dataset distillation, especially TDColER/TDBench.
6. TabPFN/in-context data distillation.
7. Coresets and data reduction for decision trees.
8. Tree-region/leaf-hyperrectangle synthetic distillation.
9. OpenML benchmarking suites and OpenML-Python.
10. PMLB benchmark collection.

---

## 45. Final Codex instruction

When Codex uses this file, it should prioritize a working minimal pipeline over a perfect implementation. The first deliverable is:

```text
One OpenML dataset -> teacher -> root-level landscape -> GainDistill synthetic rows -> downstream XGBoost result -> split-fidelity result.
```

After that works, expand to:

```text
multiple probe states -> baselines -> multiple datasets -> ablations -> final tables.
```

No result should be written into the paper until it is produced by the experiment scripts.
