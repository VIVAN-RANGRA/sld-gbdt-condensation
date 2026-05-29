# Literature Review: SLD / GAIN_DISTIL Paper

References grouped by theme, each with a 2–3 sentence summary and a sentence
on relevance to our paper. BibTeX keys match `references.bib`.

---

## 1. Dataset Distillation / Condensation (Image-Domain Foundations)

**Positioning.** This body of work motivated the core idea that a small
synthetic set can replace a large training set without significant accuracy
loss. Our paper is explicitly *not* an image distillation paper—we operate on
tabular data with GBDTs—but we import terminology, evaluation protocols, and
method categories (optimisation-based vs. distribution-matching vs. trajectory-
matching) from this literature. Our Split-Landscape Discrepancy (SLD) metric is
novel precisely because none of these image-domain metrics transfer to the tree
split-gain setting.

### `wang2018datasetdistillation`
Wang, Zhu, Torralba, Efros. *Dataset Distillation.* arXiv 1811.10959, 2018.

The paper introduces dataset distillation—optimising a small synthetic set
so that a model trained on it matches performance of a model trained on the full
set, using a bilevel meta-learning objective. It demonstrated on MNIST/CIFAR
that models trained on fewer than 10 synthetic images per class can approach
full-dataset accuracy. This is the founding reference for the concept of
distilling a dataset into synthetic examples, which our paper extends to the
tabular/GBDT setting.

### `zhao2021dcgm`
Zhao, Mopuri, Bilen. *Dataset Condensation with Gradient Matching.* ICLR 2021.

Reformulates dataset distillation as matching the gradients of model weights
trained on synthetic vs. real data, achieving strong compression ratios on
CIFAR-10/100. The method was an ICLR 2021 Oral and introduced the term "dataset
condensation." Our evaluation includes gradient-matching-style condensation
adapted to tabular data; SLD measures whether the approach preserves the split
landscape.

### `zhao2021dsa`
Zhao, Bilen. *Dataset Condensation with Differentiable Siamese Augmentation.*
ICML 2021.

Proposes differentiable Siamese augmentation (DSA) to augment both real and
synthetic batches identically during gradient matching, yielding 7% accuracy
gains on CIFAR. DSA demonstrates that augmentation consistency matters for
condensation quality. We include DSA-style methods as baselines and examine
whether their accuracy gains correspond to better SLD scores.

### `zhao2023dm`
Zhao, Bilen. *Dataset Condensation with Distribution Matching.* WACV 2023.

Synthesises condensed images by matching feature distributions in random
embedding spaces, substantially reducing synthesis cost. Distribution matching
is a training-free criterion at distillation time, analogous in spirit to our
training-free SLD diagnostic. We compare distribution-matching methods on
tabular data and find SLD provides complementary diagnostic power.

### `cazenavette2022mtt`
Cazenavette, Wang, Torralba, Efros, Zhu. *Dataset Distillation by Matching
Training Trajectories.* CVPR 2022.

Optimises distilled data so that networks trained on it trace similar parameter
trajectories to networks trained on real data; pre-computes "expert
trajectories" for efficiency. MTT substantially outperforms earlier methods on
high-resolution datasets. As a trajectory-based method it is among the most
expensive condensation approaches; SLD lets us evaluate trajectory fidelity
without re-running the full pipeline.

### `nguyen2021kip`
Nguyen, Chen, Lee. *Dataset Meta-Learning from Kernel Ridge-Regression.*
ICLR 2021.

Introduces Kernel Inducing Points (KIP): learning synthetic data that minimises
the kernel ridge-regression loss under the neural-tangent-kernel correspondence.
KIP achieves state-of-the-art dataset distillation for KRR tasks and transfers
to finite-width networks. We reference KIP as a theoretically motivated baseline
that connects distillation to kernel methods; our tabular analogue uses
histogram split-gain kernels.

### `liu2022factorization`
Liu, Wang, Yang, Ye, Wang. *Dataset Distillation via Factorization.* NeurIPS
2022.

Decomposes distilled data into hallucination networks and bases, giving
exponentially richer synthetic sets from a small parameter budget. The HaBa
factorisation framework shows that representation choice strongly affects
distillation quality. We cite this as a recent state-of-the-art image method
that our tabular-GBDT framework does not directly port, illustrating the
domain gap that motivates our work.

---

## 2. Dataset Distillation / Condensation Surveys

**Positioning.** Three independent survey papers appeared in January 2023
consolidating the dataset distillation field. Our paper positions itself within
the gap they collectively identify: the lack of interpretable, training-free
quality metrics for condensed data, particularly for non-image modalities.

### `lei2023survey`
Lei, Tao. *A Comprehensive Survey of Dataset Distillation.* IEEE TPAMI, 2023.

Surveys distillation frameworks, factorised methods, and performance comparisons
across modalities, proposing a taxonomic algorithmic framework. The survey
explicitly notes that evaluation remains dominated by downstream accuracy, with
no standard for measuring mechanistic fidelity. SLD addresses exactly this gap.

### `sachdeva2023survey`
Sachdeva, McAuley. *Data Distillation: A Survey.* arXiv 2301.04272, 2023.

Provides a broad survey treating data distillation as producing terse
summaries usable as drop-in replacements for the original dataset in training,
inference, and architecture search. The survey discusses privacy and
communication motivations. We reference it to contextualise the range of
downstream uses where SLD-based pre-selection could save cost.

### `yu2023review`
Yu, Liu, Wang. *Dataset Distillation: A Comprehensive Review.* IEEE TPAMI,
2023.

Reviews and unifies the algorithmic landscape of dataset distillation with
analysis of effectiveness and efficiency trade-offs. The review calls for
metrics that go beyond accuracy proxy, motivating our SLD diagnostic. Our work
contributes one concrete mechanism-fidelity metric for the tabular/tree setting.

---

## 3. Tabular Dataset Distillation / Condensation

**Positioning.** Only a handful of papers target distillation for tabular data
specifically, and all of them evaluate primarily via downstream accuracy. Our
paper is the first to introduce a training-free, mechanism-aware quality score
for tabular condensation methods.

### `kang2024tabdistill`
Kang, Ram, Zhou, Samulowitz, Seneviratne. *Effective Data Distillation for
Tabular Datasets (Student Abstract).* AAAI 2024.

Applies several image-domain distillation schemes to tabular data, finding that
k-means clustering in learned representation spaces achieves the best distilled
data quality. The paper establishes that naive image distillation methods do not
transfer well to tabular data. Our work benchmarks a superset of the methods
they consider and provides a mechanistic explanation via SLD decomposition.

### `kang2025tabdistill`
Kang, Ram, Zhou, Samulowitz, Seneviratne. *On Learning Representations for
Tabular Data Distillation.* TMLR 2025.

Presents TDColER, a column-embeddings framework for tabular distillation, and
TDBench, a comprehensive benchmark with 226,890 distilled datasets. TDColER
boosts distilled data quality 0.5–143% over off-the-shelf methods across seven
tabular learners. Our SLD metric is complementary: TDBench evaluates via
downstream accuracy; SLD evaluates split-landscape fidelity before any model is
retrained.

### `herurkar2024tabdist`
Herurkar, Raue, Dengel. *Tab-Distillation: Impacts of Dataset Distillation on
Tabular Data for Outlier Detection.* ICAIF 2024.

Applies standard dataset distillation to tabular data for outlier detection,
finding that distillation effectively addresses class imbalance and improves
class separation. This is one of the few application papers on tabular
distillation beyond classification. We reference it as evidence that tabular
distillation has value beyond standard classification accuracy.

### `medvedev2020tabular`
Medvedev, D'yakonov. *New Properties of the Data Distillation Method When
Working with Tabular Data.* AIST 2020 (Springer LNCS, 2021).

One of the earliest papers applying the Wang et al. (2018) distillation
procedure to tabular data, showing that distilled samples can outperform the
full training set and that multi-architecture distillation improves
generalisation. It surfaces the challenge that distilled tabular data is
architecture-sensitive, a problem our SLD diagnostic helps diagnose without
retraining.

### `xu2025c2tc`
Xu, Li, Wang, Yang, Lin. *C2TC: A Training-Free Framework for Efficient
Tabular Data Condensation.* arXiv 2602.21717, to appear ICDE 2026.

Proposes the first training-free tabular condensation framework jointly
optimising class allocation and feature representation for scalable condensation.
The concurrent C2TC work shares our training-free philosophy but targets
efficiency rather than mechanism diagnostics. Our SLD score could serve as an
evaluation layer on top of C2TC-generated sets.

---

## 4. Coreset Selection

**Positioning.** Coreset selection is the geometric / optimisation-based
alternative to generative distillation. Several of these methods (Herding,
k-center, CRAIG) appear among the ~17 condensation methods we benchmark. A key
finding of our paper is that methods with theoretically optimal *coverage*
(k-center) score poorly on SLD and downstream accuracy—"coverage is not
accuracy"—directly contradicting the intuition behind coverage-based coresets.

### `welling2009herding`
Welling. *Herding Dynamical Weights to Learn.* ICML 2009.

Introduces herding, a deterministic pseudo-sampling algorithm that converts
observed moments into a sequence of pseudo-samples respecting moment
constraints, without explicit model learning. Herding is one of the original
coreset-style methods used as a baseline in dataset condensation evaluations.
We include it as one of the ~17 methods evaluated with SLD.

### `sener2018kcenter`
Sener, Savarese. *Active Learning for Convolutional Neural Networks: A Core-Set
Approach.* ICLR 2018.

Formulates active learning as greedy k-center coreset selection, choosing a
subset that minimises the maximum distance between any training point and its
nearest selected point. k-center maximises geometric coverage by design, making
it a key subject of our "coverage is not accuracy" finding; the paper provides
the theoretical link between coverage and expected performance.

### `mirzasoleiman2020craig`
Mirzasoleiman, Bilmes, Leskovec. *Coresets for Data-efficient Training of
Machine Learning Models (CRAIG).* ICML 2020.

Proposes CRAIG, which selects a weighted coreset maximising a submodular
facility-location function that approximates the full gradient. Proven to
converge at the same rate as full-gradient methods for convex problems.
CRAIG is one of the gradient-matching coreset methods evaluated in our
benchmark, and its SLD behaviour reveals how submodular coverage objectives
differ from split-gain landscape fidelity.

### `killamsetty2021gradmatch`
Killamsetty, Sivasubramanian, Mirzasoleiman, Ramakrishnan, De, Iyer.
*GRAD-MATCH.* ICML 2021.

Selects subsets by orthogonal matching pursuit to match gradients of the full
training or validation set, achieving the best accuracy-efficiency trade-off
among contemporary methods. As with CRAIG, GRAD-MATCH is an accuracy-motivated
gradient-based selector; we examine whether gradient matching translates to
split-landscape fidelity under SLD.

### `killamsetty2021glister`
Killamsetty, Sivasubramanian, Ramakrishnan, Iyer. *GLISTER.* AAAI 2021.

Maximises log-likelihood on a held-out validation set via mixed
discrete-continuous bilevel optimisation to select training subsets for
efficient and robust learning. GLISTER is included in our evaluated method pool;
its SLD score illustrates how validation-set-guided selection compares to
landscape-driven approaches.

### `toneva2019forgetting`
Toneva, Sordoni, Combes, Trischler, Bengio, Gordon. *An Empirical Study of
Example Forgetting during Deep Neural Network Learning.* ICLR 2019.

Finds that many training examples are never "forgotten" (never transition from
correct to incorrect classification) and that unforgettable examples can be
pruned with minimal accuracy loss. Forgetting-based pruning motivated
simplification of training sets; we examine whether examples that are "easy"
for neural nets are also low-information for tree split selection.

### `paul2021el2n`
Paul, Ganguli, Dziugaite. *Deep Learning on a Data Diet: Finding Important
Examples Early in Training.* NeurIPS 2021.

Proposes EL2N (Error L2-Norm) and GraNd (Gradient Normed) scores computed
early in training to prune half of CIFAR-10 while slightly improving accuracy.
These gradient-norm-based importance scores assume a differentiable model; we
contrast them with SLD, which is a training-free, tree-specific score that does
not require early training.

### `feldman2011sensitivity`
Feldman, Langberg. *A Unified Framework for Approximating and Clustering Data.*
STOC 2011.

Establishes the theoretical sensitivity-sampling framework: each data point
receives a sensitivity score equal to its worst-case relative contribution to
the loss, and coresets can be constructed by sampling proportional to
sensitivity. This underpins many coreset algorithms and provides the theoretical
scaffold for our leaf-estimate error decomposition.

### `katharopoulos2018importance`
Katharopoulos, Fleuret. *Not All Samples Are Created Equal: Deep Learning with
Importance Sampling.* ICML 2018.

Derives a tractable upper bound to per-sample gradient norms for importance
sampling of mini-batches, reducing gradient variance and achieving up to an
order-of-magnitude improvement in training loss for a fixed wall-clock budget.
Relevant as a principled importance-sampling baseline; our SLD-based importance
differs by being specific to histogram split-gain estimation rather than
gradient variance reduction.

### `loshchilov2016online`
Loshchilov, Hutter. *Online Batch Selection for Faster Training of Neural
Networks.* ICLR 2016 Workshop.

Proposes exponential-rank-based online batch selection, prioritising high-loss
examples during SGD to accelerate convergence. Included as an early training-
based selection baseline; unlike SLD-guided selection, it requires partial
training to compute loss estimates.

---

## 5. Submodularity / Combinatorial Theory

**Positioning.** The k-center method and CRAIG both use submodular objectives.
Our "coverage is not accuracy" finding is partly a corollary of the fact that
maximising coverage (submodular cardinality-constrained set cover) is orthogonal
to maximising split-gain fidelity.

### `nemhauser1978submodular`
Nemhauser, Wolsey, Fisher. *An Analysis of Approximations for Maximizing
Submodular Set Functions—I.* Mathematical Programming, 1978.

Proves the classical greedy (1 − 1/e) approximation guarantee for maximising a
monotone submodular function subject to a cardinality constraint. This result
underpins correctness guarantees for coreset methods based on facility-location
and coverage objectives (CRAIG, k-center). We cite it when proving that
coverage-optimal coreset methods have a worst-case bound that does not translate
to accuracy or SLD fidelity.

---

## 6. Gradient-Boosted Decision Trees (GBDT) Foundations

**Positioning.** GBDTs are the primary model class studied in our paper. The
three entries below form the theoretical foundation of the split-gain landscape
that SLD measures.

### `friedman2001gbm`
Friedman. *Greedy Function Approximation: A Gradient Boosting Machine.* Annals
of Statistics, 2001.

Derives gradient boosting as a stagewise numerical optimisation in function
space, showing how decision trees serve as base learners through second-order
approximations of arbitrary differentiable loss functions. This paper defines
the split-gain criterion (second-order approximation of the loss reduction) that
SLD directly analyses. All modern GBDT systems (XGBoost, LightGBM, CatBoost)
implement the split-gain criterion from this paper.

### `friedman2002stochastic`
Friedman. *Stochastic Gradient Boosting.* Computational Statistics & Data
Analysis, 2002.

Extends gradient boosting with row subsampling at each iteration, showing both
accuracy improvements and computational savings. Stochastic boosting introduces
explicit subsampling into GBDT, making this a conceptual predecessor to GOSS
and MVS. Our analysis compares random subsampling (stochastic boosting) against
more structured condensation methods via SLD.

### `freund1997adaboost`
Freund, Schapire. *A Decision-Theoretic Generalization of On-Line Learning and
an Application to Boosting.* JCSS, 1997.

The original AdaBoost paper, framing boosting as sequentially learning weak
classifiers with adaptive sample reweighting. AdaBoost is the conceptual
ancestor of GBDT; understanding sample reweighting in AdaBoost informs our
analysis of how condensation methods implicitly reweight the split-gain
landscape.

### `biau2021optimization`
Biau, Cadre. *Optimization by Gradient Boosting.* In *Advances in
Contemporary Statistics and Econometrics*, Springer, 2021.

Provides a functional-analysis proof of convergence for gradient boosting as
the number of iterations tends to infinity under strong convexity of the risk
functional, and establishes statistical consistency as sample size grows. We
cite this for the theoretical guarantee that the learned tree structure
concentrates around the population minimiser, underpinning our threshold result
on landscape perturbation tolerance.

### `natekin2013tutorial`
Natekin, Knoll. *Gradient Boosting Machines, a Tutorial.* Frontiers in
Neurorobotics, 2013.

A pedagogically oriented tutorial covering the full gradient boosting pipeline
(tree construction, regularisation, hyperparameter choices) with illustrative
examples. We cite this as an accessible reference for readers unfamiliar with
GBDT mechanics who need background before our SLD formulation.

### `loh2011cart`
Loh. *Classification and Regression Trees.* WIREs Data Mining, 2011.

Reviews CART, C4.5, and GUIDE, covering the theory of node splitting and
impurity criteria from a statistical perspective. Background reference for the
split-scoring framework; our SLD generalises node-split gain histograms to a
full landscape representation comparable across datasets.

### `geurts2006extratrees`
Geurts, Ernst, Wehenkel. *Extremely Randomized Trees.* Machine Learning, 2006.

Proposes extremely randomised trees (ExtraTrees) where both attribute and
split-point selection are randomised, achieving computational efficiency at a
small accuracy cost. Included as one of the non-boosting tree ensemble baselines
in our experiments, providing a contrast case where split-gain sharpness is
deliberately degraded.

### `breiman2001rf`
Breiman. *Random Forests.* Machine Learning, 2001.

Introduces random forests: bootstrap aggregation of trees with random feature
subsets, establishing ensemble diversity as a key regularisation strategy.
Background reference; our paper focuses on GBDT rather than RF, but Random
Forests appear in benchmark comparisons.

---

## 7. Modern GBDT Systems

**Positioning.** We run all condensation experiments using XGBoost, LightGBM,
and CatBoost as the downstream evaluation models, so these are both primary
references and practical tools in our pipeline.

### `chen2016xgboost`
Chen, Guestrin. *XGBoost: A Scalable Tree Boosting System.* KDD 2016.

Presents a scalable, regularised GBDT system with a sparsity-aware algorithm,
weighted quantile sketch for approximate split finding, and cache-efficient
data structures. XGBoost is one of the three GBDT engines on which we evaluate
condensed datasets; its approximate histogram split finder is central to our
SLD formulation.

### `ke2017lightgbm`
Ke, Meng, Finley, Wang, Chen, Ma, Ye, Liu. *LightGBM: A Highly Efficient
Gradient Boosting Decision Tree.* NeurIPS 2017.

Proposes Gradient-based One-Side Sampling (GOSS)—keeping large-gradient
instances and randomly sampling small-gradient ones—and Exclusive Feature
Bundling (EFB) to reduce feature dimensionality, achieving 20x speedup over
XGBoost. GOSS is itself a structured condensation/subsampling heuristic for
GBDTs; we include it as a baseline method and use LightGBM's histogram API
in our SLD computation.

### `prokhorenkova2018catboost`
Prokhorenkova, Gusev, Vorobev, Dorogush, Gulin. *CatBoost: Unbiased Boosting
with Categorical Features.* NeurIPS 2018.

Introduces ordered boosting to eliminate target leakage and a novel algorithm
for categorical feature encoding, producing state-of-the-art accuracy on
heterogeneous tabular datasets. CatBoost is the third GBDT engine in our
evaluation; its ordered boosting means that condensation affects each tree's
training distribution differently, a nuance captured by SLD.

---

## 8. GBDT-Specific Subsampling

**Positioning.** GOSS (inside LightGBM) and MVS are the only existing methods
that do structured data subsampling specifically for GBDT training. They are
the closest prior work to our condensation-for-GBDT topic, but they operate
online (per boosting round) rather than producing a single static condensed set.
SLD lets us evaluate whether their dynamic sampling induces a consistent
landscape shift.

### `ibragimov2019mvs`
Ibragimov, Gusev. *Minimal Variance Sampling in Stochastic Gradient Boosting.*
NeurIPS 2019.

Derives closed-form optimal sampling probabilities that minimise variance of
the split-scoring estimator, leading to the Minimal Variance Sampling (MVS)
technique; MVS achieves better quality than GOSS while sampling fewer examples.
MVS explicitly targets the same split-gain estimation objective as our SLD
score, making it the most theoretically adjacent prior work. We compare MVS
against static condensation methods and show that SLD predicts their relative
ordering.

---

## 9. Tabular Deep Learning vs. Trees

**Positioning.** This body of work motivates why GBDTs remain the dominant
model class for tabular data, justifying our focus on condensation methods
specifically for GBDTs rather than neural-network methods.

### `grinsztajn2022why`
Grinsztajn, Oyallon, Varoquaux. *Why Do Tree-Based Models Still Outperform
Deep Learning on Tabular Data?* NeurIPS D&B, 2022.

Benchmarks standard and novel deep learning methods against tree-based models
on 45 tabular datasets, finding that tree-based models remain SOTA on medium-
sized data even without accounting for their superior speed. This paper directly
motivates our focus on GBDT as the target model; efficient condensation for
GBDTs is practically important given this finding.

### `gorishniy2021revisiting`
Gorishniy, Rubachev, Khrulkov, Babenko. *Revisiting Deep Learning Models for
Tabular Data.* NeurIPS 2021.

Systematically evaluates deep architectures (MLP, ResNet, FT-Transformer) on
tabular benchmarks, establishing the FT-Transformer as a strong DL baseline.
We reference this to position our work: even the best tabular DL models benefit
from data condensation, and their condensation behaviour could in principle be
studied with a landscape analog to SLD.

### `arik2021tabnet`
Arik, Pfister. *TabNet: Attentive Interpretable Tabular Learning.* AAAI 2021.

Proposes TabNet, which uses sequential attention to select features at each
decision step, providing interpretability alongside competitive accuracy on
tabular data. TabNet's attention mechanism is architecturally analogous to
feature selection in trees; we cite it as a DL comparison point in the tabular
accuracy results section.

### `borisov2022survey`
Borisov, Leemann, Sebler, Haug, Pawelczyk, Kasneci. *Deep Neural Networks and
Tabular Data: A Survey.* IEEE TNNLS, 2024 (online 2022).

Surveys data transformations, specialised architectures, and regularisation
for tabular DL, including generative methods. Provides the broad context of
the tabular DL field and its limitations, reinforcing why tree-based methods
(and thus GBDT condensation) remain practically important.

### `kotelnikov2023tabddpm`
Kotelnikov, Baranchuk, Rubachev, Babenko. *TabDDPM: Modelling Tabular Data
with Diffusion Models.* ICML 2023.

Applies multinomial and Gaussian diffusion to categorical and numerical tabular
features respectively, achieving state-of-the-art synthetic tabular data
generation. Generative approaches like TabDDPM can in principle be used for
condensation; SLD could evaluate whether diffusion-generated data preserves the
split landscape.

---

## 10. Data-Centric AI and Data Valuation

**Positioning.** Data Shapley and influence functions are the principled
data-valuation methods closest in spirit to SLD. Our SLD is simpler (no
retraining required) and GBDT-specific; the literature here provides the
formal valuation framework our diagnostic complements.

### `ghorbani2019shapley`
Ghorbani, Zou. *Data Shapley: Equitable Valuation of Data for Machine Learning.*
ICML 2019.

Defines Data Shapley values satisfying uniqueness axioms of equitable
data contribution measurement, with Monte Carlo and gradient-based estimators.
Data Shapley requires retraining on many subsets; SLD achieves a related goal
(ranking data contribution for GBDT) without any retraining, making it orders
of magnitude cheaper.

### `koh2017influence`
Koh, Liang. *Understanding Black-box Predictions via Influence Functions.*
ICML 2017.

Adapts influence functions from robust statistics to trace predictions back to
training data, requiring only gradient and Hessian-vector products. Influence
functions are differentiable-model tools; our SLD is the analogous training-
free diagnostic for non-differentiable GBDT models where influence functions
cannot be directly applied.

---

## 11. Hyperparameter Optimisation

**Positioning.** One of the downstream uses we evaluate for condensed sets is
whether they preserve HPO rankings (i.e., whether a cheaply condensed set can
replace the full dataset for HPO). Optuna and Hyperband are the HPO tools used
in our experiments.

### `akiba2019optuna`
Akiba, Sano, Yanase, Ohta, Koyama. *Optuna: A Next-generation Hyperparameter
Optimization Framework.* KDD 2019.

Introduces a define-by-run API, Tree-structured Parzen Estimator search, and
efficient pruning strategies for hyperparameter optimisation. Optuna is the HPO
framework used in our experiments to evaluate whether condensed datasets
preserve hyperparameter sensitivity rankings.

### `li2018hyperband`
Li, Jamieson, DeSalvo, Rostamizadeh, Talwalkar. *Hyperband: A Novel Bandit-
Based Approach to Hyperparameter Optimization.* JMLR, 2018.

Formulates HPO as a pure-exploration bandit problem allocating a fixed resource
budget to randomly sampled configurations; Hyperband provides over an order-of-
magnitude speedup over Bayesian optimisation baselines. We use Hyperband-style
successive halving as an alternative HPO strategy in the condensed-set HPO
transfer evaluation.

---

## 12. Tabular ML Benchmarks and Open Datasets

**Positioning.** We evaluate across many real tabular datasets drawn primarily
from OpenML and UCI. These benchmark references establish the data provenance
and experimental protocol.

### `vanschoren2013openml`
Vanschoren, van Rijn, Bischl, Torgo. *OpenML: Networked Science in Machine
Learning.* ACM SIGKDD Explorations, 2013.

Introduces OpenML as a networked repository for machine learning data,
experimental results, and workflows, enabling reproducibility and meta-learning.
The majority of tabular datasets in our benchmark are sourced from OpenML;
this citation establishes data provenance and the reproducibility claim of our
experimental setup.

### `dua2017uci`
Dua, Graff. *UCI Machine Learning Repository.* 2017.

The long-standing repository of tabular datasets from diverse domains, used as
a standard benchmark in tabular ML for decades. We use several UCI datasets to
complement the OpenML collection, ensuring coverage of domains not well-
represented there.

### `klein2019tabular`
Klein, Hutter. *Tabular Benchmarks for Joint Architecture and Hyperparameter
Optimization.* arXiv 1905.04970, 2019.

Provides cheap-to-evaluate tabular benchmark functions representing neural
network HPO on four regression datasets. Referenced in the HPO transfer
evaluation section; condensed datasets trained against these benchmarks test
whether rank-preservation generalises across HPO landscapes.

---

## 13. Submodularity (Supporting Theory)

### `nemhauser1978submodular`
*(Described in Section 5 above.)*

---

## Notable Gaps

- **Gradient-boosted tree convergence theory:** No widely-cited, single
  canonical paper provides finite-sample guarantees on split-gain histogram
  convergence as a function of sample size. We cite Biau & Cadre 2021 for
  asymptotic results.
- **Histogram binning theory for split finding:** The analysis of bias/variance
  of histogram-based split-gain estimation is spread across the LightGBM paper
  (ke2017lightgbm) and the MVS paper (ibragimov2019mvs); no standalone theory
  paper is available.
- **Tabular coreset theory:** No paper provides provable guarantees for coresets
  specifically under the GBDT split-gain objective; our SLD fills this gap
  empirically.
