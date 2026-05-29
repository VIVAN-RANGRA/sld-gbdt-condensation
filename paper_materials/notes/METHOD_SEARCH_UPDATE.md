# Method Search Update

## Search Design

Focused 8-dataset method search over budgets 10/25/50, seeds 0/1, and five downstream learners. Each complete method has 240 paired evaluation cells.

Candidate families tested:

- trajectory/probe sketching: `trajectory_refined`
- split-margin matching: `split_margin_refined`
- threshold witnesses: `threshold_witness`
- region-adaptive leaf allocation: `region_adaptive_refined`
- region plus trajectory sketching: `region_trajectory_refined`

## Result

The search did **not** find a better main method than `gain_path_refined`.

| Method | Mean AUROC | Avg. Rank | Cells |
| --- | --- | --- | --- |
| Gain-path refined | 0.7797 | 3.7583 | 240 |
| Trajectory refined | 0.7773 | 3.8208 | 240 |
| Gain sketch | 0.7734 | 4.0938 | 240 |
| Split-margin refined | 0.7715 | 4.4313 | 240 |
| Herding | 0.7669 | 5.0563 | 240 |
| Region trajectory refined | 0.7659 | 5.2146 | 240 |
| Region adaptive refined | 0.7652 | 5.2333 | 240 |
| Random | 0.7475 | 6.0208 | 240 |
| Threshold witness | 0.6315 | 7.3708 | 240 |

## Budget Breakdown

| Method | budget_10 | budget_25 | budget_50 |
| --- | --- | --- | --- |
| Gain sketch | 0.6642 | 0.8144 | 0.8417 |
| Gain-path refined | 0.6667 | 0.8245 | 0.8479 |
| Herding | 0.6717 | 0.8059 | 0.8232 |
| Random | 0.6348 | 0.7808 | 0.8268 |
| Region adaptive refined | 0.6695 | 0.8010 | 0.8250 |
| Region trajectory refined | 0.6600 | 0.8063 | 0.8313 |
| Split-margin refined | 0.6679 | 0.8133 | 0.8332 |
| Threshold witness | 0.5348 | 0.6473 | 0.7124 |
| Trajectory refined | 0.6668 | 0.8154 | 0.8495 |

## Candidate Deltas Against Current Main Method

Negative means the candidate is worse than `gain_path_refined`.

| Method | Diff vs Gain-path refined | wins | ties | losses | Wilcoxon p |
| --- | --- | --- | --- | --- | --- |
| Trajectory refined | -0.0025 | 76 | 84 | 80 | 0.2465 |
| Split-margin refined | -0.0083 | 78 | 38 | 124 | 0.0001 |
| Threshold witness | -0.1482 | 25 | 34 | 181 | 0.0000 |
| Region adaptive refined | -0.0145 | 59 | 32 | 149 | 0.0000 |
| Region trajectory refined | -0.0139 | 56 | 32 | 152 | 0.0000 |

## Interpretation

- `trajectory_refined` is the only credible challenger, but it still trails `gain_path_refined` once evaluated across budgets and seeds.
- `split_margin_refined` helps over random/herding but loses to the current main method.
- Region-adaptive allocation hurt, likely because hard quotas fragment already tiny per-class budgets.
- Threshold witnesses failed badly; selecting near thresholds alone does not preserve enough class/feature structure.
- Current best remains `gain_path_refined`.

CSV outputs:

- `tables/focused_method_search_performance.csv`
- `tables/focused_method_search_budget_means.csv`
- `tables/focused_method_search_pairwise.csv`
