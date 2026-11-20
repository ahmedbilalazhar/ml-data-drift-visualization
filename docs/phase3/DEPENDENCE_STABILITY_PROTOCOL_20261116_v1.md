# Dependence detector stability extension

**Protocol:** `IDMVIZ_DEPENDENCE_STABILITY_20261116_V1`  
**Status:** outcome-informed exploratory follow-up. It extends a previously observed one-intervention result; it is not confirmatory evidence for the full IDM-Viz system.  
**Question:** how does a maximum pairwise Pearson-correlation-change detector behave across selected strengths, affected pairs, and batch sizes, relative to the existing eight-feature plus three-PCA-coordinate Holm-KS alarm?

## Fixed design

- Generate independent eight-dimensional standard-normal training, reference, and monitoring innovations. Historical reference has 2,000 rows; current batch sizes are 250, 500, and 1,000. PCA is fitted to a disjoint 2,000-row training sample and never refitted.
- A dependence intervention transforms only the second coordinate of one selected pair: `X_j = rho*Z_i + sqrt(1-rho^2)*Z_j`. Every univariate marginal remains standard normal. The intervention starts after batch 8, with onset at batch 9. Pairs are `(0,1)`, `(2,5)`, and `(6,7)`; strengths are `rho = 0.30, 0.60, 0.90`. The stationary control uses `rho=0` throughout.
- Development seeds are 931000–931029. For each batch size separately, freeze the higher empirical 95th-percentile score among all development C0 stationary batches as the dependence threshold. This is a per-batch operating point, not a guaranteed stream-level false-alarm rate.
- Evaluation seeds are 931100–931149 and are disjoint from development. Do not change thresholds after evaluating them. For every identical stream/batch, evaluate the existing original-feature plus three frozen-PCA-coordinate KS family with Holm correction at 0.05.
- Use common Gaussian innovations across intervention strengths and pairs within each seed, sample size and batch index to preserve pairing. No outcomes, labels, condition IDs, or seed IDs are detector inputs.

## Outcomes and analysis

For each of the nine strength/pair settings and each batch size report held-out seed count, detection by batch 12, detection delay (conditional on detection), missed seeds, first-alarm feature-pair recovery, and pre-onset alarms. For the stationary control report false alarms per 24 batches for each detector and sample size. Report per-setting paired seed-bootstrap 95% percentile intervals for detection-rate differences; resample the 50 seeds and retain each seed's complete setting grid. Do not treat batches, pairs, or batch sizes as independent replications. Include all settings without selecting a favorable one. Also report the exact development threshold and achieved development alarm burden.

## Scope

The result applies only to Gaussian pairwise-correlation interventions with unchanged univariate marginals. It does not test arbitrary dependence, non-Gaussian or higher-order changes, causal feature responsibility, model harm, or natural-domain performance. The comparator's PCA coordinates may detect some interventions; both methods receive identical current and historical inputs. This protocol and threshold locks are created before generation of the held-out seeds; the follow-up itself is outcome-informed because the family and feature-pair hypothesis follow the earlier C6 result.
