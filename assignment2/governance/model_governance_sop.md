# Model governance and SOP (proposed process)

Scope: the loan-default model in this pipeline. The pipeline detects and records; people decide. The thresholds below are provisional policy limits, to be recalibrated with more history.

## 1. Units and roles
| Unit | What it does | Roles inside it |
|---|---|---|
| Business (loans) | Owns the outcome | Business owner: proposes the cutoff and the margin, loss and recovery assumptions |
| Data and Technology | Builds and runs | Model owner (trains, monitors, proposes changes, may execute an emergency rollback on a predefined trigger); data engineering (pipeline, infrastructure, access, data incidents); data owner (feed quality, lineage, retention of personal data) |
| Risk and Compliance | Validates and advises | Independent validator (reviews challenger comparisons, signs off recalibration and promotion, reviews any rollback within the week); compliance and legal (consulted on fairness, explanations, privacy and disclosures) |
| Credit-risk committee | Decides | Approves production promotions and cutoff changes; cross-functional, chaired by risk |

The validator sits in Risk and Compliance, apart from the model owner in Data and Technology.

## 2. Change types and approval
| Change | Approval |
|---|---|
| Train a challenger | Data and Technology; no production effect; Risk and Compliance reviews the comparison |
| Recalibrate the deployed model | Risk and Compliance signs off; committee informed |
| Promote a challenger | Risk and Compliance signs off, then the committee approves |
| Rollback | Data and Technology acts on a predefined model-impacting trigger; Risk and Compliance reviews within the week |
| Cutoff change | Business proposes; committee approves |

## 3. Thresholds (provisional policy limits)
- Rolling three-cohort AUC below baseline: amber 0.03, red 0.05. The limits were first set as round-number conventions and then checked against measured noise (section 9): amber is about 2.5 SDs of the rolling AUC, red about 4.
- Score and feature PSI: amber 0.10, red 0.25. These are common heuristics, not a standard.
- Calibration gap: an alert requires both a gap beyond 3 points (amber) or 5 points (red) and a 95% CI that excludes zero.
- Clickstream and label coverage: amber below 99%, red below 95%, raised immediately.
- Two consecutive runs are required for PSI, AUC and calibration, because a month of about 500 loans is noisy. This is a persistence rule, not independent confirmation, since the rolling windows overlap.

## 4. Trigger to action
| # | Trigger | Owner | Action |
|---|---|---|---|
| 1 | Coverage alert | Data and Technology (data owner notified) | Find the cause. For clickstream: why no rows exist for newly originated customers (unresolved). Not a retrain trigger. |
| 2 | PSI above 0.10 | Data and Technology | Use the feature heatmap to separate a real population change from a data fault. |
| 3 | Score PSI above 0.25 | Data and Technology, Risk and Compliance | Confirm it is real, then train a challenger on newer data and compare on unseen loans. |
| 4 | Rolling AUC amber (0.03 below baseline, two consecutive runs) | Data and Technology, Risk and Compliance | Train a challenger on newer data and compare on unseen loans. Red (0.05 below): the same, escalated to Risk and Compliance. |
| 5 | Calibration alert | Data and Technology, Risk and Compliance | Recalibrate (Risk and Compliance signs off) or revisit the cutoff. Not a retrain by itself. |
| 6 | Challenger lift of at least +0.010 with CI above zero | Data and Technology proposes | The pipeline raises an advisory "human review required" flag; see sections 5 and 9. |
| 7 | Red model metric within three months of a swap | Data and Technology | Roll back to the previous champion (kept in the model bank); Risk and Compliance reviews within the week. A coverage alert alone never triggers rollback. |
| 8 | Missing partition, schema or row-count anomaly, or unknown model version | Data and Technology | Hold: stop the run and release no scores until it is fixed. |
| 9 | Every 6 months (time trigger); quarterly review of all metrics | Data and Technology, Risk and Compliance | Train a challenger every 6 months and judge it once five cohorts have matured (section 9.3). At each quarterly review, check all metrics and this table. A promotion happens only if the gate in section 5 is met. |
| 10 | Annual cutoff review; any change in margin, loss or recovery assumptions | Business proposes; committee approves | Review whether the cutoff needs to change. The break-even analysis is scenario evidence, not proof of an optimal cutoff. |

## 5. Promotion preconditions
1. The pipeline flags the cumulative lift as at least +0.010 with the CI above zero. The flag is advisory and its CI is exploratory: the cohorts overlap and the intervals are loan-level. The +0.010 limit is about three standard deviations of the measured noise (section 9.2).
2. The decision is taken at the scheduled quarterly review on the full cumulative window, with Risk and Compliance reviewing at least three cohorts. This reduces month-to-month churn but is still repeated looks at overlapping data, not a clean confirmation test.
3. Manual checks, not produced by the monitoring output today. On the pooled cumulative out-of-time cohorts since the challenger's training window, and using each model's own calibrator and own training reference: (a) mean default probability minus bad rate within +/-3 points, and (b) score PSI below 0.10, for the challenger and for the champion on the same loans. The evidence is attached to the decision-log entry.
4. A segment and fairness review led by Risk and Compliance (compliance and legal consulted). Not performed in this submission.
5. The previous champion is retained. The rollback procedure (pointer swap, re-scoring check) must be tested before any future release; it has not been tested in this submission.
6. The decision is recorded in the log.

## 6. Deployment
- Demonstrated: monthly batch scoring and monitoring in Airflow, with model training on two scheduled dates (2024-05-01 and 2025-01-01), not monthly.
- Possible option, not a default recommendation: real-time scoring at application, only after application-time feature lineage, online and offline feature parity, latency targets and per-decision reasons are verified. Monthly batch would remain for monitoring and retraining.

## 7. Current pipeline behaviour and disclosed gaps
- Current behaviour: inference stops if its feature partition or the champion's calibrator is missing; training refuses a shorter window; monitoring skips metrics whose inputs are absent. The standalone integrity checks (scripts/leakage_checks.py) are not an Airflow gate. Making them one is a next step.
- Gaps: no per-decision explanations (importance is global); fairness not assessed; Name and SSN are dropped before the feature table but remain in the raw, bronze and silver layers, so retention and access controls would be needed; the label is a delinquency proxy for default; the alert notification channel is not implemented.

## 8. Worked example from the real data
Only the clickstream coverage alert fired. The champion is XGBoost, chosen at 2024-05-01 by the highest pooled walk-forward AUC. The 2025-01-01 scheduled retrain (the same algorithm on newer data) did not cross the promotion gate: the cumulative lift was +0.0055 (95% range about 0.000 to +0.011) at the last run, against the +0.010 limit. No promotion advisory exists, and no scoring run after 2025-06 is in the submission.

## 9. How the thresholds were set (evidence and limits)
Scripts: `scripts/monitoring_threshold_analysis.py`, `scripts/promotion_gate_analysis.py`. Outputs: `datamart/model_diagnostics/monitoring_threshold_*` and `promotion_gate_*`. Both use the champion and challenger from the final run on loans neither model trained on.

### 9.1 AUC drop limits (retrain trigger)
- Origin: amber 0.03 and red 0.05 were set as round-number conventions in the design. They were evaluated afterwards, not derived.
- Measured noise (13 cohorts, about 505 loans each): monthly cohort AUC SD 0.0238; rolling three-cohort AUC SD 0.0111 (actual series) and 0.0121 (bootstrap). The baseline AUC 0.8036 has a standard error of about 0.0075.
- Principle: a retrain only creates a challenger, so a false alarm is cheap and the trigger can be sensitive. A promotion is costly, so its gate is strict (9.2).
- False alarms with no decay (share of simulated 22-run series with at least one alarm, two-run rule): limit 0.030: 4% (baseline exact) and 8% (baseline uncertainty included); 0.040: 0.1% and 1.2%; 0.050: 0% and 0.1%.
- Detection of a real decline (ranking quality degraded from a set month; median months from onset include the 5-month label lag): amber 0.03 catches a 0.03 drop 94% of the time in 9 months and a 0.02 drop 66% in 12 months; red 0.05 catches a 0.05 drop 96% in 9 months, a 0.03 drop 23% in 13 months.
- The label lag sets the floor: even a 0.10 drop is first seen about 6 months after the first affected loans.
- Conclusion: amber 0.03 is the retrain trigger. It has a 4% to 8% false-alarm rate over 22 runs and catches a 0.03 decline 94% of the time. Red 0.05 is kept as the escalation level.
- Limits: cohorts drawn independently from pooled loans (no drift or autocorrelation); step decay of ranking quality only; 13 cohorts give a rough noise estimate; for limits of 0.02 or below the detection shares include false alarms.

### 9.2 Promotion gate (lift of at least +0.010 with the 95% CI above zero)
- Evaluation loans: 2,478 originations from 2024-09 to 2025-01, after the challenger's training window.
- Noise in the lift (1 SD): sampling 0.0049 at the first look (493 loans) and 0.0029 at the last (2,478 loans); retraining on a different draw of the same-size data about 0.004; retraining with only a different random seed 0.0013; a single cohort's lift on its own ranges from -0.003 to +0.013.
- Gate behaviour: with zero true benefit it fires about 2% of the time at the first look (normal approximation) and about 0.1% at the last (swap test: 0.1%); across the five looks the combined rate is roughly 5% or less (estimate, not simulated). With a true lift of +0.005 it fires 4% of the time, +0.010: 50%, +0.015: 96%. A true lift of about 0.0124 gives 80% power.
- The 50% at +0.010 comes from the point-estimate condition (lift at least 0.010), not from the width of the interval.
- Rough value: +0.01 AUC is about 9,000 to 40,000 per 1,000 applicants across the three margin scenarios (best cutoff per model, loss per bad loan 7,544, no recovery). The observed lift (+0.0055) is below the +0.010 limit. The cost of changing a model is not in the data and is an assumption for the business to set.
- Conclusion: +0.010 is about three noise SDs. It rarely prompts a review by chance and misses a true gain of 0.005 to 0.010. It is a trigger for human review, not a deployment order.
- Limits: five looks at overlapping cohorts; retrain-resample noise comes from bootstrap resampling of the 2023 loans, which lowers the average lift, so its spread is used and its mean is not; the value estimate is rough.

### 9.3 Retraining triggers
- Performance: rolling AUC amber (9.1), score PSI above 0.25, or a red metric, as in section 4.
- Time: a challenger is trained every 6 months. Reasons:
  - A challenger needs five matured cohorts (about 2,500 loans) to be judged: its lift estimate has a standard error of 0.0049 at one cohort, 0.0044 at two and 0.0029 at five.
  - Labels take 5 months, so the fifth cohort matures 5 months after the training date (trained 2025-01-01, last cohort matured 2025-06). One month for validation and sign-off makes 6 months.
  - A shorter cadence (for example quarterly) would start a new challenger before the last one can be judged; at two cohorts the +0.010 gate is only about 2 standard errors away.
  - A longer cadence would leave slow decay unseen: the performance trigger takes about 9 months to confirm a 0.03 drop and 12 months for a 0.02 drop (9.1), so 6 months sits inside that delay.
  - The 6-month figure is a judgement from these facts. This data set is stable, so it shows no need for a refresh. Revisit with more history.
- Not evaluated this way: PSI limits (common heuristics), calibration limits (3 and 5 points with the CI rule) and coverage limits.
