# Model promotion decision log

## Decision 1: challenger credit_model_2025_01_01 (XGBoost), recorded 2026-10-30

- Decision: NOT PROMOTED on this evidence. The champion stays credit_model_2024_05_01 (XGBoost). The 2025-01-01 model stays registered as challenger and monitoring continues.
- Basis: the promotion gate was not crossed at any run in the 30-run backfill.
- Scope: nine cohorts (2024-05 to 2025-01) were scored by the champion; no scoring run after 2025-06 is in the submission.

### Rule
Promote when the challenger's AUC lift over the champion is at least +0.010 on out-of-time cohorts, the 95% confidence interval is above 0, and a human signs off. The pipeline raises an advisory flag only (`challenger_promotion_gate`).

### Design of the comparison
- Both models are XGBoost. The champion was selected at 2024-05-01 as the candidate with the highest pooled walk-forward AUC. The challenger is the champion's algorithm retrained on newer data (originations to 2024-08) at the scheduled retrain on 2025-01-01, so the lift measures the effect of retraining. All three candidate algorithms are evaluated again at every retrain; if another algorithm leads, the run flags a separate model-change review.
- The challenger is scored only on cohorts originated after its training window (2024-09 onward), which neither model has seen.

### Evidence
- Cumulative lift (challenger minus champion): +0.0041 (493 loans), +0.0085, +0.0047, +0.0032, +0.0055 (2,478 loans; 95% range -0.0004 to +0.011 in the monitoring table). The gate was not crossed.
- Candidate selection at the retrain date (pooled walk-forward AUC): XGBoost 0.8084, random forest 0.7951, logistic regression 0.7599; XGBoost leads random forest by +0.0134 (95% CI +0.0081 to +0.0187).
- Candidate selection at the first training date: XGBoost 0.8036, random forest 0.797, logistic regression 0.765; XGBoost leads random forest by +0.0066 (95% CI -0.0005 to +0.0138).
- Noise: the lift estimate has a standard deviation of 0.0029 at the last look; the gate limit of +0.010 is about three of these (governance SOP, section 9.2).

### Limits
- Selection folds are unpurged and the selection AUC is pooled across folds.
- Lift results are cumulative over overlapping cohorts at five looks, so they are not independent confirmations.
- The sealed block and later cohorts were reused for model comparisons. No untouched holdout exists.

### Earlier run
An earlier run of this pipeline (random forest champion, XGBoost challenger) crossed the gate with a lift of +0.0138 at 2025-06 and recorded an approval. That lift included the change of algorithm. The champion rule was then changed to highest pooled walk-forward AUC, the pipeline was rerun, and that approval no longer applies. The first-run outputs are kept in the backup folder made before the rerun.
