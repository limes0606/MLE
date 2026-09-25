CS611 Assignment 1 -- Data Processing Pipelines
=================================================

Repo: https://github.com/limes0606/MLE/tree/main/assignment1

How to run:
1. docker compose up --build
2. Open a terminal in JupyterLab (http://localhost:8888/lab)
3. cd /home/jovyan/work
4. python main.py

This builds a datamart/ folder with:
  datamart/bronze/{clickstream,loan_daily,financials,attributes}/...
  datamart/silver/{clickstream,loan_daily,financials,attributes}/...
  datamart/gold/{label_store,feature_store}/...

Label definition:
  bad=1 if overdue_amt > 0 at installment_num == 6 (fixed mob=6 checkpoint),
  else bad=0. All loans have tenure=10 months and are fully observed, so
  every loan reaches mob=6 with no censoring. This is a point-in-time
  definition, not a cumulative one: a loan overdue at mob=6 but cured by
  mob=7 still counts as bad, and vice versa -- a deliberate simplification,
  stated explicitly here as an assumption.

Feature store:
  Built from attributes + financials, joined on (Customer_ID, snapshot_date)
  -- both are captured exactly at loan origination for every customer, so
  this join is always aligned with no leakage risk. Clickstream is
  left-joined after filtering to rows where its own snapshot_date matches
  the customer's loan_start_date; only ~72% of customers (8,974 / 12,500)
  have clickstream coverage, so the remaining customers keep null
  clickstream features rather than being dropped from the feature store.

Sanity check:
  A simple logistic regression fit on the joined feature+label store
  (train/test split before any imputation/encoding, to avoid contaminating
  the test set) gives a test AUC of ~0.50. Read narrowly: this confirms the
  join/keys work end-to-end with no leakage red flag (which would look like
  AUC near 1.0), not that the pipeline "performs well" -- actual predictive
  performance is out of scope for this assignment.