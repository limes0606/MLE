"""
Monthly ML pipeline DAG.

For each logical date D (= snapshot date, {{ ds }}):
  data   : bronze+silver (4 tables) -> gold feature store (loans originated in D)
                                     -> gold label store  (loans originated D-5 months; MOB=5 known on D)
  model  : train + select (champion on TRAIN_DATE; challenger on CHALLENGER_DATE, never auto-promoted)
  serve  : inference -> gold predictions (live window 2024-05..2025-01)
  monitor: stability (cohort D) + performance (cohort D-5, lagged) -> gold monitoring table;
           on the last run also charts + summary
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import ShortCircuitOperator

TRAIN_DATE = "2024-05-01"  # first month all 2023 originations have a mature MOB=5 label (champion)
CHALLENGER_DATE = "2025-01-01"  # scheduled retrain: champion's algorithm refitted on originations up to 2024-08; shadow-compared, never auto-promoted
INFER_START, INFER_END = "2024-05-01", "2025-01-01"  # live scoring window (9 monthly cohorts)
MONITOR_START, MONITOR_END = "2024-05-01", "2025-06-01"  # stability from first scored month; performance until last cohort matures
REPORT_DATE = "2025-06-01"  # last run: build charts + summary
S = "cd /opt/airflow/scripts && python3 "

default_args = {"owner": "airflow", "depends_on_past": False,
                "retries": 1, "retry_delay": timedelta(minutes=2)}


def script(task_id, filename):
    return BashOperator(task_id=task_id, bash_command=S + f'{filename} --snapshotdate "{{{{ ds }}}}"')



with DAG(
    "ml_pipeline",
    default_args=default_args,
    description="Monthly datamart -> train -> inference -> monitoring",
    schedule_interval="0 0 1 * *",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2025, 6, 1),
    catchup=True,
    max_active_runs=1,
) as dag:

    start = EmptyOperator(task_id="start")

    bronze_silver = script("bronze_silver", "bronze_silver.py")
    gold_features = script("gold_feature_store", "gold_feature_store.py")
    gold_labels = script("gold_label_store", "gold_label_store.py")
    data_done = EmptyOperator(task_id="data_done")

    # ignore_downstream_trigger_rules=False: skip only `train`, not the shared `end` task
    is_train_date = ShortCircuitOperator(task_id="is_train_date",
                                         python_callable=lambda ds, **_: ds == TRAIN_DATE,
                                         ignore_downstream_trigger_rules=False)
    train = script("train_and_select", "train_and_select.py")
    # challenger branch is separate: a failure here must never block champion scoring or monitoring
    is_challenger_date = ShortCircuitOperator(task_id="is_challenger_date",
                                              python_callable=lambda ds, **_: ds == CHALLENGER_DATE,
                                              ignore_downstream_trigger_rules=False)
    train_challenger = script("train_challenger", "train_and_select.py")

    # score only inside the live window; train must have run first (model bank), so
    # 2024-05-01 inference waits for train_and_select (see dependency below)
    in_infer_window = ShortCircuitOperator(task_id="in_inference_window",
                                           python_callable=lambda ds, **_: INFER_START <= ds <= INFER_END,
                                           ignore_downstream_trigger_rules=False)
    inference = BashOperator(task_id="inference", trigger_rule="none_failed_min_one_success",
                             bash_command=S + 'inference.py --snapshotdate "{{ ds }}"')
    in_monitor_window = ShortCircuitOperator(task_id="in_monitoring_window",
                                             python_callable=lambda ds, **_: MONITOR_START <= ds <= MONITOR_END,
                                             ignore_downstream_trigger_rules=False)
    # inference is skipped after 2025-01 (no new scoring) but performance monitoring of the
    # last cohorts continues, so monitoring runs when at least one upstream succeeded
    monitoring = BashOperator(task_id="monitoring", trigger_rule="none_failed_min_one_success",
                              bash_command=S + 'monitoring.py --snapshotdate "{{ ds }}"')
    is_report_date = ShortCircuitOperator(task_id="is_report_date",
                                          python_callable=lambda ds, **_: ds == REPORT_DATE,
                                          ignore_downstream_trigger_rules=False)
    report = script("monitoring_report", "monitoring_report.py")
    end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    start >> bronze_silver >> [gold_features, gold_labels] >> data_done
    data_done >> is_train_date >> train >> end
    data_done >> is_challenger_date >> train_challenger >> end
    data_done >> in_infer_window >> inference >> monitoring
    data_done >> in_monitor_window >> monitoring
    monitoring >> is_report_date >> report >> end
    monitoring >> end
    data_done >> end  # keeps `end` green on runs where every optional branch is skipped
    train >> inference  # on the training run, score only after the model exists
