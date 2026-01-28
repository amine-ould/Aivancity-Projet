"""DAG Airflow minimal pour orchestrer le pipeline."""

from datetime import datetime

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
except Exception:  # Airflow peut ne pas être installé localement
    DAG = None
    BashOperator = None

if DAG and BashOperator:
    with DAG(
        dag_id="predictive_maintenance_pipeline",
        start_date=datetime(2024, 1, 1),
        schedule_interval=None,
        catchup=False,
        tags=["ml", "predictive-maintenance"],
    ) as dag:
        run_training = BashOperator(
            task_id="run_training",
            bash_command="python run_training.py",
        )

        run_training
