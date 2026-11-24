"""
Milestone 7: Apache Airflow Retraining & Serving Orchestration DAG.

Schedules the end-to-end daily retraining workflow:
  1. Temporal Ingestion & Leakage Verification (`prepare_movielens.py`)
  2. Spark Feature Store Rebuild (`feature_engineering.py`)
  3. Two-Tower Retrieval Model Training (`two_tower_tfrs.py`)
  4. FAISS Index Build & Validation (`faiss_index.py`)
  5. Stage-2 DCN-v2 & Wide & Deep Ranker Training (`ranking_models.py`)
  6. Offline Quality Gate (verifies NDCG@10 & Recall@50 exceed promotion threshold)
  7. Triton Model Repository Hot-Reload & Redis Cache Warm-Up
"""

from __future__ import annotations

from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

DEFAULT_ARGS = {
    "owner": "badal_singh_aiml03",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def validate_offline_quality_gate(**context) -> str:
    """Fail deployment if candidate DCN-v2 ranker regresses below production NDCG@10 threshold."""
    import json
    from pathlib import Path

    report_path = Path("/opt/airflow/project/artifacts_store/benchmark_report.json")
    if not report_path.exists():
        raise FileNotFoundError(f"Missing benchmark report at {report_path}")

    report = json.loads(report_path.read_text())
    dcn_ndcg = report["models"]["stage2_dcn_v2"]["ndcg_at_10"]
    s1_recall50 = report["models"]["stage1_two_tower_faiss"]["recall_at_50"]

    if dcn_ndcg < 0.20 or s1_recall50 < 0.40:
        raise ValueError(
            f"Quality gate failed: DCN-v2 NDCG@10={dcn_ndcg:.4f}, Stage-1 Recall@50={s1_recall50:.4f}"
        )
    return f"PROMOTED: NDCG@10={dcn_ndcg:.4f}, Recall@50={s1_recall50:.4f}"


if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="movielens_two_stage_retraining",
        default_args=DEFAULT_ARGS,
        description="Daily two-stage TFRS + FAISS + DCN-v2 retraining, Triton reload & Redis warm-up",
        schedule_interval="0 2 * * *",
        start_date=datetime(2026, 10, 29),
        catchup=False,
        tags=["aiml03", "recsys", "tfrs", "faiss", "triton"],
    ) as dag:

        t1_temporal_split = BashOperator(
            task_id="temporal_data_split",
            bash_command="cd /opt/airflow/project && python data/prepare_movielens.py --synthetic",
        )

        t2_spark_features = BashOperator(
            task_id="spark_feature_engineering",
            bash_command="cd /opt/airflow/project && python spark/feature_engineering.py",
        )

        t3_train_pipeline = BashOperator(
            task_id="train_two_tower_faiss_and_rankers",
            bash_command="cd /opt/airflow/project && python models/local_engine.py",
        )

        t4_quality_gate = PythonOperator(
            task_id="offline_quality_gate",
            python_callable=validate_offline_quality_gate,
        )

        t5_triton_reload = BashOperator(
            task_id="triton_model_reload_and_redis_flush",
            bash_command=(
                "curl -sf -X POST http://triton:8000/v2/repository/models/dcn_ranker/load || true && "
                "redis-cli -h redis FLUSHDB || true"
            ),
        )

        t1_temporal_split >> t2_spark_features >> t3_train_pipeline >> t4_quality_gate >> t5_triton_reload
