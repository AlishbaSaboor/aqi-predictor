"""
logger.py — Shared logging utility for the AQI pipeline.
Every script calls log_run() after finishing.
All logs stored in MongoDB pipeline_logs collection.
Training metrics stored in training_history collection.
"""
from pymongo import MongoClient
from datetime import datetime, timezone
import os
from dotenv import load_dotenv

load_dotenv()


def _connect(collection_name):
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("MONGO_DB")]
    return client, db[collection_name]


def log_run(script, status, rows_inserted=0, rows_skipped=0,
            date_range=None, extra_info=None, error_message=None):
    """
    Write one log entry to pipeline_logs collection.

    script        : 'fetch_data' | 'feature_engineering' | 'training' | 'backfill' | 'shap'
    status        : 'success' | 'failed'
    rows_inserted : new rows added to database
    rows_skipped  : duplicate rows skipped
    date_range    : e.g. '2026-05-30 to 2026-06-02'
    extra_info    : any extra dict
    error_message : error text if failed
    """
    # Detect if running on GitHub Actions
    run_env = "github_actions" if os.getenv("GITHUB_ACTIONS") else "local"
    run_id  = os.getenv("GITHUB_RUN_ID", "local_run")

    entry = {
        "timestamp":       datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "script":          script,
        "status":          status,
        "rows_inserted":   rows_inserted,
        "rows_skipped":    rows_skipped,
        "date_range":      date_range or "N/A",
        "run_environment": run_env,
        "github_run_id":   run_id,
        "error_message":   error_message,
        "extra":           extra_info or {},
    }
    try:
        client, col = _connect("pipeline_logs")
        col.insert_one(entry)
        client.close()
        print(f"📝 LOG: {script} | {status} | inserted={rows_inserted} | env={run_env}")
    except Exception as e:
        print(f"⚠️ Logging failed (non-critical): {e}")


def log_training(best_model, all_metrics, train_rows, test_rows):
    """
    Write one training record to training_history collection.
    Called at the end of every train.py run.
    """
    run_env = "github_actions" if os.getenv("GITHUB_ACTIONS") else "local"
    entry = {
        "timestamp":       datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "best_model":      best_model,
        "all_metrics":     all_metrics,
        "train_rows":      train_rows,
        "test_rows":       test_rows,
        "run_environment": run_env,
        "github_run_id":   os.getenv("GITHUB_RUN_ID", "local_run"),
    }
    try:
        client, col = _connect("training_history")
        col.insert_one(entry)
        client.close()
        print(f"📝 TRAINING LOG: best={best_model} RMSE={all_metrics[best_model]['rmse']}")
    except Exception as e:
        print(f"⚠️ Training log failed (non-critical): {e}")


def get_pipeline_logs(limit=100):
    """Fetch recent pipeline logs. Used by dashboard."""
    try:
        client, col = _connect("pipeline_logs")
        logs = list(col.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
        client.close()
        return logs
    except Exception as e:
        print(f"⚠️ Could not fetch logs: {e}")
        return []


def get_training_history(limit=30):
    """Fetch recent training history. Used by dashboard."""
    try:
        client, col = _connect("training_history")
        hist = list(col.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
        client.close()
        return hist
    except Exception as e:
        print(f"⚠️ Could not fetch training history: {e}")
        return []