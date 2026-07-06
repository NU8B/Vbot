"""
Optional MLflow tracking layer for the Vbot evaluation scripts.

The schema-versioned JSON artifacts remain the source of truth (they are
what the gates consume and what gets committed as baselines). MLflow is an
index and UI over them: each eval run logs its parameters and headline
metrics and attaches the artifact file, into a local file store (mlruns/).

Design rules:
  - Fully optional: if mlflow is not installed, log_eval_run() is a silent
    no-op and every eval script behaves exactly as before. mlflow is NOT
    in requirements-ci.txt on purpose — CI stays light.
  - Never break an eval because tracking failed: all exceptions are
    swallowed with a warning line.

Browse runs with:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MLflow 3.x deprecated the plain ./mlruns file store; a local SQLite file
# is the supported lightweight backend. Run artifacts land in ./mlartifacts.
TRACKING_URI = "sqlite:///" + os.path.join(PROJECT_ROOT, "mlflow.db").replace("\\", "/")


def log_eval_run(experiment, run_name, params=None, metrics=None, artifact=None, tags=None):
    """Log one eval run to the local MLflow store. Returns True if logged.

    metrics must be flat {name: number}; non-finite/None values are
    skipped. artifact is a file path attached to the run.
    """
    try:
        import mlflow
    except ImportError:
        return False

    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name):
            if tags:
                mlflow.set_tags(tags)
            if params:
                mlflow.log_params({key: str(value) for key, value in params.items()})
            if metrics:
                clean = {
                    key: float(value)
                    for key, value in metrics.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
                if clean:
                    mlflow.log_metrics(clean)
            if artifact and os.path.isfile(artifact):
                mlflow.log_artifact(artifact)
        return True
    except Exception as exc:  # noqa: BLE001 - tracking must never break evals
        print(f"[WARN] MLflow tracking skipped: {exc}")
        return False
