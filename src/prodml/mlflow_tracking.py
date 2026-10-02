from pathlib import Path

import mlflow

from .config import settings


def configure_mlflow() -> None:
    """Configure the MLflow tracking server."""

    mlflow.set_tracking_uri(
        settings.mlflow_tracking_uri
    )

    mlflow.set_experiment(
        settings.mlflow_experiment_name
    )


def log_training_run(
    model,
    metrics: dict[str, float],
    params: dict[str, object],
    model_path: Path,
    dvc_data_hash: str,
) -> None:
    """Log a training run to MLflow."""
    
    with mlflow.start_run():

        mlflow.log_params(params)

        mlflow.log_metrics(metrics)

        mlflow.set_tag(
            "model_version",
            settings.model_version,
        )

        mlflow.set_tag(
            "dvc_data_hash",
            dvc_data_hash,
        )

        mlflow.log_artifact(
            str(model_path)
        )

def get_dvc_data_hash(dvc_file: Path) -> str:
    """Extract the MD5 hash from a DVC metadata file."""

    for line in dvc_file.read_text(
        encoding="utf-8"
    ).splitlines():

        line = line.strip()

        if line.startswith("- md5:"):
            return line.split(":", 1)[1].strip()

    raise ValueError(
        f"DVC hash not found in {dvc_file}"
    )