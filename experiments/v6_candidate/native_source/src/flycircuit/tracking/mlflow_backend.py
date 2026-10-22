"""Local MLflow backend: SQLite store + filesystem artifacts, both inside the project (WSL)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

EXPERIMENT = "flycircuit"


def tracking_uri(root: Path) -> str:
    d = root / "runs" / "mlflow"
    d.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{d / 'mlflow.db'}"


def log_manifest(manifest: dict[str, Any], manifest_path: Path, root: Path) -> dict[str, Any]:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri(root))
    art = (root / "runs" / "mlflow" / "artifacts").resolve()
    art.mkdir(parents=True, exist_ok=True)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    exp_id = exp.experiment_id if exp else client.create_experiment(EXPERIMENT, art.as_uri())
    with mlflow.start_run(experiment_id=exp_id, run_name=manifest["run_id"]) as run:
        mlflow.set_tags({
            "git_revision": str(manifest["git_revision"]), "git_dirty": str(manifest["git_dirty"]),
            "exit_status": manifest["exit_status"], "algorithm": manifest["algorithm"],
            "data_release": str(manifest["data_release"]),
        })
        mlflow.log_params({"config_hash": manifest["config_hash"], "seeds": str(manifest["seeds"])[:250]})
        for k, v in manifest.get("metrics", {}).items():
            if isinstance(v, int | float):
                mlflow.log_metric(k, float(v))
        mlflow.log_metric("elapsed_seconds", manifest["elapsed_seconds"])
        mlflow.log_metric("peak_ram_mib", manifest["peak_ram_mib"])
        if manifest.get("peak_cuda_mib") is not None:
            mlflow.log_metric("peak_cuda_mib", manifest["peak_cuda_mib"])
        mlflow.log_artifact(str(manifest_path))
        return {"status": "logged", "mlflow_run_id": run.info.run_id, "tracking_uri": tracking_uri(root)}
