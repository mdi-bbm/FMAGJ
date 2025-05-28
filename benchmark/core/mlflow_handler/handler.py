from typing import Any

import mlflow

from benchmark.core.mlflow_handler.base import BaseMLFlowHandler
from benchmark.core.mlflow_handler.version import MLFlowVersion


class MLFlowHandler(BaseMLFlowHandler):
    def __init__(self, dir_mlflow: str | None, version: MLFlowVersion):
        self.dir_mlflow = dir_mlflow
        self.version = version
        self._init_mlflow()

    def _init_mlflow(self) -> None:
        if self.dir_mlflow:
            mlflow.set_tracking_uri(self.dir_mlflow)
        mlflow.set_experiment(str(self.version))

    def log_metric(
        self,
        metric_name: str,
        metric_value: float,
        sample_idx_or_epoch: int | None = None
    ) -> None:
        mlflow.log_metric(metric_name, metric_value, step=sample_idx_or_epoch)

    def log_param(self, param_name: str, param_value: Any) -> None:
        mlflow.log_param(param_name, param_value)

    def log_artifact(self, artifact_path: str) -> None:
        mlflow.log_artifact(artifact_path)
