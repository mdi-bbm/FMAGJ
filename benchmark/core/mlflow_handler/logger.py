import logging
from typing import Any

from benchmark.core.mlflow_handler.base import BaseMLFlowHandler


class MLFlowLogger:
    def __init__(self, handler: BaseMLFlowHandler, name: str = "mlflow_logger"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.handler = handler
        self._setup_logger()

    def _setup_logger(self) -> None:
        mlflow_handler = logging.StreamHandler()
        mlflow_handler.setLevel(logging.INFO)
        self.logger.addHandler(mlflow_handler)

    def log_metric(
        self,
        metric_name: str,
        metric_value: float,
        sample_idx_or_epoch: int | None = None
    ) -> None:
        logging_str = 'Metric logged: '
        if sample_idx_or_epoch is not None:
            logging_str += f'epoch = {sample_idx_or_epoch}, '
        logging_str += f'{metric_name} = {metric_value}'
        self.logger.debug(f"Metric logged: {logging_str}")
        self.handler.log_metric(metric_name, metric_value, sample_idx_or_epoch=sample_idx_or_epoch)

    def log_param(self, param_name: str, param_value: Any) -> None:
        self.logger.debug(f"Parameter logged: {param_name} = {param_value}")
        self.handler.log_param(param_name, param_value)

    def log_artifact(self, artifact_path: str) -> None:
        self.logger.debug(f"Artifact logged: {artifact_path}")
        self.handler.log_artifact(artifact_path)
