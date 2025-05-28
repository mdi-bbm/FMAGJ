from pathlib import Path

from benchmark.dataset.base import BaseBenchmarkDataset
from benchmark.evaluator.base import EvaluatorConfig
from benchmark.logger.factory import create_default_console_logger
from benchmark.evaluator.mean import MeanMetricsEvaluator
from benchmark.task.base import BaseBenchmarkTask
from benchmark.core.mlflow_handler.handler import MLFlowHandler
from benchmark.core.mlflow_handler.logger import MLFlowLogger
from benchmark.core.mlflow_handler.version import MLFlowVersion, git_sha


def create_mean_metrics_evaluator(
    task: BaseBenchmarkTask,
    dataset: BaseBenchmarkDataset,
    mlflow_dir: Path | None,
    dataset_version: str | None,
    model_version: str | None,
) -> MeanMetricsEvaluator:
    version = MLFlowVersion(
        model=model_version,
        benchmark=git_sha(),
        task=task.__class__.__name__,
        dataset=dataset.__class__.__name__ + ('_' + dataset_version) if dataset_version is not None else ''
    )
    handler = MLFlowHandler(dir_mlflow=mlflow_dir, version=version)
    mlflow_logger = MLFlowLogger(handler=handler)
    console_logger = create_default_console_logger()

    evaluator_config = EvaluatorConfig(console_logger=console_logger, mlflow_logger=mlflow_logger, task=task)
    return MeanMetricsEvaluator(config=evaluator_config)
