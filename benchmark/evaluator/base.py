import logging
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from benchmark.core.model.base_model import PydanticFrozen
from benchmark.dataset.base import BaseBenchmarkDataset
from benchmark.logger.factory import create_null_logger
from benchmark.task.base import BaseBenchmarkTask
from benchmark.core.mlflow_handler.logger import MLFlowLogger


class EvaluatorConfig(PydanticFrozen):
    console_logger: logging.Logger = Field(default_factory=create_null_logger)
    mlflow_logger: MLFlowLogger
    task: BaseBenchmarkTask


SAMPLE_METRICS = dict[str, float]


class BaseEvaluator(ABC):
    def __init__(self, config: EvaluatorConfig):
        self.config = config

    def evaluate(self, dataset: BaseBenchmarkDataset) -> SAMPLE_METRICS:
        metric_values_dataset = []
        for sample_idx, sample in enumerate(dataset):
            metric_values = self.config.task.run(sample=sample)
            self._log_metric_values(sample_idx=sample_idx, sample_metrics=metric_values)
            metric_values_dataset.append(metric_values)
        metric_values_dict = self._calculate_metric_values_dict(metric_values_dataset=metric_values_dataset)
        aggregated_metrics = self._aggregate_metrics(metric_values_dict=metric_values_dict)
        return aggregated_metrics

    @abstractmethod
    def _aggregate_metrics(self, metric_values_dict: dict[str, NDArray[float]]) -> SAMPLE_METRICS:
        pass

    def _calculate_metric_values_dict(self, metric_values_dataset: list[SAMPLE_METRICS]) -> dict[str, NDArray[float]]:
        if len(metric_values_dataset) == 0:
            return {}
        metric_values_dict: dict[str, NDArray[float]] = {}
        for metric_name in self.config.task.metric_names:
            metric_values = [metric_values_sample[metric_name] for metric_values_sample in metric_values_dataset]
            metric_values_dict[metric_name] = np.asarray(metric_values)
        return metric_values_dict

    def _log_metric_values(self, sample_idx: int, sample_metrics: SAMPLE_METRICS) -> None:
        self.config.console_logger.info(f'Sample idx {sample_idx}, metrics: {sample_metrics}')
        for metric_name, metric_value in sample_metrics.items():
            self.config.mlflow_logger.log_metric(
                metric_name=metric_name,
                metric_value=metric_value,
                sample_idx_or_epoch=sample_idx
            )

    def _log_aggregated_metric_values(self, sample_metrics: SAMPLE_METRICS, aggregation_type_str: str) -> None:
        self.config.console_logger.info(f'{aggregation_type_str} metrics: {sample_metrics}')
        for metric_name, metric_value in sample_metrics.items():
            self.config.mlflow_logger.log_metric(
                metric_name=f'{metric_name}_{aggregation_type_str}',
                metric_value=metric_value
            )
