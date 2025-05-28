import numpy as np
from numpy.typing import NDArray

from benchmark.evaluator.base import BaseEvaluator, SAMPLE_METRICS


class MeanMetricsEvaluator(BaseEvaluator):
    def _aggregate_metrics(self, metric_values_dict: dict[str, NDArray[float]]) -> SAMPLE_METRICS:
        mean_metric_value: SAMPLE_METRICS = {}
        for metric_name, values in metric_values_dict.items():
            mean_metric_value[metric_name] = np.nanmean(values).item()
        self._log_aggregated_metric_values(sample_metrics=mean_metric_value, aggregation_type_str='mean')
        return mean_metric_value
