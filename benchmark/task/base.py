from abc import ABC, abstractmethod

from benchmark.metric.base import BaseMetric
from benchmark.preprocessing.base import BasePreprocessingOperation
from benchmark.sample.sample import BaseSample

METRIC_PREPROCESSING_TYPE = dict[BaseMetric, BasePreprocessingOperation]


class BaseBenchmarkTask(ABC):
    @property
    @abstractmethod
    def metric_preprocessing(self) -> METRIC_PREPROCESSING_TYPE:
        pass

    @property
    def metric_names(self) -> list[str]:
        return [metric.__class__.__name__ for metric in self.metric_preprocessing.keys()]

    def run(self, sample: BaseSample) -> dict[str, float]:
        metric_values = {}
        for metric, preprocessing_operation in self.metric_preprocessing.items():
            input_data = preprocessing_operation(sample=sample)
            input_data_dict = {
                metric_input_name.value: metric_input
                for metric_input_name, metric_input in input_data.items()
            }
            metric_value = metric.calculate(**input_data_dict)
            metric_values[metric.__class__.__name__] = metric_value
        return metric_values
