from abc import ABC, abstractmethod
from typing import Any

from benchmark.metric.metric_input import MetricInputName
from benchmark.sample.sample import BaseSample


class BasePreprocessingOperation(ABC):
    @abstractmethod
    def run(self, sample: BaseSample) -> dict[MetricInputName, Any]:
        pass

    def __call__(self, sample: BaseSample) -> dict[MetricInputName, Any]:
        return self.run(sample=sample)
