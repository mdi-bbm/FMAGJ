from abc import ABC, abstractmethod

from benchmark.metric.base import BaseMetric


class GroundTruthMetric(BaseMetric, ABC):
    @abstractmethod
    def calculate(self, *args, **kwargs) -> float:
        pass
