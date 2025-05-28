from abc import ABC, abstractmethod

import numpy as np


class BaseMetric(ABC):
    @abstractmethod
    def calculate(self, *args, **kwargs) -> float:
        pass


class ImageMulticlassMetric:
    def __init__(self, labels: list[np.uint8]):
        self.labels = labels
