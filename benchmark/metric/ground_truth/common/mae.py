import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error

from benchmark.metric.ground_truth.base import GroundTruthMetric


class MAE(GroundTruthMetric):
    def calculate(self, prediction: NDArray | list, ground_truth: NDArray | list) -> float:
        prediction = np.asarray(prediction)
        ground_truth = np.asarray(ground_truth)
        return mean_absolute_error(ground_truth, prediction)
