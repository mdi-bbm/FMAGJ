import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_percentage_error

from benchmark.metric.ground_truth.base import GroundTruthMetric


class MAPE(GroundTruthMetric):
    def calculate(self, prediction: NDArray, ground_truth: NDArray) -> float:
        adjusted_true_values = np.where((ground_truth == 0) & (prediction == 0), 1, ground_truth)
        non_zero_true_values_mask = adjusted_true_values != 0

        if not np.any(non_zero_true_values_mask):
            return float('nan')

        return mean_absolute_percentage_error(
            y_true=adjusted_true_values[non_zero_true_values_mask],
            y_pred=prediction[non_zero_true_values_mask]
        )
