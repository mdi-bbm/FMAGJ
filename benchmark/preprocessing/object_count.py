import numpy as np
from numpy.typing import NDArray

from benchmark.metric.metric_input import MetricInputName
from benchmark.preprocessing.base import BasePreprocessingOperation
from benchmark.sample.sample import ObjectCountSample


class ObjectCountPreprocessingOperation(BasePreprocessingOperation):
    def run(self, sample: ObjectCountSample) -> dict[MetricInputName, NDArray[int]]:
        return {
            MetricInputName.PREDICTION: np.asarray([sample.predicted_count]),
            MetricInputName.GROUND_TRUTH: np.asarray([sample.ground_truth_count]),
        }
