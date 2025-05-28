from numpy.typing import NDArray

from benchmark.metric.metric_input import MetricInputName
from benchmark.preprocessing.base import BasePreprocessingOperation
from benchmark.sample.sample import ImageSample


class ImagePreprocessingOperation(BasePreprocessingOperation):
    def run(self, sample: ImageSample) -> dict[MetricInputName, NDArray[int]]:
        return {
            MetricInputName.PREDICTION: sample.predicted,
            MetricInputName.GROUND_TRUTH: sample.ground_truth,
        }
