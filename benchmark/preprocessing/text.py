from numpy.typing import NDArray

from benchmark.metric.metric_input import MetricInputName
from benchmark.preprocessing.base import BasePreprocessingOperation
from benchmark.sample.sample import TextSample
from benchmark.core.model.text_embedder import TextEmbedderBase


class TextBasicPreprocessingOperation(BasePreprocessingOperation):
    def run(self, sample: TextSample) -> dict[MetricInputName, str]:
        return {
            MetricInputName.PREDICTION: sample.predicted.strip(),
            MetricInputName.GROUND_TRUTH: sample.ground_truth.strip(),
        }


class MaxSizeEmbeddingPreprocessingOperation(BasePreprocessingOperation):
    def __init__(self, embedder: TextEmbedderBase):
        self.embedder = embedder

    def run(self, sample: TextSample) -> dict[MetricInputName, NDArray]:
        return {
            MetricInputName.PREDICTION: self.embedder(sample.predicted),
            MetricInputName.GROUND_TRUTH: self.embedder(sample.ground_truth)
        }
