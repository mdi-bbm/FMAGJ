import numpy as np

from benchmark.metric.ground_truth.common.f1_score import ImageF1Score
from benchmark.metric.ground_truth.image.iou import MeanIoU
from benchmark.preprocessing.image import ImagePreprocessingOperation
from benchmark.task.base import BaseBenchmarkTask, METRIC_PREPROCESSING_TYPE


class SemanticSegmentationTask(BaseBenchmarkTask):
    def __init__(self, labels: list[np.uint8]):
        self.labels = labels

    @property
    def metric_preprocessing(self) -> METRIC_PREPROCESSING_TYPE:
        return {
            MeanIoU(labels=self.labels): ImagePreprocessingOperation(),
            ImageF1Score(labels=self.labels): ImagePreprocessingOperation()
        }
