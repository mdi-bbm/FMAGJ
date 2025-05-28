from benchmark.metric.ground_truth.common.average_precision import AveragePrecision
from benchmark.preprocessing.object_detection import ObjectDetectionPreprocessingOperation
from benchmark.task.base import BaseBenchmarkTask, METRIC_PREPROCESSING_TYPE


class ObjectDetectionTask(BaseBenchmarkTask):
    @property
    def metric_preprocessing(self) -> METRIC_PREPROCESSING_TYPE:
        return {
            AveragePrecision(): ObjectDetectionPreprocessingOperation()
        }
