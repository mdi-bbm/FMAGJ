from benchmark.metric.ground_truth.common.mae import MAE
from benchmark.metric.ground_truth.common.mape import MAPE
from benchmark.preprocessing.object_count import ObjectCountPreprocessingOperation
from benchmark.task.base import BaseBenchmarkTask, METRIC_PREPROCESSING_TYPE


class ObjectCountingTask(BaseBenchmarkTask):
    @property
    def metric_preprocessing(self) -> METRIC_PREPROCESSING_TYPE:
        return {
            MAE(): ObjectCountPreprocessingOperation(),
            MAPE(): ObjectCountPreprocessingOperation()
        }
