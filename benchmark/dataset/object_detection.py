from pathlib import Path
from typing import Mapping, Type

import pandas as pd
from pydantic import parse_obj_as

from benchmark.dataset.base_object_detection import (
    BaseObjectDetectionGroundTruthDataset,
    BaseObjectDetectionGroundTruthDatasetConfig,
    BaseObjectDetectionDatasetLoader
)
from benchmark.dataset.detected_object_info import DetectedObjectInfo
from benchmark.sample.sample import ObjectDetectionSample


class ObjectDetectionGroundTruthDatasetConfig(BaseObjectDetectionGroundTruthDatasetConfig):
    input_label_map: dict[str, str] | None = None
    forced_label: str | None = None


class CsvObjectDetectionLoader(BaseObjectDetectionDatasetLoader):
    def __init__(self, config: ObjectDetectionGroundTruthDatasetConfig):
        super().__init__(config=config)

    def load(self, path: Path) -> list[DetectedObjectInfo]:
        data = pd.read_csv(path)
        if data[self.LABEL_COLUMN_NAME].isna().any():
            raise ValueError('NaN values in input data')

        if self.config.input_label_map is not None:
            labels = data[self.LABEL_COLUMN_NAME]
            filtered_data = data.copy(deep=True)
            filtered_data[self.LABEL_COLUMN_NAME] = labels.map(self.config.input_label_map)
            if filtered_data[self.LABEL_COLUMN_NAME].isna().any():
                raise ValueError('NaN values detected in filtered data. Some input labels are not in input_label_map')
        else:
            filtered_data = data

        if self.config.forced_label is not None:
            filter_rule = filtered_data[self.LABEL_COLUMN_NAME] == self.config.forced_label
            filtered_data = filtered_data[filter_rule]

        detected_object_info = parse_obj_as(list[DetectedObjectInfo], filtered_data.to_dict(orient="records"))
        return detected_object_info


class ObjectDetectionGroundTruthDataset(BaseObjectDetectionGroundTruthDataset):
    def __init__(self, config: ObjectDetectionGroundTruthDatasetConfig):
        super().__init__(config=config)

    @property
    def format_parser(self) -> Mapping[str, Type[BaseObjectDetectionDatasetLoader]]:
        return {
            '.csv': CsvObjectDetectionLoader
        }

    def get_sample_data(self, idx: int) -> ObjectDetectionSample:
        filename = self.common_filenames[idx]
        return ObjectDetectionSample(
            predicted_detections=self.prediction_dict[filename],
            ground_truth_detections=self.ground_truth_dict[filename]
        )
