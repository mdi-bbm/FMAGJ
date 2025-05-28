from pathlib import Path
from typing import Mapping, Type

import pandas as pd

from benchmark.dataset.base_object_detection import (
    BaseObjectDetectionGroundTruthDataset,
    BaseObjectDetectionGroundTruthDatasetConfig,
    BaseObjectDetectionDatasetLoader
)
from benchmark.sample.sample import ObjectCountSample


class ObjectCountsGroundTruthDatasetConfig(BaseObjectDetectionGroundTruthDatasetConfig):
    label: str | None = None
    label_case_sensitive: bool = False


class CsvObjectCountsLoader(BaseObjectDetectionDatasetLoader):
    def __init__(self, config: ObjectCountsGroundTruthDatasetConfig):
        super().__init__(config=config)

    def load(self, path: Path) -> int:
        data = pd.read_csv(path)
        labels = data[self.LABEL_COLUMN_NAME]
        filtered_labels = labels.str.contains(self.config.label, case=self.config.label_case_sensitive)
        return filtered_labels.sum()


class ObjectCountsGroundTruthDataset(BaseObjectDetectionGroundTruthDataset):
    def __init__(self, config: ObjectCountsGroundTruthDatasetConfig):
        super().__init__(config=config)
    
    @property
    def format_parser(self) -> Mapping[str, Type[BaseObjectDetectionDatasetLoader]]:
        return {
            '.csv': CsvObjectCountsLoader
        }

    def get_sample_data(self, idx: int) -> ObjectCountSample:
        filename = self.common_filenames[idx]
        return ObjectCountSample(
            predicted_count=self.prediction_dict[filename],
            ground_truth_count=self.ground_truth_dict[filename]
        )
