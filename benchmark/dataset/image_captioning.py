from abc import ABC
from functools import cached_property
from pathlib import Path

import pandas as pd

from benchmark.dataset.base import BaseBenchmarkDataset, DiskDataset, BaseBenchmarkDatasetConfig, BaseDatasetLoader
from benchmark.sample.sample import TextSample


class ImageCaptioningGroundTruthDatasetConfig(BaseBenchmarkDatasetConfig):
    path_csv: Path
    prediction_column: str
    ground_truth_column: str


class CsvModelAnswerLoader(BaseDatasetLoader):
    def __init__(self, config: ImageCaptioningGroundTruthDatasetConfig):
        super().__init__(config=config)

    def load(self, path: Path) -> list[TextSample]:
        data = pd.read_csv(path)
        if self.config.prediction_column not in data.columns:
            raise ValueError(f'Column {self.config.prediction_column} not found in CSV data')
        if self.config.ground_truth_column not in data.columns:
            raise ValueError(f'Column {self.config.ground_truth_column} not found in CSV data')
        predictions = data[self.config.prediction_column]
        ground_truth = data[self.config.ground_truth_column]
        samples = [
            TextSample(predicted=predicted, ground_truth=ground_truth_str)
            for predicted, ground_truth_str in zip(predictions, ground_truth)
        ]
        return samples


class ImageCaptioningGroundTruthDataset(BaseBenchmarkDataset, DiskDataset, ABC):
    def __init__(self, config: ImageCaptioningGroundTruthDatasetConfig):
        super().__init__(config=config)

    def __len__(self) -> int:
        return len(self.samples)

    @cached_property
    def samples(self) -> list[TextSample]:
        # noinspection PyTypeChecker
        return CsvModelAnswerLoader(config=self.config).load(self.config.path_csv)

    def get_sample_data(self, idx: int) -> TextSample:
        return self.samples[idx]
