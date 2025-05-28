import warnings
from functools import cached_property
from pathlib import Path

import cv2

from benchmark.dataset.base import BaseBenchmarkDatasetConfig, BaseBenchmarkDataset, DiskDataset
from benchmark.sample.sample import ImageSample


class ImagesGroundTruthDatasetConfig(BaseBenchmarkDatasetConfig):
    prediction_dir: Path
    ground_truth_dir: Path
    filename_extension: str
    imread_flags: int = 0


class ImagesGroundTruthDataset(BaseBenchmarkDataset, DiskDataset):
    def __init__(self, config: ImagesGroundTruthDatasetConfig):
        super().__init__(config=config)
        self.prediction_filename_list = self._load_filenames_list(dir_path=self.config.prediction_dir)
        self.ground_truth_filename_list = self._load_filenames_list(dir_path=self.config.ground_truth_dir)

    def __len__(self) -> int:
        return len(self.common_filenames)

    @cached_property
    def common_filenames(self) -> list[str]:
        common_filename_list = list(set(self.prediction_filename_list) & set(self.ground_truth_filename_list))
        if len(common_filename_list) == 0:
            warnings.warn('Empty common files list: no files to process')
        return common_filename_list

    def get_sample_data(self, idx: int) -> ImageSample:
        common_filename = self.common_filenames[idx]
        predicted_path = self.config.prediction_dir / common_filename
        ground_truth_path = self.config.ground_truth_dir / common_filename
        predicted_image = cv2.imread(str(predicted_path.resolve()), flags=self.config.imread_flags)
        ground_truth_image = cv2.imread(str(ground_truth_path.resolve()), flags=self.config.imread_flags)
        return ImageSample(
            predicted=predicted_image,
            ground_truth=ground_truth_image
        )

    def _load_filenames_list(self, dir_path: Path) -> list[str]:
        return [
            path.name for path in dir_path.iterdir()
            if path.suffix == self.config.filename_extension
        ]
