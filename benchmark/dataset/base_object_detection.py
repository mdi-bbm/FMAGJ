import warnings
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping, Type, Final

from benchmark.dataset.base import BaseBenchmarkDataset, DiskDataset, BaseBenchmarkDatasetConfig
from benchmark.sample.sample import ObjectCountSample


class BaseObjectDetectionGroundTruthDatasetConfig(BaseBenchmarkDatasetConfig):
    prediction_dir: Path
    ground_truth_dir: Path
    forced_filenames_path: Path | None
    prediction_extension: str = '.csv'
    ground_truth_extension: str = '.csv'


class BaseObjectDetectionDatasetLoader(ABC):
    LABEL_COLUMN_NAME: Final[str] = 'label_name'

    def __init__(self, config: BaseObjectDetectionGroundTruthDatasetConfig):
        self.config = config

    @abstractmethod
    def load(self, path: Path):
        ...


class BaseObjectDetectionGroundTruthDataset(BaseBenchmarkDataset, DiskDataset, ABC):
    def __init__(self, config: BaseObjectDetectionGroundTruthDatasetConfig):
        super().__init__(config=config)

    def __len__(self) -> int:
        return len(self.common_filenames)

    @cached_property
    def forced_filenames(self) -> list[str] | None:
        if self.config.forced_filenames_path is None:
            return None
        with open(self.config.forced_filenames_path, 'r') as file:
            filenames = [filename.strip() for filename in file.readlines()]
        non_empty_filenames = [filename for filename in filenames if len(filename) > 0]
        if len(non_empty_filenames) == 0:
            raise ValueError('If forced filenames used, their count must be greater than 0')
        return non_empty_filenames

    @cached_property
    def common_filenames(self) -> list[str]:
        common_list = list(set(self.prediction_dict.keys()) & set(self.ground_truth_dict.keys()))
        if self.forced_filenames is not None:
            common_list = list(set(common_list) & set(self.forced_filenames))
            if len(common_list) < len(self.forced_filenames):
                count_not_found_in_forced = len(self.forced_filenames) - len(common_list)
                warnings.warn(
                    f'Some filenames from list of forced filenames are not found: {count_not_found_in_forced}'
                )
        if len(common_list) == 0:
            warnings.warn('Empty common files list: no files to process')
        return common_list

    @cached_property
    def prediction_dict(self) -> dict[str, Any]:
        return self._load_dir(path_dir=self.config.prediction_dir, file_extension=self.config.prediction_extension)

    @cached_property
    def ground_truth_dict(self) -> dict[str, Any]:
        return self._load_dir(path_dir=self.config.ground_truth_dir, file_extension=self.config.ground_truth_extension)

    @property
    @abstractmethod
    def format_parser(self) -> Mapping[str, Type[BaseObjectDetectionDatasetLoader]]:
        ...

    def _load_dir(self, path_dir: Path, file_extension: str) -> dict[str, Any]:
        if file_extension in self.format_parser:
            loader_cls = self.format_parser[file_extension]
        else:
            raise NotImplementedError('Unsupported file extension')
        # noinspection PyTypeChecker
        loader = loader_cls(config=self.config)
        data = {
            path_filename.stem: loader.load(path=path_filename)
            for path_filename in path_dir.iterdir()
            if path_filename.suffix == file_extension
        }
        return data

    @abstractmethod
    def get_sample_data(self, idx: int) -> ObjectCountSample:
        ...
