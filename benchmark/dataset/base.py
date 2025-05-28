from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from benchmark.core.model.base_model import PydanticFrozen
from benchmark.sample.sample import BaseSample


class BaseBenchmarkDatasetConfig(PydanticFrozen, ABC):
    pass


class BaseBenchmarkDataset(Dataset, ABC):
    def __init__(self, config: BaseBenchmarkDatasetConfig) -> None:
        self.config = config

    @abstractmethod
    def get_sample_data(self, idx: int) -> BaseSample:
        pass

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> BaseSample:
        return self.get_sample_data(idx)


class RamDataset(ABC):
    pass


class DiskDataset(ABC):
    pass


class BaseDatasetLoader(ABC):
    def __init__(self, config: BaseBenchmarkDatasetConfig):
        self.config = config

    @abstractmethod
    def load(self, path: Path) -> Any:
        ...
