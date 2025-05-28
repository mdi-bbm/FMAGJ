from abc import ABC

from numpy.typing import NDArray

from benchmark.dataset.base import BaseBenchmarkDatasetConfig, BaseBenchmarkDataset, RamDataset


class PassthroughGroundTruthDatasetConfig(BaseBenchmarkDatasetConfig):
    prediction: list[int] | NDArray[int]
    ground_truth: list[int] | NDArray[int]


class PassthroughGroundTruthDataset(BaseBenchmarkDataset, RamDataset, ABC):
    def __init__(self, config: PassthroughGroundTruthDatasetConfig):
        super().__init__(config=config)
        if len(self.config.prediction) != len(self.config.ground_truth):
            raise ValueError(f'\
                prediction size ({len(self.config.prediction)}) must be equal to \
                ground_truth size ({len(self.config.ground_truth)})\
            ')

    def __len__(self) -> int:
        return len(self.config.prediction)
