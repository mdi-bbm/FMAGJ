from abc import ABC, abstractmethod
from typing import Any


class BaseMLFlowHandler(ABC):
    @abstractmethod
    def log_metric(
        self,
        metric_name: str,
        metric_value: float,
        sample_idx_or_epoch: int | None = None
    ) -> None:
        pass

    @abstractmethod
    def log_param(self, param_name: str, param_value: Any) -> None:
        pass

    @abstractmethod
    def log_artifact(self, artifact_path: str) -> None:
        pass
