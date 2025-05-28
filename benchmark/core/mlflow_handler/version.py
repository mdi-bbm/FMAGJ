import os

from benchmark.core.model.base_model import PydanticFrozen


def git_sha() -> str:
    return os.popen('git rev-parse HEAD').read().strip()


class MLFlowVersion(PydanticFrozen):
    model: str | None = None
    benchmark: str | None = None
    dataset: str | None = None
    task: str | None = None

    def __str__(self) -> str:
        fields = []
        if self.model:
            fields.append(self.model)
        if self.benchmark:
            fields.append(self.benchmark)
        if self.dataset:
            fields.append(self.dataset)
        if self.task:
            fields.append(self.task)
        if len(fields) == 0:
            return 'NO_VERSION'
        return '_'.join(fields)
