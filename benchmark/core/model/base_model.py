from pydantic import BaseModel, ConfigDict
from abc import ABC


class PydanticBase(BaseModel, ABC):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        from_attributes=True,
        populate_by_name=True
    )


class PydanticFrozen(BaseModel, ABC):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        from_attributes=True,
        populate_by_name=True,
        frozen=True
    )
