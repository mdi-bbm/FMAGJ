from pydantic import Field

from benchmark.core.model.base_model import PydanticFrozen
from benchmark.core.model.bounding_box import BoundingBox2D


class DetectedObjectInfo(PydanticFrozen):
    label_name: str
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    image_name: str
    image_width: int
    image_height: int
    confidence: float = Field(default=1.0, description="Confidence score for the detection")

    @property
    def bbox(self) -> BoundingBox2D:
        return BoundingBox2D(
            left=self.bbox_x,
            top=self.bbox_y,
            width=self.bbox_width,
            height=self.bbox_height
        )
