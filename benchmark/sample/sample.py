from benchmark.core.model.base_model import PydanticFrozen
from benchmark.core.model.image import CV2Image
from benchmark.dataset.detected_object_info import DetectedObjectInfo


class BaseSample(PydanticFrozen):
    ...


class ObjectCountSample(BaseSample):
    predicted_count: int
    ground_truth_count: int


class ObjectDetectionSample(BaseSample):
    predicted_detections: list[DetectedObjectInfo]
    ground_truth_detections: list[DetectedObjectInfo]


class ImageSample(BaseSample):
    predicted: CV2Image
    ground_truth: CV2Image


class TextSample(BaseSample):
    predicted: str
    ground_truth: str
