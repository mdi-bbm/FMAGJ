from typing import Final, Any, Mapping

from benchmark.core.model.base_model import PydanticFrozen
from benchmark.dataset.detected_object_info import DetectedObjectInfo
from benchmark.metric.metric_input import MetricInputName
from benchmark.preprocessing.base import BasePreprocessingOperation
from benchmark.sample.sample import ObjectDetectionSample


class MatchedBoundingBoxes(PydanticFrozen):
    ground_truth: list[DetectedObjectInfo]
    predictions: list[DetectedObjectInfo]
    matched_ground_truth_prediction_indices: Mapping[int, int]


class BoundingBoxMatcher:
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def match_object_info(
        self,
        ground_truth_detections: list[DetectedObjectInfo],
        predicted_detections: list[DetectedObjectInfo]
    ) -> MatchedBoundingBoxes:
        ground_truth_list = list(set(ground_truth_detections))
        predicted_list = list(set(predicted_detections))
        matched_ground_truth_prediction_indices = {}

        for prediction_idx, prediction in enumerate(predicted_list):
            if prediction_idx in matched_ground_truth_prediction_indices.values():
                continue

            best_ground_truth_idx = None
            best_iou = 0.

            for ground_truth_idx, ground_truth in enumerate(ground_truth_list):
                if ground_truth_idx in matched_ground_truth_prediction_indices.keys():
                    continue

                iou = prediction.bbox.iou(ground_truth.bbox)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_ground_truth_idx = ground_truth_idx
                    best_iou = iou

            if best_ground_truth_idx is not None:
                matched_ground_truth_prediction_indices[best_ground_truth_idx] = prediction_idx

        return MatchedBoundingBoxes(
            ground_truth=ground_truth_list,
            predictions=predicted_list,
            matched_ground_truth_prediction_indices=matched_ground_truth_prediction_indices
        )


class ObjectDetectionPreprocessingOperation(BasePreprocessingOperation):
    DEFAULT_IOU_MATCHING_THRESHOLD: Final[float] = 0.5

    def __init__(self, iou_threshold: float = DEFAULT_IOU_MATCHING_THRESHOLD):
        self.matcher = BoundingBoxMatcher(iou_threshold)

    def run(self, sample: ObjectDetectionSample) -> dict[MetricInputName, Any]:
        matched_detection_info = self.matcher.match_object_info(
            ground_truth_detections=sample.ground_truth_detections,
            predicted_detections=sample.predicted_detections
        )
        return {
            MetricInputName.PREDICTION: matched_detection_info.predictions,
            MetricInputName.GROUND_TRUTH: matched_detection_info.ground_truth,
            MetricInputName.MATCH_GROUND_TRUTH_PREDICTIONS: (
                matched_detection_info.matched_ground_truth_prediction_indices
            )
        }
