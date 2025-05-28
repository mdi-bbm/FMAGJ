import numpy as np
from numpy.typing import NDArray

from benchmark.metric.base import ImageMulticlassMetric
from benchmark.metric.ground_truth.base import GroundTruthMetric
from benchmark.core.model.class_mask import ClassMask
from benchmark.core.model.image import CV2Mask


def binary_iou_score(prediction: CV2Mask, ground_truth: CV2Mask) -> float:
    """
    Calculate the Intersection over Union (IoU) score for binary masks.

    IoU is a metric used to measure the overlap between two binary masks.
    It is calculated as the area of intersection divided by the area of union
    of the predicted and ground truth masks.

    :param prediction: Binary mask for the predicted segmentation.
        - Format: 2D numpy array of boolean values where 1 indicates the presence of an object.
        - Example: np.array([[0, 1], [1, 1]], dtype=bool)

    :param ground_truth: Binary mask for the ground truth segmentation.
        - Format: 2D numpy array of boolean values where 1 indicates the presence of an object.
        - Example: np.array([[1, 1], [0, 1]], dtype=bool)

    :return: IoU score as a float.
        - Range: [0, 1], where 1 indicates perfect overlap and 0 indicates no overlap.
        - If both masks are empty, IoU is defined to be 1.
        - If one mask is empty and the other is not, IoU is 0.

    Method:
        - Calculates the intersection and union areas using logical operations.
        - Returns 1 if both masks are empty (no objects in either).
        - Returns 0 if one mask is empty and the other is not.
    """
    intersection = np.logical_and(ground_truth, prediction).sum()
    union = np.logical_or(ground_truth, prediction).sum()
    if union == 0:
        is_both_masks_empty = np.count_nonzero(ground_truth) == 0 and np.count_nonzero(prediction) == 0
        if is_both_masks_empty:
            return 1.
        else:
            return 0.
    return intersection / union


class MeanIoU(GroundTruthMetric, ImageMulticlassMetric):
    def __init__(self, labels: list[np.uint8]):
        """
        Initialize the mIoU metric for multiclass segmentation.

        :param labels: intensity values per class
        """
        GroundTruthMetric.__init__(self)
        ImageMulticlassMetric.__init__(self, labels=labels)

    def calculate(self, prediction: NDArray, ground_truth: NDArray) -> float:
        """
        Calculate the IoU score for each class in a multiclass segmentation task and average result.

        :param prediction: Multiclass prediction mask, where each pixel has a class index.
            - Format: 2D or 3D numpy array of integers with shape (H, W) for 2D images or (H, W, D) for 3D images.
            - Example: np.array([[0, 1], [1, 2]], dtype=int), where each integer represents a class index.

        :param ground_truth: Multiclass ground truth mask, where each pixel has a class index.
            - Format: 2D or 3D numpy array of integers with shape (H, W) for 2D images or (H, W, D) for 3D images.
            - Example: np.array([[0, 1], [1, 2]], dtype=int), where each integer represents a class index.

        :return: IoU score averaged by classes.

        Example Usage:
            >>> iou_metric = MeanIoU(labels=[0, 1, 2])
            >>> prediction_sample = np.array([[0, 1], [1, 2]])
            >>> ground_truth_sample = np.array([[0, 1], [2, 2]])
            >>> iou_score_sample = iou_metric.calculate(prediction_sample, ground_truth_sample)
            >>> print(iou_score_sample)
        """
        iou_scores = []
        for label in self.labels:
            prediction_mask = ClassMask(image=prediction, intensity=label).get_value()
            ground_truth_mask = ClassMask(image=ground_truth, intensity=label).get_value()
            iou_score = binary_iou_score(prediction=prediction_mask, ground_truth=ground_truth_mask)
            iou_scores.append(iou_score)
        return np.mean(iou_scores).item()
