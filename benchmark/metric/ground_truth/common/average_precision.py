from typing import Mapping

import numpy as np
from sklearn.metrics import average_precision_score

from benchmark.dataset.detected_object_info import DetectedObjectInfo
from benchmark.metric.ground_truth.base import GroundTruthMetric


class AveragePrecision(GroundTruthMetric):
    def calculate(
        self,
        prediction: list[DetectedObjectInfo],
        ground_truth: list[DetectedObjectInfo],
        ground_truth_prediction_match: Mapping[int, int]
    ) -> float:
        """
        Calculate the Average Precision (AP) score for object detection.

        :param prediction: List of `DetectedObjectInfo` representing predicted bounding boxes
                           with confidence scores.
        :param ground_truth: List of `DetectedObjectInfo` representing ground truth bounding boxes.
        :param ground_truth_prediction_match: Mapping between ground truth indices and prediction indices
                                              where a match occurs.

        :return: Average Precision score as a float.

        This implementation uses scikit-learn's `average_precision_score` to calculate the area under
        the Precision-Recall curve. It supports multiclass labels natively by converting inputs to a
        (n_samples, n_classes) format.
        """
        labels = {gt.label_name for gt in ground_truth} | {pred.label_name for pred in prediction}

        label_mapping = {label: idx for idx, label in enumerate(sorted(labels))}

        y_true = np.zeros((len(ground_truth), len(label_mapping)), dtype=bool)
        y_score = np.zeros((len(ground_truth), len(label_mapping)))

        for ground_truth_idx, ground_truth_info in enumerate(ground_truth):
            label_idx = label_mapping[ground_truth_info.label_name]
            y_true[ground_truth_idx, label_idx] = True
            if ground_truth_idx in ground_truth_prediction_match:
                prediction_idx = ground_truth_prediction_match[ground_truth_idx]
                prediction_label_idx = label_mapping[prediction[prediction_idx].label_name]
                y_score[ground_truth_idx][prediction_label_idx] = prediction[prediction_idx].confidence

        unmatched_prediction_indices = [
            idx for idx in range(len(prediction))
            if idx not in ground_truth_prediction_match.values()
        ]
        for prediction_idx in unmatched_prediction_indices:
            y_true = np.vstack([y_true, np.zeros((1, len(label_mapping)), dtype=bool)])
            score_row = np.zeros((1, len(label_mapping)), dtype=float)
            prediction_label_idx = label_mapping[prediction[prediction_idx].label_name]
            score_row[0, prediction_label_idx] = prediction[prediction_idx].confidence
            y_score = np.vstack([y_score, score_row])

        return average_precision_score(y_true=y_true, y_score=y_score, average="weighted")
