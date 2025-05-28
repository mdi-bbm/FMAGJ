from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.metrics import f1_score as sklearn_f1_score

from benchmark.metric.base import ImageMulticlassMetric
from benchmark.metric.ground_truth.base import GroundTruthMetric
from benchmark.core.model.image import CV2Image


class BaseF1Score(GroundTruthMetric, ABC):
    @abstractmethod
    def calculate(self, prediction: Any, ground_truth: Any) -> float:
        pass


class ImageF1Score(BaseF1Score, ImageMulticlassMetric):
    def __init__(self, labels: list[np.uint8]):
        GroundTruthMetric.__init__(self)
        ImageMulticlassMetric.__init__(self, labels=labels)

    def calculate(self, prediction: CV2Image, ground_truth: CV2Image) -> float:
        return sklearn_f1_score(
            ground_truth.flatten(),
            prediction.flatten(),
            labels=self.labels,
            average='macro',
            zero_division=0.
        )


class TextTokenF1Score(BaseF1Score):
    def calculate(self, prediction: list, ground_truth: list) -> float:
        """
        Calculate the F1 Score for text data in tokenized format (list of tokens).

        The F1 Score is calculated by treating the presence of each unique token as a binary feature.
        For each unique token, the model checks if it is present in both `prediction` and `ground_truth`,
        then calculates Precision, Recall, and F1 based on the overlap.

        :param prediction: List of predicted tokens.
            - Format: list of strings, where each string is a token.
            - Example: ["the", "cat", "is", "on", "the", "mat"]

        :param ground_truth: List of ground truth tokens.
            - Format: list of strings, where each string is a token.
            - Example: ["the", "cat", "sat", "on", "the", "mat"]

        :return: F1 Score as a float, representing the harmonic mean of Precision and Recall.

        Method:
            - Combines tokens from both `prediction` and `ground_truth` to create a unique set of tokens.
            - For each token in this unique set:
                - Creates a binary indicator (1 if the token is present, 0 if not) for both `prediction`
                  and `ground_truth`.
            - Calculates F1 Score between the two binary vectors using `sklearn_f1_score`.

        Example Usage:
            >>> f1_metric = TextTokenF1Score()
            >>> prediction_list = ["the", "cat", "is", "on", "the", "mat"]
            >>> ground_truth_list = ["the", "cat", "sat", "on", "the", "mat"]
            >>> f1_score_value = f1_metric.calculate(prediction, ground_truth)
            >>> print(f"F1 Score: {f1_score_value}")

        """
        unique_tokens = list(set(prediction + ground_truth))
        pred_binary = [1 if token in prediction else 0 for token in unique_tokens]
        gt_binary = [1 if token in ground_truth else 0 for token in unique_tokens]
        score = sklearn_f1_score(gt_binary, pred_binary)
        return score
