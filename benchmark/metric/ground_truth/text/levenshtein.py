from benchmark.metric.ground_truth.base import GroundTruthMetric
from Levenshtein import ratio


class NormalizedLevenshteinDistance(GroundTruthMetric):
    def calculate(self, prediction: str, ground_truth: str) -> float:
        """
            Calculate the Normalized Levenshtein Distance between two strings.

            Normalized Levenshtein Distance is a measure of the difference between
            two sequences (strings) and is normalized by dividing the raw Levenshtein
            distance by the maximum length of the two strings. The distance ranges
            from 0 to 1, where 0 indicates identical strings and 1 indicates maximum
            difference.

            :param prediction: Predicted text as a string.
                - Example: "kitten"

            :param ground_truth: Ground truth text as a string.
                - Example: "sitting"

            :return: Normalized Levenshtein Distance as a float.
                - Range: [0.0, 1.0]
                - 0.0 means the strings are identical.
                - 1.0 means the strings are completely different.

            Method:
                - Calculates the raw Levenshtein distance between `prediction` and `ground_truth`.
                - Normalizes this distance by dividing by the maximum length of the two strings.

            Example Usage:
                >>> distance_metric = NormalizedLevenshteinDistance()
                >>> prediction_str = "kitten"
                >>> ground_truth_str = "sitting"
                >>> distance = distance_metric.calculate(prediction_str, ground_truth_str)
                >>> print(f"Normalized Levenshtein Distance: {distance}")
        """
        return ratio(s1=prediction, s2=ground_truth)
