from enum import Enum


class MetricInputName(Enum):
    PREDICTION = 'prediction'
    GROUND_TRUTH = 'ground_truth'
    MATCH_GROUND_TRUTH_PREDICTIONS = 'ground_truth_prediction_match'
