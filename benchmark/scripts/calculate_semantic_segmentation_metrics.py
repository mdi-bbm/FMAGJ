import argparse
from pathlib import Path

import cv2
import numpy as np

from benchmark.dataset.images import ImagesGroundTruthDataset, ImagesGroundTruthDatasetConfig
from benchmark.evaluator.factory import create_mean_metrics_evaluator
from benchmark.task.semantic_segmentation import SemanticSegmentationTask


def main() -> None:
    parser = argparse.ArgumentParser(description='Run benchmark for semantic segmentation')
    parser.add_argument(
        '--prediction_dir',
        type=Path,
        default=Path(__file__).parents[1] / 'data_examples' / 'images' / 'predictions',
        help='Path to dataset with predicted images'
    )
    parser.add_argument(
        '--ground_truth_dir',
        type=Path,
        default=Path(__file__).parents[1] / 'data_examples' / 'images' / 'ground_truth',
        help='Path to dataset with ground truth images'
    )
    parser.add_argument(
        '--filename_extension',
        type=str,
        default='.png',
        help='Extension of image files'
    )
    parser.add_argument(
        '--labels',
        type=list,
        default=[255],
        help='List of all possible intensity values. For binary segmentation, set only intensity for positive class'
    )
    parser.add_argument(
        '--mlflow_dir',
        type=Path,
        default=Path(__file__).parents[2] / 'mlruns',
        help='Path to MLFlow directory'
    )
    parser.add_argument(
        '--model_version',
        required=False,
        type=str,
        help='Model version'
    )
    parser.add_argument(
        '--dataset_version',
        required=False,
        type=str,
        help='Dataset version'
    )
    args = parser.parse_args()

    labels = [np.uint8(value) for value in args.labels]
    task = SemanticSegmentationTask(labels=[np.uint8(value) for value in labels])

    dataset = ImagesGroundTruthDataset(config=ImagesGroundTruthDatasetConfig(
        prediction_dir=args.prediction_dir,
        ground_truth_dir=args.ground_truth_dir,
        filename_extension=args.filename_extension,
        imread_flags=cv2.IMREAD_GRAYSCALE
    ))

    evaluator = create_mean_metrics_evaluator(
        task=task,
        dataset=dataset,
        mlflow_dir=args.mlflow_dir,
        dataset_version=args.dataset_version,
        model_version=args.model_version,
    )
    _ = evaluator.evaluate(dataset=dataset)


if __name__ == "__main__":
    main()
