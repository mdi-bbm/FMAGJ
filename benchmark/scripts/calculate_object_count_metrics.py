import argparse
from pathlib import Path

from benchmark.dataset.object_counts import ObjectCountsGroundTruthDataset, ObjectCountsGroundTruthDatasetConfig
from benchmark.evaluator.factory import create_mean_metrics_evaluator
from benchmark.task.object_counting import ObjectCountingTask


def main() -> None:
    parser = argparse.ArgumentParser(description='Run benchmark for object counting')
    parser.add_argument(
        '--prediction_dir',
        type=Path,
        default=Path(__file__).parents[1] / 'data_examples' / 'object_detections' / 'predictions',
        help='Path to dataset with predicted files'
    )
    parser.add_argument(
        '--ground_truth_dir',
        type=Path,
        default=Path(__file__).parents[1] / 'data_examples' / 'object_detections' / 'ground_truth',
        help='Path to dataset with ground truth files'
    )
    parser.add_argument(
        '--forced_filenames_path',
        type=Path,
        required=False,
        help='Path to file with list of filenames to process'
    )
    parser.add_argument(
        '--label',
        type=str,
        help='Label of objects to count'
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

    task = ObjectCountingTask()
    dataset = ObjectCountsGroundTruthDataset(config=ObjectCountsGroundTruthDatasetConfig(
        prediction_dir=args.prediction_dir,
        ground_truth_dir=args.ground_truth_dir,
        forced_filenames_path=args.forced_filenames_path,
        label=args.label
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
