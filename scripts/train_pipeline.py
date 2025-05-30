import os
import logging
import argparse

from train import train_detection_model
from inference import run_predict

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser(description='Object Detection Training Pipeline')
    parser.add_argument('--presets',
                        type=str,
                        default='high_quality',
                        choices=['best_quality', 'high_quality', 'good_quality', 'medium_quality'],
                        help='Model training presets (default: high_quality)')
    parser.add_argument('--max_epochs',
                        type=int,
                        default=1,
                        help='Maximum number of training epochs (default: 1)')
    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help='Training batch size (default: 4)')
    parser.add_argument('--num_workers',
                        type=int,
                        default=2,
                        help='Number of data loading workers (default: 2)')
    parser.add_argument('--num_gpus',
                        type=int,
                        default=1,
                        help='Number of GPUs to use (default: 1)')
    parser.add_argument('--skip_training',
                        action='store_true',
                        help='Skip training phase and run only inference')
    parser.add_argument('--skip_inference',
                        action='store_true',
                        help='Skip inference phase and run only training')

    args = parser.parse_args()

    storage_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    train_data_folder = os.path.join(storage_root, 'data', 'dataset')
    data_folder = os.path.join(storage_root, 'data', 'for_inference')

    try:
        if not args.skip_training:
            logger.info(f'TRAINING with presets: {args.presets}')
            train_detection_model(
                train_data_folder,
                presets=args.presets,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                num_gpus=args.num_gpus
            )

        if not args.skip_inference:
            logger.info('INFERENCE')
            run_predict(data_folder)

    except Exception as e:
        logger.error(f"Error processing tasks: {e}")


if __name__ == "__main__":
    main()