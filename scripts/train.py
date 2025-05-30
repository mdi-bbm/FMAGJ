import os
import json
import random
import logging
import time
import shutil
from autogluon.multimodal import MultiModalPredictor
from lightning.pytorch import Trainer

# Automatically logs to a directory (by default ``lightning_logs/``)
trainer = Trainer()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_classes_from_coco_json(json_path: str) -> list:
    """Extract class names from COCO JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return [category['name'] for category in data['categories']]
    except Exception as e:
        logger.error(f"Failed to extract classes from {json_path}: {e}")
        raise


def create_batch_coco_json(images: list, coco_data: dict, temp_dir: str, stage: int, image_dir: str) -> str:
    """Create a batch COCO JSON file with corrected image paths."""
    batch_image_ids = {img['id'] for img in images}
    batch_data = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "images": images,
        "annotations": [ann for ann in coco_data['annotations'] if ann['image_id'] in batch_image_ids],
        "categories": coco_data['categories']
    }
    # Correct image paths
    for img in batch_data['images']:
        img['file_name'] = os.path.join(image_dir, os.path.basename(img['file_name']))
    temp_json = os.path.join(temp_dir, f'temp_batch_{stage}.json')
    try:
        with open(temp_json, 'w') as f:
            json.dump(batch_data, f)
        return temp_json
    except Exception as e:
        logger.error(f"Failed to create batch JSON {temp_json}: {e}")
        raise


def get_unique_model_path(base_path: str) -> str:
    """Generate a unique model save path with timestamp."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_path, f"model-{timestamp}")


def train_detection_model(train_data_folder: str,
                          model_base_path: str = "models",
                          presets: str = "high_quality",
                          max_epochs: int = 1,
                          batch_size: int = 4,
                          num_workers: int = 2,
                          num_gpus: int = 1):
    """Train an object detection model with batch processing."""
    train_json_path = os.path.join(train_data_folder, 'Annotations', 'train.json')
    image_dir = os.path.join(train_data_folder, 'images')
    temp_path = os.path.join('temp')
    os.makedirs(temp_path, exist_ok=True)

    # Create labels.txt
    labels_path = os.path.join(train_data_folder, 'labels.txt')
    class_names = extract_classes_from_coco_json(train_json_path)
    try:
        with open(labels_path, 'w') as f:
            f.write("\n".join(class_names))
    except Exception as e:
        logger.error(f"Failed to create labels.txt: {e}")

    # Load and shuffle data
    try:
        with open(train_json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {train_json_path}: {e}")

    images = data.get('images', [])
    if len(images) < 15:
        logger.warning(f"Only {len(images)} training images found. Consider adding more images for better training.")
    images = random.sample(images, len(images))
    logger.info(f"Found {len(images)} training images")

    # Generate unique model path
    model_save_path = get_unique_model_path(model_base_path)
    os.makedirs(model_save_path, exist_ok=True)
    trainer = Trainer(log_every_n_steps=1)

    # Log training parameters
    logger.info(f"Training parameters: presets={presets}, max_epochs={max_epochs}, "
                f"batch_size={batch_size}, num_workers={num_workers}, num_gpus={num_gpus}")

    # Initialize predictor
    try:
        predictor = MultiModalPredictor(
            problem_type="object_detection",
            presets=presets,
            sample_data_path=train_data_folder,
            hyperparameters={
                "optimization.max_epochs": max_epochs,
                "env.batch_size": batch_size,
                "env.num_workers": num_workers,
                "env.num_gpus": num_gpus,
                "env.strategy": "dp",
            },
            path=model_save_path
        )
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        raise

    batch_processing_size = 30
    try:
        # Try full dataset training
        logger.info("Attempting full dataset training")
        predictor.fit(
            train_data=train_json_path,
            time_limit=3600,
            save_path=model_save_path
        )
    except Exception as e:
        os.rmdir(model_save_path)
        logger.info(f"Model deleted: {model_save_path}")
        logger.warning(f"Full dataset training failed: {str(e)}")
        logger.info("Attempting batched training as fallback")

        # Batched training
        for i in range(0, len(images), batch_processing_size):
            batch_images = images[i:i + batch_processing_size]
            if not batch_images:
                logger.warning(f"Batch {i // batch_processing_size + 1} is empty, skipping")
                continue

            # Create batch COCO JSON
            try:
                temp_json = create_batch_coco_json(batch_images, data, temp_path, i // batch_processing_size, image_dir)
            except Exception as e:
                logger.error(f"Failed to create batch JSON for batch {i // batch_processing_size + 1}: {e}")
                continue

            try:
                logger.info(f"Training batch {i // batch_processing_size + 1}")
                predictor.fit(
                    train_data=temp_json,
                    time_limit=1800,
                    save_path=model_save_path,
                )
                logger.info(
                    f"Training for batch {i // batch_processing_size + 1} complete. Model saved at: {model_save_path}")

            except Exception as e:
                logger.error(f"Failed training batch {i // batch_processing_size + 1}: {str(e)}")
                os.rmdir(model_save_path)
                logger.info(f"Model deleted: {model_save_path}")
            finally:
                if os.path.exists(temp_json):
                    os.remove(temp_json)

    finally:
        copy_and_update_labels(train_data_folder, model_save_path, class_names)

        if os.path.exists(temp_path):
            try:
                os.rmdir(temp_path)
            except OSError:
                logger.warning(f"Could not remove temporary directory {temp_path}: Directory not empty")


def copy_and_update_labels(train_data_folder, model_save_path, class_names):
    labels_path = os.path.join(train_data_folder, "labels.txt")
    model_labels_path = os.path.join(model_save_path, "labels.txt")

    # Check if source exists (optional, since we'll overwrite)
    try:
        if not os.path.exists(labels_path):
            logging.warning(f"{labels_path} does not exist. A new file will be created.")

            # Write new content to source (train_data_folder)
            with open(labels_path, "w") as f:
                f.write("\n".join(class_names))
            logging.info(f"Updated {labels_path} with new class names.")
    except Exception as e:
        logging.error(f"Failed to update {labels_path}: {str(e)}")

    # Copy to destination
    try:
        shutil.copy2(labels_path, model_labels_path)
        logging.info(f"Copied {labels_path} to {model_labels_path}.")
    except Exception as e:
        logging.error(f"Failed to copy to {model_labels_path}: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Object Detection Model')
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

    args = parser.parse_args()

    storage_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    train_data_folder = os.path.join(storage_root, 'data', 'dataset')

    train_detection_model(
        train_data_folder,
        presets=args.presets,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_gpus=args.num_gpus
    )