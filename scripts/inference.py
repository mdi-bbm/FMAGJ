import os
import sys
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('torch').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

from autogluon.multimodal import MultiModalPredictor
import glob
import json
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from benchmark.conversion.objects_single_json_to_csv import JsonToCsvConverter


storage_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

def run_predict(data_folder: str) -> list:
    logging.info("Starting detection inference...")
        
    try:
        # Validate input folder
        image_files = glob.glob(f"{data_folder}/*.png")
        if not image_files:
            raise FileNotFoundError(f"No PNG images found in {data_folder}")
            
        return process_predictions(data_folder, image_files)
            
    except Exception as e:
        logging.error(f"Inference failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Inference failed: {str(e)}") from e

def process_predictions(data_folder:str, image_files: list) -> list:
    result_paths = []
    model_folder = os.path.join(storage_root, 'models')
    predictor = None

    try:
        # Create model folder if it doesn't exist
        if not os.path.exists(model_folder):
            os.makedirs(model_folder, exist_ok=True)
            logger.info(f"Created model folder at: {model_folder}")

        # Get all version folders
        model_versions = glob.glob(os.path.join(model_folder, "model*-*"))

        if not model_versions:
            logger.info(f"No models found for {model_folder}.")
            return model_folder

        # Return the most recent version path
        model_folder = max(model_versions, key=os.path.getctime)

        if not os.path.exists(model_folder):
            raise FileNotFoundError(f"No valid model found in {model_folder}")
        if not os.path.exists(os.path.join(model_folder, "assets.json")):
            raise FileNotFoundError(f"No valid model found in {model_folder} - missing assets.json")

        # Ensure labels.txt exists
        labels_path = os.path.join(model_folder, "labels.txt")
        if not os.path.exists(labels_path):
            logging.warning(f"labels.txt not found in model directory at {labels_path}")
            with open(labels_path, 'w') as f:
                f.write("gap_junction\nclass2\nclass3\n")  # Replace with your classes
            logging.info(f"Created default labels.txt at {labels_path}")

        # Log paths for debugging
        logger.info(f"Model folder: {model_folder}")
        logger.info(f"Labels path: {labels_path}")
        logger.info(f"Data folder: {data_folder}")
        logger.info(f"Annotations directory exists: {os.path.exists(os.path.join(data_folder, 'Annotations'))}")

        # Initialize predictor without VOC dependency
        predictor = MultiModalPredictor(
            problem_type="object_detection",
            label=labels_path,
            sample_data_path=None  # Avoid VOC format checks
        ).load(model_folder)

        if predictor is not None:
            predictor.set_num_gpus(1)
        
        # Prepare results container
        glob_json_str = "{"

        # Process images
        for image_path in image_files:
            try:
                # Create temporary prediction data
                data = {
                    "images": [{
                        "id": 0, 
                        "width": -1, 
                        "height": -1, 
                        "file_name": image_path
                    }],
                    "categories": []
                }

                temp_json = os.path.join(data_folder, os.path.basename(image_path) + ".json")
                with open(temp_json, "w") as f:
                    json.dump(data, f)

                # Run prediction
                res = predictor.predict(temp_json, save_results=True)
                
                # Process results
                result_path = process_combined_results(res)
                glob_json_str += result_path
                glob_json_str += "," if image_files[-1] != image_path else "}"
                result_paths.append(result_path)

            except Exception as e:
                logging.error(f"Failed to process image {image_path}: {str(e)}", exc_info=True)
                continue
            finally:
                # Clean up temp file
                if os.path.exists(temp_json):
                    os.remove(temp_json)
        
        # Save combined results
        output_json_path = os.path.join(data_folder, "output.json")
        with open(output_json_path, "w") as f:
            f.write(glob_json_str)

        convert_result_to_benchmark(output_json_path)

        # After processing all images, convert the latest result.txt to benchmark CSV
        try:
            base_path = os.path.join(storage_root, "AutogluonModels")
            latest_txt_folder = get_latest_result_txt(base_path)
            result_txt_path = os.path.join(latest_txt_folder, "result.txt")
            
            # if os.path.exists(output_json_path):
            #     convert_result_to_benchmark(output_json_path)
            # else:
            #     logging.warning("No output.json found to convert to benchmark format")
        except Exception as e:
            logging.error(f"Failed to convert result.txt to benchmark format: {str(e)}", exc_info=True)
        
        return result_paths
        
    except Exception as e:
        logging.error(f"Prediction processing failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Prediction processing failed: {str(e)}") from e
    finally:
        # Clean up predictor resources
        if predictor is not None:
            try:
                if hasattr(predictor, 'cleanup'):
                    predictor.cleanup()
                del predictor
            except Exception as e:
                logging.warning(f"Error during predictor cleanup: {str(e)}", exc_info=True)

def convert_result_to_benchmark(output_json_path: str):
    json_path = output_json_path
    output_csv_dir = os.path.join(storage_root, "benchmark_data", "object_detections", "predictions")
    # forced_label = os.path.join(storage_root, "benchmark_data", "object_detections", "forced_filenames.txt")

    # JsonToCsvConverter(forced_label=forced_label).convert(
    JsonToCsvConverter().convert(
        json_path=json_path,
        output_csv_dir=output_csv_dir
    )
    
    logging.info(f"Saved benchmark predictions to {output_csv_dir}")

@staticmethod
def process_combined_results(self) -> str:
    base_path = os.path.join(storage_root, "AutogluonModels")
    latest_txt_file = get_latest_result_txt(base_path)
    result_txt_path = os.path.join(latest_txt_file, "result.txt")

    if os.path.exists(result_txt_path):
        results_json = process_single_result(result_txt_path)
        results_json_str = json.dumps(results_json)
        results_json_str = results_json_str[1:-1]
    else:
        raise FileNotFoundError("result.txt not found in the latest model folder.")
    return results_json_str

@staticmethod
def get_latest_result_txt(base_folder: str) -> str:
    base_path = base_folder
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"AutogluonModels directory not found at: {base_path}")
    
    txt_dirs = [d for d in os.listdir(base_path) 
            if os.path.isdir(os.path.join(base_path, d))]
    
    if not txt_dirs:
        raise FileNotFoundError("No result.txt found after prediction")
    
    latest_dir = max(txt_dirs, key=lambda d: os.path.getmtime(os.path.join(base_path, d)))
    
    return os.path.join(base_path, latest_dir)

@staticmethod
def process_single_result(txt_path: str)-> str:
    with open(txt_path, "r") as file:
        lines = file.readlines()

    header = lines[0].strip()
    rows = lines[1:]

    results = []
    for i in range(len(rows)):
        image_path, bboxes_str = rows[i].split(",", 1)
        image = cv2.imread(image_path)
        width, height = image.shape[1], image.shape[0]
        image_name = os.path.basename(image_path.strip())

        bboxes = eval(bboxes_str.strip())
        bboxes = "{ '" + image_name +"': " + bboxes + '}'
        bboxes = bboxes.replace("'", '"')
        bboxes = json.loads(bboxes)

        detections = []
        for bbox in bboxes:
            for elems in bboxes[bbox]: 
                if elems["score"] > 0.35:
                    detections.append({
                        "label_name": elems["class"],
                        "bbox_x": elems["bbox"][0],
                        "bbox_y": elems["bbox"][1],
                        "bbox_width": elems["bbox"][2] - elems["bbox"][0],
                        "bbox_height": elems["bbox"][3] - elems["bbox"][1],
                        "image_name": image_name,
                        "image_width": width,
                        "image_height": height,
                    })

        results.append({
            image_name[:-4]: detections
        })
    return results[0]

if __name__ == "__main__":
    # storage_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    # data_folder = os.path.join(storage_root, 'data', 'for_inference')
    run_predict(data_folder)