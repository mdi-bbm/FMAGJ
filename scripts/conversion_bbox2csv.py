import sys
import os
import json
import shutil
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('torch').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class JsonToCsvConverter:
    def convert(self, upload_dir: str, output_csv_dir: str):
        """
        Converts all JSON files in the specified upload_dir (without searching in Detection subfolders)
        
        :param upload_dir: Path to the directory containing JSON files
        :param output_csv_dir: Directory to save CSV files
        """
        os.makedirs(output_csv_dir, exist_ok=True)
        
        # Find JSON files directly in upload_dir, not in Detection subfolders
        json_files = list(Path(upload_dir).glob('*.json'))
        logger.info(f"Found JSON files: {len(json_files)}")
        
        for json_file in json_files:
            try:
                self._convert_single_json(json_file, output_csv_dir)
                # logger.info(f"Successfully processed file: {json_file}")
            except Exception as e:
                logger.error(f"Error processing file {json_file}: {str(e)}")
    
    def _convert_single_json(self, json_path: Path, output_dir: str):
        """
        Converts a single JSON file to CSV with required format
        
        :param json_path: Path to JSON file
        :param output_dir: Directory to save CSV
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Determine JSON structure type and convert to common format
        if isinstance(data, dict):
            # Format where key is image name, value is object list
            csv_data = []
            for image_name, objects in data.items():
                for obj in objects:
                    if not isinstance(obj, dict):
                        continue
                    csv_row = self._create_csv_row(obj, image_name)
                    if csv_row:
                        csv_data.append(csv_row)
        elif isinstance(data, list):
            # Format where each item contains image and objects
            csv_data = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                image_name = item.get('image', '')
                objects = item.get('objects', [])
                for obj in objects:
                    if not isinstance(obj, dict):
                        continue
                    csv_row = self._create_csv_row(obj, image_name)
                    if csv_row:
                        csv_data.append(csv_row)
        else:
            raise ValueError(f"Unknown JSON format in file {json_path}")
        
        if not csv_data:
            logger.warning(f"No data to convert in file {json_path}")
            return
        
        df = pd.DataFrame(csv_data)
        output_filename = f"{json_path.stem}.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        columns_order = [
            'label_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height',
            'image_name', 'image_width', 'image_height', 'confidence'
        ]
        df[columns_order].to_csv(output_path, index=False, float_format='%.6f')
    
    def _create_csv_row(self, obj: Dict[str, Any], image_name: str) -> Dict[str, Any]:
        """
        Creates a CSV row from annotation object
        
        :param obj: Dictionary with object data
        :param image_name: Image name
        :return: Dictionary with data for CSV row
        """
        try:
            return {
                'label_name': obj.get('label_name', obj.get('class', '')),
                'bbox_x': float(obj.get('bbox_x', obj.get('x', 0))),
                'bbox_y': float(obj.get('bbox_y', obj.get('y', 0))),
                'bbox_width': float(obj.get('bbox_width', obj.get('w', 0))),
                'bbox_height': float(obj.get('bbox_height', obj.get('h', 0))),
                'image_name': obj.get('image_name', image_name),
                'image_width': int(obj.get('image_width', 0)),
                'image_height': int(obj.get('image_height', 0)),
                'confidence': float(obj.get('confidence', 1.0))
            }
        except (ValueError, TypeError) as e:
            logger.warning(f"Error converting object data: {obj}. Error: {str(e)}")
            return None
        
def remove_bbox_from_filename(ground_truth_csv_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for filename in os.listdir(ground_truth_csv_dir):
        if filename.endswith(".csv"):
            new_filename = filename.replace("_bbox", "")
            
            source_path = os.path.join(ground_truth_csv_dir, filename)
            target_path = os.path.join(target_dir, new_filename)
            
            shutil.copy2(source_path, target_path)
            # print(f"Copied: {filename} -> {new_filename}")
    logging.info(f"Removed _bbox from filenames in {ground_truth_csv_dir}")
    shutil.rmtree(ground_truth_csv_dir)
    logging.info(f"Removed {ground_truth_csv_dir}")

def convert_all_jsons_to_csv(ground_truth_json_dir: str, output_csv_dir: str):
    """
    Main function to convert all JSONs in all directories
    
    :param ground_truth_json_dir: Path to ground truth json directory (containing various subdirectories)
    :param output_csv_dir: Directory to save CSV files
    """
    converter = JsonToCsvConverter()
    converter.convert(ground_truth_json_dir, output_csv_dir)
    logging.info(f"All CSV files saved to {output_csv_dir}")

    remove_bbox_from_filename(output_csv_dir, target_dir)

    shutil.rmtree(ground_truth_json_dir)
    logging.info(f"Removed {ground_truth_json_dir}")

if __name__ == "__main__":
    storage_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    benchmark_directory = os.path.join(storage_root, "benchmark_data/object_detections")
    target_dir = "/home/dstu601/isolated/isolated/benchmark_data/object_detections/ground_truth"
    
    ground_truth_json_dir = os.path.join(benchmark_directory, "input_jsons_ground_truth")
    output_csv_dir = os.path.join(benchmark_directory, "ground_truth_csv_dir")
    
    convert_all_jsons_to_csv(ground_truth_json_dir, output_csv_dir)