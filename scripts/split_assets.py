import os
import shutil
import random
from pathlib import Path
from PIL import Image

def convert_to_png_and_move(json_file, raw_dir, output_dir):
    """Convert corresponding image to PNG and move to output directory"""
    # Get corresponding image filename
    img_name = json_file.name.replace('_bbox.json', '.webp')
    img_path = raw_dir / img_name
    
    if not img_path.exists():
        print(f"Warning: Image {img_name} not found for {json_file.name}")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert and save as PNG
    png_filename = img_name.replace('.webp', '.png')
    output_path = output_dir / png_filename
    
    try:
        with Image.open(img_path) as img:
            img.save(output_path, 'PNG')
        print(f"Converted and saved: {output_path}")
    except Exception as e:
        print(f"Error converting {img_path}: {str(e)}")

def split_data(source_dir, uploads_dir, benchmark_dir):
    # Convert string paths to Path objects
    source_dir = Path(source_dir)
    uploads_dir = Path(uploads_dir)
    benchmark_dir = Path(benchmark_dir)
    
    # Create target directories if they don't exist
    uploads_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    
    # Directory for converted PNGs
    inference_dir = source_dir.parent / "for_inference"
    inference_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all subdirectories in the source directory
    for subdir in source_dir.iterdir():
        if not subdir.is_dir():
            continue
            
        print(f"Processing {subdir.name}...")
        
        # Paths to Detection and Raw inside the subdirectory
        detection_dir = subdir / "Detection"
        raw_dir = subdir / "Raw"
        
        if not detection_dir.exists() or not raw_dir.exists():
            print(f"Skipping {subdir.name} - no Detection or Raw folders")
            continue
        
        # Get all JSON files with _bbox.json suffix
        json_files = list(detection_dir.glob("*_bbox.json"))
        if not json_files:
            print(f"No JSON files found in {detection_dir}")
            continue
            
        # Shuffle files for random selection
        random.shuffle(json_files)
        
        # Calculate number of files for benchmark (20%)
        benchmark_count = max(1, int(len(json_files) * 0.2))
        
        # Split files
        benchmark_files = json_files[:benchmark_count]
        train_files = json_files[benchmark_count:]
        
        # Create directory structure in uploads
        uploads_subdir = uploads_dir / subdir.name
        uploads_detection = uploads_subdir / "Detection"
        uploads_raw = uploads_subdir / "Raw"
        
        uploads_detection.mkdir(parents=True, exist_ok=True)
        uploads_raw.mkdir(parents=True, exist_ok=True)
        
        # Copy train files (80%) to uploads
        for json_file in train_files:
            # Copy JSON
            shutil.copy2(json_file, uploads_detection)
            
            # Find and copy corresponding image
            img_name = json_file.name.replace('_bbox.json', '.webp')
            img_path = raw_dir / img_name
            if img_path.exists():
                shutil.copy2(img_path, uploads_raw)
            else:
                print(f"Warning: Image {img_name} not found for {json_file.name}")
        
        # Process benchmark files (20%)
        for json_file in benchmark_files:
            # Copy JSON to benchmark directory
            shutil.copy2(json_file, benchmark_dir / json_file.name)
            
            # Convert corresponding image to PNG and move to inference directory
            convert_to_png_and_move(json_file, raw_dir, inference_dir)
        
        print(f"  Copied {len(train_files)} files to train, {len(benchmark_files)} to benchmark")

if __name__ == "__main__":
    storage_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    
    source_directory = os.path.join(storage_root, 'data', "RAT__gap_junctions")
    uploads_directory = os.path.join(storage_root, 'data', "uploads")
    benchmark_directory = os.path.join(storage_root, "benchmark_data/object_detections/input_jsons_ground_truth")
    
    split_data(source_directory, uploads_directory, benchmark_directory)
    print("Done!")