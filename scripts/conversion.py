import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import json
import glob
from PIL import Image
import random

def convert_to_coco(data_folder, output_folder, train_split=0.8):
    # Ensure output directories exist
    image_dir = os.path.join(output_folder, 'images')
    annotations_dir = os.path.join(output_folder, 'Annotations')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    # Read label properties
    label_properties_path = os.path.join(data_folder, 'Label_Properties', 'label_properties.json')
    if not os.path.exists(label_properties_path):
        raise FileNotFoundError(f"Label properties file not found at {label_properties_path}")
    
    with open(label_properties_path, 'r') as f:
        label_properties = json.load(f)
    
    # Create category mapping
    categories = []
    label_to_id = {}
    for idx, label in enumerate(label_properties.keys(), 1):
        categories.append({
            'id': idx,
            'name': label,
            'supercategory': 'cell'
        })
        label_to_id[label] = idx

    # Write labels.txt
    labels_path = os.path.join(output_folder, 'labels.txt')
    with open(labels_path, 'w') as f:
        for label in label_properties.keys():
            f.write(f"{label}\n")

    # Collect all annotations
    all_annotations = []
    all_images = []
    image_id = 1
    annotation_id = 1
    image_paths = []

    # Process each asset folder
    for asset_folder in glob.glob(os.path.join(data_folder, '*')):
        if not os.path.isdir(asset_folder) or os.path.basename(asset_folder) == 'Label_Properties':
            continue

        detection_dir = os.path.join(asset_folder, 'Detection')
        raw_dir = os.path.join(asset_folder, 'Raw')

        if not os.path.exists(detection_dir):
            print(f"Skipping {asset_folder}: Not a detection asset (missing Detection folder)")
            continue
        if not os.path.exists(raw_dir):
            print(f"Skipping {asset_folder}: Missing Raw folder")
            continue

        # Process each JSON file in Detection
        for json_path in glob.glob(os.path.join(detection_dir, '*.json')):
            # Remove '_bbox' from JSON filename to match image name
            json_base = os.path.splitext(os.path.basename(json_path))[0]
            if json_base.endswith('_bbox'):
                image_base = json_base[:-len('_bbox')]
            else:
                image_base = json_base
            image_name = image_base + '.webp'
            image_path = os.path.join(raw_dir, image_name)

            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue

            # Convert WebP to PNG
            png_image_name = image_base + '.png'
            output_image_path = os.path.join(image_dir, png_image_name)
            try:
                with Image.open(image_path) as img:
                    img.convert('RGB').save(output_image_path, 'PNG')
                print(f"Converted {image_name} to {png_image_name}")
            except Exception as e:
                print(f"Failed to convert {image_name}: {str(e)}")
                continue

            # Read JSON annotations
            try:
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                annotations = json_data.get(image_base, [])
                if not annotations:
                    print(f"No annotations found in {json_path}")
                    continue
            except Exception as e:
                print(f"Error reading {json_path}: {str(e)}")
                continue

            # Add image info
            try:
                with Image.open(output_image_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Error opening {output_image_path}: {str(e)}")
                continue

            image_info = {
                'id': image_id,
                'file_name': f"{output_folder}/images/{png_image_name}",
                'width': width,
                'height': height
            }
            all_images.append(image_info)

            # Process annotations
            for ann in annotations:
                if ann.get('label_name') not in label_to_id:
                    print(f"Unknown label {ann.get('label_name')} in {json_path}")
                    continue
                all_annotations.append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': label_to_id[ann['label_name']],
                    'bbox': [
                        float(ann['bbox_x']),
                        float(ann['bbox_y']),
                        float(ann['bbox_width']),
                        float(ann['bbox_height'])
                    ],
                    'area': float(ann['bbox_width']) * float(ann['bbox_height']),
                    'iscrowd': 0
                })
                annotation_id += 1

            print(f"Processed {image_name} with {len(annotations)} annotations")
            image_id += 1
            image_paths.append((image_info, annotations))

    if not all_images:
        raise ValueError("No images or annotations processed. Check input folder structure.")

    # Split into train and validation
    random.shuffle(image_paths)
    split_idx = int(len(image_paths) * train_split)
    train_images = [x[0] for x in image_paths[:split_idx]]
    val_images = [x[0] for x in image_paths[split_idx:]]
    
    train_image_ids = {img['id'] for img in train_images}
    train_annotations = [ann for ann in all_annotations if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in all_annotations if ann['image_id'] not in train_image_ids]

    print(f"Train set: {len(train_images)} images, {len(train_annotations)} annotations")
    print(f"Val set: {len(val_images)} images, {len(val_annotations)} annotations")

    # Create COCO JSON structure
    coco_train = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': categories
    }
    coco_val = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': categories
    }

    # Save COCO JSON files
    with open(os.path.join(annotations_dir, 'train.json'), 'w') as f:
        json.dump(coco_train, f, indent=2)
    with open(os.path.join(annotations_dir, 'val.json'), 'w') as f:
        json.dump(coco_val, f, indent=2)
    print(f"Saved train.json and val.json in {annotations_dir}")

if __name__ == '__main__':
    storage_root: str = os.path.abspath((os.path.join(os.path.dirname(__file__), '../')))
    data_folder = os.path.join(storage_root,'data', 'uploads')
    output_folder = os.path.join(storage_root, 'data', 'dataset')

    convert_to_coco(data_folder, output_folder)