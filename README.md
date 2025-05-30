# Fluorescence Microscopic Astrocyte Gap Junction Images Dataset
This repository contains the following folders, which are described below:
  - [Datasets of Fluorescence Microscopic Astrocyte Gap Junction Images](#datasets-structure-and-metadata) 
  - [Training Detection Models on Datasets](#training-and-test-scripts)
  - [Benchmarks of Trained Detection Models](#benchmark)

# Datasets of Fluorescence Microscopic Astrocyte Gap Junction Images

## Overview
The `data` folder contains datasets of human gap junctions (`HUMAN_gap_junctions`) and rat gap junctions (`RAT_gap_junctions`).  
This datasets includes organized folders and metadata related to biological or medical images.

## Root Dataset Structure

The root folder contains the following elements:

- **Label_Properties/**  
  Contains the label properties file that defines the annotation classes.

- **Metadata_Static/**  
  Contains static metadata: dataset parameters, device types, and scaling values.

- **Assets/**  
  Folders for each asset containing raw images, annotations, and metadata.

## Asset Folder Structure

Each asset folder contains:

- **Raw/**  
  Raw images for this asset in WEBP format.

- **Detection/**  
  JSON files with bounding box annotations.

- **asset_metadata.json**  
  JSON file with dynamic metadata about the biological or medical properties of the asset.

## Root Metadata and Structures

- **Label_Properties/label_properties.json**  
  Defines the labels/classes used for annotations in the dataset.

- **Metadata_Static/metadata_static.json**  
  Information about device types involved in data collection and device scaling for measurement normalization.

## JSON Metadata Parameters Description

### asset_metadata.json

Dynamic metadata for each asset. Example:

```json
{
  "species": "rat",
  "age": 18,
  "weight": 0.15,
  "sex": "Male",
  "localization": "brain_cortex",
  "diagnosis": "healthy"
}
```

- **species** — Species name (e.g., "rat").  
- **age** — Age of the subject/object (in months).
- **weight** — Weight (in kilograms).
- **sex** — Biological sex ("Male", "Female", "Undefined").
- **diagnosis** — Diagnosis related to the subject.
- **localization** — Anatomical localization of the pathology.
- **Note**: All fields may be null if data is missing; JSON includes only available parameters.

### label_properties.json

Defines the annotation categories/classes. Example:

```json
{
  
  "gap_junction": "#4EF8FF"
  
}
```  
- **gap_junction** — Class name.
- **#d6f176** — Hexadecimal color code for visualization.


### metadata_static.json

Information about device types involved in data collection and device scaling for measurement normalization. Example:

```json
{
  "device_type": "Olympus UPlanXApo",
  "scaling_value": "60x"
}
```  

# Training Detection Models on Datasets 
This repo contains research code to run Fluorescence Microscopic Astrocyte Gap Junction Images Dataset training and inference.

## Rules and Guidelines

### Train recommendations 
  - train. For best results, use at least 15 images for training
  - inference recommended to process no more than 1000 images at once

Hardware Recommendations: NVIDIA RTX 4090

### Project Structure

- **scripts**
   - conversion.py — prepares images and markup files from platform format to dino format
   - train.py — model training
   - inference.py — inference + preparation of results to benchmark format
   - train_pipeline.py — run train and inference scripts alternately


## Setting Up a Virtual Environment

### Windows

1. Open Command Prompt or PowerShell.
2. Navigate to the repository directory.
3. Create a virtual environment:
```shell
python -m venv venv
```
4. Activate the virtual environment:
```shell
venv/Scripts/activate
```

### Linux

1. Open a terminal.
2. Navigate to the repository directory.
3. Create a virtual environment:
```shell
python3 -m venv venv
```
4. Activate the virtual environment:
```shell
source venv/bin/activate
```

### Installing Packages
With the virtual environment activated, install the necessary packages using:

```shell
pip install -r requirements.txt
```

## Code launch

### Standard startup without parameters
```shell
python3 scripts/train_pipeline.py
```

### Run with select presets
```shell
python3 scripts/train_pipeline.py --presets best_quality
```
or
```shell
python3 scripts/train_pipeline.py --presets medium_quality
```

### Startup with customization of various parameters
```shell
python3 scripts/train_pipeline.py --presets high_quality --max_epochs 5 --batch_size 8
```

### Run training only (no inference)
```shell
python3 scripts/train_pipeline.py --presets best_quality --skip_inference
```

### Run inference only (no training)
```shell
python3 scripts/train_pipeline.py --skip_training
```

### Full customization
```shell
python3 scripts/train_pipeline.py --presets best_quality --max_epochs 10 --batch_size 8 --num_workers 4 --num_gpus 2
```

### Presets
Preset Models Quality
    - high_quality: Uses DINO-Resnet50 preset model
    - best_quality: Uses DINO-SwinL preset model


# Benchmarks of Trained Detection Models

Tools to calculate benchmarks, save and analyze results.

See `scripts` module for examples.

## Using MLFlow

This project uses MLFlow for tracking experiments, logging metrics, and saving parameters. The following instructions will guide you through how to set up and view your experiment results.

### 1. Setting Up the Environment
Make sure that you have MLflow installed in your Python environment. You can install it using pip:

```shell
pip install mlflow
```

### 2. Running the Experiment
Each benchmark script automatically logs metrics, the model version
and the current Git commit hash to MLflow.

#### 2.1 Default parameters 
You can run the script with default or custom directories for ground truth and predicted data.
For example:

```shell
python scripts/calculate_object_count_metrics.py
```

By default, the script looks for data in the following directories:
- Ground Truth: `benchmark/data_examples/object_counts/ground_truth`
- Predictions: `benchmark/data_examples/object_counts/predictions`

If you want to specify custom directories, use the command line arguments provided via `argparse`.

### 3. Viewing Results in MLFlow
If you omit the `--mlflow_dir` parameter, the default MLflow folder 
`mlruns` is created in the repository root.
To view the results of your experiment, launch the MLFlow UI with the following command:

```shell
mlflow ui
```

Once the server starts, open your browser and navigate to:

http://127.0.0.1:5000

Here, you can explore the logged metrics (such as mean IoU), parameters (like the first 5 characters of the Git commit hash), and other experiment-related data.

### 4. Experiment Output
- Git Commit Hash: The current Git commit hash is automatically logged as a parameter 
in MLFlow for version tracking.
- IoU Metric (for semantic segmentation case): 
The calculated Intersection over Union (IoU) is logged as the main evaluation metric.

### 5. Example Output
After running the script and opening the MLFlow UI, you will be able to see the logged metrics like this:

| Experiment | Git Commit | mean IoU |
|------------|-------------|----------|
| Exp1       | abcde       | 0.75     |
| Exp2       | fghij       | 0.80     |


