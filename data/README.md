# Datasets Structure and Metadata

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
- **gap_junction** — Class name.
- **#d6f176** — Hexadecimal color code for visualization.
