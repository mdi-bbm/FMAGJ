import csv
import json
from pathlib import Path
from typing import Any

from benchmark.core.model.base_model import PydanticFrozen
from benchmark.dataset.detected_object_info import DetectedObjectInfo


class JsonParser(PydanticFrozen):
    json_content: dict[str, Any]

    def extract_bbox_detection_info(self) -> list[DetectedObjectInfo]:
        """Extracts all bbox data from the JSON content."""
        all_bbox_data = []
        for key in self.json_content:
            if key.startswith('bbox_data_'):
                for item in self.json_content[key]:
                    all_bbox_data.append(DetectedObjectInfo(
                        label_name=item['label_name'],
                        bbox_x=item['bbox_x'],
                        bbox_y=item['bbox_y'],
                        bbox_width=item['bbox_width'],
                        bbox_height=item['bbox_height'],
                        image_name=item['image_name'],
                        image_width=item['image_width'],
                        image_height=item['image_height'],
                        confidence=item.get('confidence', 1.0)
                    ))
        return all_bbox_data


class BoundingBoxGrouper(PydanticFrozen):
    bbox_data: list[DetectedObjectInfo]

    def group_by_image_name(self) -> dict[str, list[DetectedObjectInfo]]:
        """Groups bbox data by image name."""
        images = {}
        for item in self.bbox_data:
            images.setdefault(item.image_name, []).append(item)
        return images


class CsvWriter(PydanticFrozen):
    image_name_detection_info: dict[str, list[DetectedObjectInfo]]
    output_csv_dir: Path
    forced_label: str | None = None

    def write_csv_files(self) -> None:
        """Writes grouped bbox data into CSV files per image."""
        self.output_csv_dir.mkdir(parents=True, exist_ok=True)

        for image_name, items in self.image_name_detection_info.items():
            csv_file_name = f"{Path(image_name).stem}.csv"
            csv_file_path = self.output_csv_dir / csv_file_name

            with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'label_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height',
                    'image_name', 'image_width', 'image_height', 'confidence'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for item in items:
                    writer.writerow({
                        'label_name': self.forced_label if self.forced_label else item.label_name,
                        'bbox_x': item.bbox_x,
                        'bbox_y': item.bbox_y,
                        'bbox_width': item.bbox_width,
                        'bbox_height': item.bbox_height,
                        'image_name': item.image_name,
                        'image_width': item.image_width,
                        'image_height': item.image_height,
                        'confidence': item.confidence
                    })


class JsonToCsvConverter(PydanticFrozen):
    forced_label: str | None = None

    def convert(
        self,
        json_path: Path,
        output_csv_dir: Path
    ) -> None:
        """Converts a JSON file containing bbox data into CSV files per image."""
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        parser = JsonParser(json_content=json_data)
        all_bbox_data = parser.extract_bbox_detection_info()

        grouper = BoundingBoxGrouper(bbox_data=all_bbox_data)
        image_name_detection_info = grouper.group_by_image_name()

        writer = CsvWriter(
            image_name_detection_info=image_name_detection_info,
            output_csv_dir=output_csv_dir,
            forced_label=self.forced_label
        )
        writer.write_csv_files()
