import sys
import os
import json
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
        Конвертирует все JSON файлы в указанной папке upload_dir (без поиска Detection)
        
        :param upload_dir: Путь к папке с JSON файлами
        :param output_csv_dir: Папка для сохранения CSV файлов
        """
        os.makedirs(output_csv_dir, exist_ok=True)
        
        # Ищем JSON-файлы прямо в upload_dir, а не в подпапках Detection
        json_files = list(Path(upload_dir).glob('*.json'))
        logger.info(f"Найдено JSON файлов: {len(json_files)}")
        
        for json_file in json_files:
            try:
                self._convert_single_json(json_file, output_csv_dir)
                logger.info(f"Успешно обработан файл: {json_file}")
            except Exception as e:
                logger.error(f"Ошибка при обработке файла {json_file}: {str(e)}")
    
    def _convert_single_json(self, json_path: Path, output_dir: str):
        """
        Конвертирует один JSON файл в CSV с нужным форматом
        
        :param json_path: Путь к JSON файлу
        :param output_dir: Папка для сохранения CSV
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Определяем тип JSON структуры и преобразуем к общему формату
        if isinstance(data, dict):
            # Формат, где ключ - имя изображения, значение - список объектов
            csv_data = []
            for image_name, objects in data.items():
                for obj in objects:
                    if not isinstance(obj, dict):
                        continue
                    csv_row = self._create_csv_row(obj, image_name)
                    if csv_row:
                        csv_data.append(csv_row)
        elif isinstance(data, list):
            # Формат, где каждый элемент содержит image и objects
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
            raise ValueError(f"Неизвестный формат JSON в файле {json_path}")
        
        if not csv_data:
            logger.warning(f"Нет данных для конвертации в файле {json_path}")
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
        Создает строку CSV из объекта аннотации
        
        :param obj: Словарь с данными объекта
        :param image_name: Имя изображения
        :return: Словарь с данными для CSV строки
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
            logger.warning(f"Ошибка преобразования данных объекта: {obj}. Ошибка: {str(e)}")
            return None

def convert_all_jsons_to_csv(upload_dir: str, output_csv_dir: str):
    """
    Основная функция для конвертации всех JSON во всех папках Detection
    
    :param upload_dir: Путь к папке upload (содержащей различные подпапки)
    :param output_csv_dir: Папка для сохранения CSV файлов
    """
    converter = JsonToCsvConverter()
    converter.convert(upload_dir, output_csv_dir)
    logging.info(f"Все CSV файлы сохранены в {output_csv_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python script.py <путь_к_папке_upload> [output_csv_dir]")
        sys.exit(1)
    
    upload_dir = sys.argv[1]
    output_csv_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(upload_dir, "output_csv_dir")
    
    convert_all_jsons_to_csv(upload_dir, output_csv_dir)