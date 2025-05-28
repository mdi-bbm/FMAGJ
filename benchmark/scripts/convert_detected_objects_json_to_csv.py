import argparse
from pathlib import Path

from benchmark.conversion.objects_single_json_to_csv import JsonToCsvConverter


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert predictions in single json-file to supported csv files')
    parser.add_argument(
        '--json_path',
        type=Path,
        help='Path to original json file'
    )
    parser.add_argument(
        '--output_csv_dir',
        type=Path,
        help='Path to dataset converted csv files of supported format'
    )
    parser.add_argument(
        '--forced_label',
        type=str,
        required=False,
        help='Forced label to be set up instead of original labels'
    )
    args = parser.parse_args()

    JsonToCsvConverter(forced_label=args.forced_label).convert(
        json_path=args.json_path,
        output_csv_dir=args.output_csv_dir
    )


if __name__ == '__main__':
    main()
