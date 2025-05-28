import argparse
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

import torch
from transformers import CLIPProcessor, CLIPModel

from benchmark.dataset.image_captioning import (
    ImageCaptioningGroundTruthDataset,
    ImageCaptioningGroundTruthDatasetConfig
)
from benchmark.evaluator.factory import create_mean_metrics_evaluator
from benchmark.task.image_captioning import ImageCaptioningTask
from benchmark.core.model.text_chunker import create_default_chunker, ChunkConfig
from benchmark.core.model.text_embedder import TextEmbedderMaxLengthCLIP


def _build_clip_embedder(device: str | torch.device = "cpu") -> TextEmbedderMaxLengthCLIP:
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    chunker = create_default_chunker(
        hugging_face_tokenizer=processor.tokenizer,
        config=ChunkConfig(max_length=processor.tokenizer.model_max_length),
    )
    return TextEmbedderMaxLengthCLIP(
        model=model,
        processor=processor,
        chunker=chunker,
        device=device,
    )


EmbedderFactory = Callable[[str | torch.device], TextEmbedderMaxLengthCLIP]

_TESTING_EMBEDDERS: Mapping[str, EmbedderFactory] = MappingProxyType({
    "CLIP": _build_clip_embedder,
})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run benchmark for image interpretation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_result_path",
        type=Path,
        default=Path(__file__).parents[1] / "data_examples" / "texts" / "model_result.csv",
        help="CSV with model predictions and ground truth.",
    )
    parser.add_argument(
        "--prediction_column",
        type=str,
        default="finetuned",
        help="Column with predicted strings.",
    )
    parser.add_argument(
        "--ground_truth_column",
        type=str,
        default="ground_truth",
        help="Column with ground‑truth strings.",
    )
    parser.add_argument(
        "--embedder",
        choices=list(_TESTING_EMBEDDERS.keys()),
        default="CLIP",
        help="Which embedder factory to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for the embedder (e.g. 'cuda:0').",
    )
    parser.add_argument(
        "--mlflow_dir",
        type=Path,
        default=Path(__file__).parents[2] / "mlruns",
        help="Path to MLFlow tracking directory.",
    )
    parser.add_argument("--model_version", required=False, type=str)
    parser.add_argument("--dataset_version", required=False, type=str)
    args = parser.parse_args()

    embedder_factory: EmbedderFactory = _TESTING_EMBEDDERS[args.embedder]
    embedder = embedder_factory(args.device)

    task = ImageCaptioningTask(embedder=embedder)

    dataset_config = ImageCaptioningGroundTruthDatasetConfig(
        path_csv=args.model_result_path,
        prediction_column=args.prediction_column,
        ground_truth_column=args.ground_truth_column,
    )
    dataset = ImageCaptioningGroundTruthDataset(config=dataset_config)

    evaluator = create_mean_metrics_evaluator(
        task=task,
        dataset=dataset,
        mlflow_dir=args.mlflow_dir,
        dataset_version=args.dataset_version,
        model_version=args.model_version,
    )

    evaluator.evaluate(dataset=dataset)


if __name__ == "__main__":
    main()
