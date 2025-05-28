from abc import ABC, abstractmethod

import numpy as np
import torch

from numpy.typing import NDArray
from transformers import CLIPModel, CLIPProcessor

from benchmark.core.model.text_chunker import TextChunker


class TextEmbedderBase(ABC):
    @abstractmethod
    def embed(self, text: str) -> NDArray:
        pass

    def __call__(self, text: str) -> NDArray:
        return self.embed(text=text)


class TextEmbedderMaxLengthCLIP(TextEmbedderBase):
    """CLIP-based text embedder that transparently chunks long strings."""

    def __init__(
        self,
        model: CLIPModel,
        processor: CLIPProcessor,
        chunker: TextChunker,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        self._model = model.to(device)
        self._processor = processor
        self._chunker = chunker
        self._device = device
        self._model.eval()

    @torch.no_grad()
    def embed(self, text: str) -> np.ndarray:
        chunks = self._chunker.chunk(text)
        if not chunks:
            return np.empty((0,))

        vectors: list[torch.Tensor] = []
        for chunk in chunks:
            chunk = {k: v.to(self._device) for k, v in chunk.items()}
            vectors.append(self._model.get_text_features(**chunk).cpu())

        pooled = torch.stack(vectors).mean(dim=0)
        return pooled.numpy()
