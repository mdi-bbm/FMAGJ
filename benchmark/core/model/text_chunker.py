from typing import runtime_checkable, Protocol, Sequence, Any, Iterable

import torch
from transformers import PreTrainedTokenizerBase, BatchEncoding

from benchmark.core.model.base_model import PydanticFrozen


@runtime_checkable
class Tokenizer(Protocol):
    def ids(self, text: str) -> list[int]: ...
    def text(self, ids: Sequence[int]) -> str: ...


class HuggingFaceTokenizerAdapter:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    def ids(self, text: str) -> list[int]:
        return self._tokenizer(text, add_special_tokens=False)["input_ids"]

    def text(self, ids: Sequence[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=True)


class Splitter(Protocol):
    def split(self, seq: Sequence[Any], *, window: int, step: int) -> Iterable[Sequence[Any]]: ...


class TorchWindowSplitter(Splitter):
    def split(self, seq: Sequence[Any], *, window: int, step: int) -> Iterable[Sequence[Any]]:
        if not seq:
            return []
        tensor = torch.as_tensor(seq)
        if window >= tensor.size(0):
            yield seq
            return
        for i in range(0, tensor.size(0) - window + 1, step):
            yield tensor[i : i + window].tolist()


class BatchEncoder(Protocol):
    def encode(
        self,
        texts: list[str],
        *,
        max_length: int,
        padding: bool | str,
        return_tensors: str | None,
    ) -> BatchEncoding | dict[str, Any]: ...


class HuggingFaceEncoderAdapter:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    def encode(
        self,
        texts: list[str],
        *,
        max_length: int,
        padding: bool | str,
        return_tensors: str | None,
    ) -> BatchEncoding | dict[str, Any]:
        return self._tokenizer.batch_encode_plus(
            texts,
            truncation=True,
            max_length=max_length,
            padding=padding,
            return_tensors=return_tensors,
        )


class ChunkConfig(PydanticFrozen):
    max_length: int
    step: int | None = None
    return_tensors: str | None = "pt"
    padding: bool | str = True


class TextChunker:
    def __init__(
        self,
        tokenizer: Tokenizer,
        splitter: Splitter,
        encoder: BatchEncoder,
        config: ChunkConfig,
    ) -> None:
        self._tokenizer = tokenizer
        self._splitter = splitter
        self._encoder = encoder
        self._config = config

    def chunk(self, text: str) -> list[dict[str, torch.Tensor]]:
        step = self._config.step or self._config.max_length
        ids = self._tokenizer.ids(text)
        id_chunks = list(
            self._splitter.split(ids, window=self._config.max_length, step=step)
        )
        txt_chunks = [self._tokenizer.text(ids) for ids in id_chunks]
        batch = self._encoder.encode(
            txt_chunks,
            max_length=self._config.max_length,
            padding=self._config.padding,
            return_tensors=self._config.return_tensors,
        )
        return [
            {
                k: (v[i].unsqueeze(0) if self._config.return_tensors else v[i])
                for k, v in batch.items()
            }
            for i in range(len(txt_chunks))
        ]


def create_default_chunker(
    hugging_face_tokenizer: PreTrainedTokenizerBase, *, config: ChunkConfig
) -> TextChunker:
    tokenizer_adapter = HuggingFaceTokenizerAdapter(hugging_face_tokenizer)
    return TextChunker(
        tokenizer=tokenizer_adapter,
        splitter=TorchWindowSplitter(),
        encoder=HuggingFaceEncoderAdapter(hugging_face_tokenizer),
        config=config,
    )
