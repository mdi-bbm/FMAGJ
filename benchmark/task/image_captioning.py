from benchmark.metric.ground_truth.text.full_text import FullTextCosineSimilarityScore
from benchmark.metric.ground_truth.text.levenshtein import NormalizedLevenshteinDistance
from benchmark.preprocessing.text import TextBasicPreprocessingOperation, MaxSizeEmbeddingPreprocessingOperation
from benchmark.task.base import BaseBenchmarkTask, METRIC_PREPROCESSING_TYPE
from benchmark.core.model.text_embedder import TextEmbedderBase


class ImageCaptioningTask(BaseBenchmarkTask):
    def __init__(self, embedder: TextEmbedderBase):
        self.embedder = embedder

    @property
    def metric_preprocessing(self) -> METRIC_PREPROCESSING_TYPE:
        return {
            NormalizedLevenshteinDistance(): TextBasicPreprocessingOperation(),
            FullTextCosineSimilarityScore(): MaxSizeEmbeddingPreprocessingOperation(embedder=self.embedder)
        }
