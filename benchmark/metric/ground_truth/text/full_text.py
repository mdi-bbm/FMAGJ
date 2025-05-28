import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity

from benchmark.metric.ground_truth.base import GroundTruthMetric
from benchmark.core.model.text_embedder import TextEmbedderBase


class FullTextCosineSimilarityScore(GroundTruthMetric):
    def calculate(self, prediction: NDArray, ground_truth: NDArray) -> float:
        """
        Calculate the cosine similarity between the embeddings of a predicted text and a ground truth text.

        This metric is used to evaluate the semantic similarity between two pieces of text by comparing
        their embeddings. Higher values indicate greater similarity, with a maximum of 1.0 (completely
        identical direction) and a minimum of -1.0 (completely opposite direction).

        :param prediction: Embedding vector of the predicted text.
            - Format: 1D numpy array with the embedding of the predicted text.
            - Example: np.array([0.2, 0.8, ..., 0.5])

        :param ground_truth: Embedding vector of the ground truth text.
            - Format: 1D numpy array with the embedding of the ground truth text.
            - Example: np.array([0.3, 0.7, ..., 0.6])

        :return: Cosine similarity score as a float.
            - Range: [-1.0, 1.0]
            - A score of 1.0 indicates identical embedding directions, meaning high semantic similarity.
            - A score of 0 indicates no similarity in direction.
            - A score of -1.0 indicates completely opposite directions.

        Method:
            - Reshapes the embeddings into 2D arrays to be compatible with scikit-learn's `cosine_similarity`.
            - Uses `cosine_similarity` from `scikit-learn` to calculate the similarity between the vectors.

        Example Usage:
            >>> similarity_metric = FullTextCosineSimilarityScore()
            >>> prediction_embedding = np.array([0.2, 0.8, ..., 0.5])
            >>> ground_truth_embedding = np.array([0.3, 0.7, ..., 0.6])
            >>> similarity_score = similarity_metric.calculate(prediction_embedding, ground_truth_embedding)
            >>> print(f"Cosine Similarity Score: {similarity_score}")
        """
        prediction = np.asarray(prediction).reshape(1, -1)
        ground_truth = np.asarray(ground_truth).reshape(1, -1)
        score = cosine_similarity(prediction, ground_truth)[0, 0]
        return score


class FullTextEmbedderScore(GroundTruthMetric):
    def __init__(self, embedder: TextEmbedderBase):
        """
        Initialize the FullTextEmbedderScore with a provided embedder model.

        :param embedder: An embedding model or function that generates a vector representation
                         of a text input. The embedder should have a callable interface,
                         such as embedder(text: str) -> NDArray.

                         Example:
                         - SentenceTransformer model from Hugging Face's Transformers library.
                         - Custom embedding function that returns a text embedding.
        """
        self.embedder = embedder

    def calculate(self, prediction: str, ground_truth: str) -> float:
        """
        Calculate the semantic similarity between the embeddings of predicted and ground truth texts.

        This method first generates embeddings for both the predicted and ground truth texts
        using the provided embedder model. Then, it calculates the cosine similarity score
        between the two embeddings to assess the semantic similarity.

        :param prediction: The predicted text as a string.
            - Example: "The cat is on the mat."

        :param ground_truth: The ground truth text as a string.
            - Example: "A cat sits on the mat."

        :return: Cosine similarity score as a float.
            - Range: [-1.0, 1.0]
            - A score of 1.0 indicates identical embedding directions, meaning high semantic similarity.
            - A score of 0 indicates no similarity in direction.
            - A score of -1.0 indicates completely opposite directions.

        Method:
            - Calls the embedder on `prediction` and `ground_truth` to generate their embeddings.
            - Passes the generated embeddings to `FullTextCosineSimilarityScore` to compute cosine similarity.
        """
        prediction_embedding = self.embedder(prediction)
        ground_truth_embedding = self.embedder(ground_truth)
        score = FullTextCosineSimilarityScore().calculate(
            prediction=prediction_embedding,
            ground_truth=ground_truth_embedding
        )
        return score
