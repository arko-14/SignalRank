from __future__ import annotations

from typing import Optional

import numpy as np

from . import config
from .embeddings import EmbeddingBackend, cosine_similarity

try:
    import hnswlib
except ImportError:
    hnswlib = None


class HNSWIndex:
    def __init__(self, embedding_backend: EmbeddingBackend) -> None:
        if hnswlib is None:
            raise RuntimeError("hnswlib is required to satisfy AGENTS.md HNSW requirements.")
        self.embedding_backend = embedding_backend
        self.document_vectors: list[list[float]] = []
        self.index: Optional[object] = None

    def fit(self, documents: list[str]) -> None:
        self.document_vectors = self.embedding_backend.embed(documents)
        if not self.document_vectors:
            self.index = None
            return

        dimension = len(self.document_vectors[0])
        index = hnswlib.Index(space=config.HNSW_SPACE, dim=dimension)
        index.init_index(
            max_elements=len(self.document_vectors),
            ef_construction=config.HNSW_EF_CONSTRUCTION,
            M=config.HNSW_M,
        )
        index.add_items(np.array(self.document_vectors, dtype=np.float32), np.arange(len(self.document_vectors)))
        index.set_ef(max(config.HNSW_EF_SEARCH, len(self.document_vectors)))
        self.index = index

    def score(self, query_text: str) -> list[float]:
        if not self.document_vectors:
            return []

        query_vector = self.embedding_backend.embed([query_text])[0]
        if self.index is None:
            return [cosine_similarity(query_vector, document_vector) for document_vector in self.document_vectors]

        labels, distances = self.index.knn_query(
            np.array([query_vector], dtype=np.float32),
            k=len(self.document_vectors),
        )
        scores = [0.0] * len(self.document_vectors)
        for label, distance in zip(labels[0], distances[0]):
            cosine = 1.0 - float(distance)
            scores[int(label)] = max(0.0, min(1.0, (cosine + 1.0) / 2.0))
        return scores
