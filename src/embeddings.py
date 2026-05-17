from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.request
from typing import Protocol, Sequence

from . import config


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    cleaned = []
    current = []
    for char in lowered:
        if char.isalnum() or char == "_":
            current.append(char)
        else:
            if current:
                cleaned.append("".join(current))
                current = []
    if current:
        cleaned.append("".join(current))
    return cleaned


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    similarity = dot / (left_norm * right_norm)
    similarity = max(-1.0, min(1.0, similarity))
    return (similarity + 1.0) / 2.0


class EmbeddingBackend(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class HashEmbeddingBackend:
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            tokens = _tokenize(text)
            grams = tokens + [f"{tokens[i]}__{tokens[i + 1]}" for i in range(len(tokens) - 1)]
            for gram in grams:
                digest = hashlib.sha1(gram.encode("utf-8")).hexdigest()
                index = int(digest[:8], 16) % self.dimensions
                sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
                vector[index] += sign
            vectors.append(normalize_vector(vector))
        return vectors


class SentenceTransformerBackend:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            config.LOCAL_EMBEDDING_MODEL,
            local_files_only=config.LOCAL_EMBEDDING_LOCAL_FILES_ONLY,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return [list(map(float, row)) for row in embeddings]


class OpenRouterEmbeddingBackend:
    def __init__(self) -> None:
        if not config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is required when EMBEDDING_PROVIDER=openrouter")

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": config.OPENROUTER_EMBEDDING_MODEL, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            config.OPENROUTER_BASE_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter embedding request failed: {exc}") from exc
        data = parsed.get("data") or []
        return [list(map(float, item.get("embedding") or [])) for item in data]


def build_embedding_backend() -> EmbeddingBackend:
    if config.EMBEDDING_PROVIDER == "openrouter":
        return OpenRouterEmbeddingBackend()
    if config.EMBEDDING_PROVIDER == "local":
        try:
            return SentenceTransformerBackend()
        except Exception:
            return HashEmbeddingBackend()
    raise ValueError(f"Unsupported embedding provider: {config.EMBEDDING_PROVIDER}")
