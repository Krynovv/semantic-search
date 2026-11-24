"""Векторный поиск: эмбеддинги multilingual-e5-base + FAISS (Шаг 2).

Модель e5 требует префиксы: `query: ` для запроса и `passage: ` для документа —
без них качество заметно падает (так предписывает карточка модели
intfloat/multilingual-e5-base). Эмбеддинги L2-нормированы, поиск — по
скалярному произведению, равному косинусу (FAISS IndexFlatIP).

Модель загружается один раз в Embedder и переиспользуется для индексирования
корпуса и кодирования запросов.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from semsearch.config import EvalConfig
from semsearch.core.chunk import CodeChunk
from semsearch.core.index_lexical import check_query

IDS_FILE = "ids.json"
EMBEDDINGS_FILE = "embeddings.npy"


class Embedder:
    """Обёртка над SentenceTransformer с префиксами e5."""

    def __init__(self, cfg: EvalConfig, model: SentenceTransformer | None = None) -> None:
        self.cfg = cfg
        # готовая модель — чтобы оценивать дообучаемую прямо в памяти, без перечитывания весов
        if model is None:
            model = SentenceTransformer(cfg.model_name, device=cfg.device)
        self.model = model
        self.model.max_seq_length = cfg.max_seq_length

    def encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        """L2-нормированные эмбеддинги текстов с добавленным префиксом."""
        vectors = self.model.encode(
            [prefix + text for text in texts],
            batch_size=self.cfg.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class DenseIndex:
    """FAISS-индекс над эмбеддингами фрагментов кода."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.ids: list[str] = []
        self.faiss: faiss.IndexFlatIP | None = None

    def build(self, chunks: Sequence[CodeChunk]) -> DenseIndex:
        """Закодировать код фрагментов и положить в FAISS."""
        if not chunks:
            raise ValueError("нельзя построить векторный индекс по пустому корпусу")
        prefix = self.embedder.cfg.passage_prefix
        vectors = self.embedder.encode([chunk.code for chunk in chunks], prefix)
        self.ids = [chunk.id for chunk in chunks]
        self.faiss = faiss.IndexFlatIP(vectors.shape[1])
        self.faiss.add(vectors)
        return self

    def search(self, query: str, k: int = 10) -> list[str]:
        """id первых k документов по косинусной близости."""
        return self.search_many([query], k)[0]

    def search_many(self, queries: Sequence[str], k: int = 10) -> list[list[str]]:
        """Выдачи для нескольких запросов (запросы кодируются одним батчем)."""
        if not queries:
            return []
        for query in queries:
            check_query(query)
        if self.faiss is None:
            raise RuntimeError("индекс не построен: вызовите build() или load()")
        vectors = self.embedder.encode(queries, self.embedder.cfg.query_prefix)
        _, indices = self.faiss.search(vectors, min(k, len(self.ids)))
        return [[self.ids[i] for i in row] for row in indices]

    def save(self, directory: Path) -> None:
        """Сохранить id документов и матрицу эмбеддингов."""
        if self.faiss is None:
            raise RuntimeError("пустой индекс: сохранять нечего")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / IDS_FILE).write_text(
            json.dumps(self.ids, ensure_ascii=False), encoding="utf-8"
        )
        np.save(directory / EMBEDDINGS_FILE, self.faiss.reconstruct_n(0, self.faiss.ntotal))

    @classmethod
    def load(cls, directory: Path, embedder: Embedder) -> DenseIndex:
        """Загрузить эмбеддинги в новый FAISS-индекс без пересчёта."""
        index = cls(embedder)
        index.ids = json.loads((directory / IDS_FILE).read_text(encoding="utf-8"))
        vectors = np.load(directory / EMBEDDINGS_FILE)
        index.faiss = faiss.IndexFlatIP(vectors.shape[1])
        index.faiss.add(vectors)
        return index
