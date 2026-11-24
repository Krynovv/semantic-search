"""Тесты векторного индекса (Шаг 2).

Настоящая модель e5 (~1 ГБ) в тесты не загружается: подменяется `Embedder`,
который считает эмбеддинг из слов текста. Проверяются механика FAISS,
префиксы, граничные случаи и build → save → load.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from semsearch.config import EvalConfig
from semsearch.core.chunk import CodeChunk
from semsearch.core.index_dense import DenseIndex

VOCAB = ["status", "text", "sum", "items", "parse", "xml", "def", "return"]
FILES = ("embeddings.npy", "ids.json")


class FakeEmbedder:
    """Эмбеддинг — мешок слов (сколько раз слово встречается), L2-нормированный."""

    cfg = EvalConfig()

    def encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        assert prefix in (self.cfg.query_prefix, self.cfg.passage_prefix)
        rows = []
        for text in texts:
            words = text.lower().replace("_", " ").split()
            rows.append([float(words.count(word)) for word in VOCAB])
        vectors = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1.0, norms)


def _chunks() -> list[CodeChunk]:
    return [
        CodeChunk("f1", "def get: return status text"),
        CodeChunk("f2", "def sum_items: return sum items"),
        CodeChunk("f3", "def parse: return xml parse"),
    ]


def _index() -> DenseIndex:
    return DenseIndex(FakeEmbedder()).build(_chunks())  # type: ignore[arg-type]


def test_search_finds_relevant_chunk() -> None:
    assert _index().search("sum items", 1) == ["f2"]


def test_empty_query_raises() -> None:
    with pytest.raises(ValueError, match="пустой запрос"):
        _index().search("", 1)


def test_empty_corpus_raises() -> None:
    with pytest.raises(ValueError, match="пустому корпусу"):
        DenseIndex(FakeEmbedder()).build([])  # type: ignore[arg-type]


def test_k_larger_than_corpus_returns_all() -> None:
    assert sorted(_index().search("status", 100)) == ["f1", "f2", "f3"]


def test_search_before_build_raises() -> None:
    with pytest.raises(RuntimeError, match="не построен"):
        DenseIndex(FakeEmbedder()).search("status")  # type: ignore[arg-type]


def test_search_many_matches_single_search() -> None:
    index = _index()
    queries = ["status text", "parse xml", "sum items"]
    assert index.search_many(queries, 3) == [index.search(q, 3) for q in queries]


def test_save_load_same_results(tmp_path: Path) -> None:
    index = _index()
    queries = ["status text", "parse xml", "sum items"]
    before = index.search_many(queries, 3)
    index.save(tmp_path)
    loaded = DenseIndex.load(tmp_path, FakeEmbedder())  # type: ignore[arg-type]
    assert loaded.search_many(queries, 3) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(FILES)
