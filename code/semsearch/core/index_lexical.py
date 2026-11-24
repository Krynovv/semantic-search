"""Лексический поиск по коду: BM25 с разбиением идентификаторов (Шаг 2).

Токенизация решает главную проблему BM25 на коде: `get_status_text` и
`status_text` — разные токены для наивного сплита по пробелам. Поэтому
идентификатор делится по `_` и по границе строчных/заглавных (`XMLParser` →
`xml`, `parser`), всё приводится к нижнему регистру.

Индекс сохраняется в каталог: jsonl документов + .npy «сырая» матрица
счётчиков токенов в словаре. Сохранение считается по документам, так что
повторная сборка из этих файлов даёт ту же выдачу.

Собственно BM25-оценки — из библиотеки rank_bm25 (BM25Okapi, k1=1.5, b=0.75).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from semsearch.core.chunk import CodeChunk

_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_WORD = re.compile(r"\w+")

CHUNKS_FILE = "chunks.jsonl"
COUNTS_FILE = "counts.npy"
VOCAB_FILE = "vocab.json"


def tokenize(text: str) -> list[str]:
    """Токены кода: snake_case и camelCase разбиваются, регистр снят."""
    tokens: list[str] = []
    for word in _WORD.findall(text):
        for part in word.split("_"):
            tokens.extend(match.group(0).lower() for match in _CAMEL.finditer(part))
    return tokens


def check_query(query: str) -> str:
    """Пустой запрос — ValueError (граничный случай из задания)."""
    if not query.strip():
        raise ValueError("пустой запрос: передайте непустую строку")
    return query


def _count_matrix(corpus: Sequence[Sequence[str]]) -> tuple[list[str], np.ndarray]:
    """Словарь токенов и матрица счётчиков (документы × словарь)."""
    vocab = {token for doc in corpus for token in doc}
    index = {token: i for i, token in enumerate(sorted(vocab))}
    counts = np.zeros((len(corpus), len(index)), dtype=np.int32)
    for row, doc in enumerate(corpus):
        for token in doc:
            counts[row, index[token]] += 1
    return sorted(vocab), counts


def _from_counts(vocab: Sequence[str], counts: np.ndarray, k1: float, b: float) -> BM25Okapi:
    """Восстановить BM25Okapi из матрицы счётчиков (корпус развёрнут из словаря)."""
    corpus = [[vocab[i] for i in np.nonzero(row)[0] for _ in range(int(row[i]))] for row in counts]
    return BM25Okapi(corpus, k1=k1, b=b)


@dataclass
class BM25Index:
    """BM25-индекс над списком фрагментов кода."""

    chunks: list[CodeChunk]
    k1: float = 1.5
    b: float = 0.75
    _bm25: BM25Okapi | None = None

    def build(self) -> BM25Index:
        """Построить индекс по токенам кода."""
        if not self.chunks:
            raise ValueError("нельзя построить BM25-индекс по пустому корпусу")
        corpus = [tokenize(chunk.code) for chunk in self.chunks]
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        return self

    def search(self, query: str, k: int = 10) -> list[str]:
        """id первых k документов по убыванию BM25-оценки."""
        check_query(query)
        if self._bm25 is None:
            raise RuntimeError("индекс не построен: вызовите build() или load()")
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(-scores, kind="stable")[:k]
        return [self.chunks[i].id for i in order]

    def search_many(self, queries: Sequence[str], k: int = 10) -> list[list[str]]:
        """Выдачи для нескольких запросов."""
        return [self.search(query, k) for query in queries]

    def save(self, directory: Path) -> None:
        """Сохранить документы, словарь и матрицу счётчиков токенов."""
        if not self.chunks:
            raise RuntimeError("пустой индекс: сохранять нечего")
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / CHUNKS_FILE).open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps({"id": chunk.id, "code": chunk.code}, ensure_ascii=False))
                handle.write("\n")
        vocab, counts = _count_matrix([tokenize(chunk.code) for chunk in self.chunks])
        (directory / VOCAB_FILE).write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
        np.save(directory / COUNTS_FILE, counts)

    @classmethod
    def load(cls, directory: Path, k1: float = 1.5, b: float = 0.75) -> BM25Index:
        """Загрузить индекс без пересчёта BM25 из сохранённых счётчиков."""
        with (directory / CHUNKS_FILE).open(encoding="utf-8") as handle:
            chunks = [CodeChunk(row["id"], row["code"]) for row in map(json.loads, handle)]
        vocab = json.loads((directory / VOCAB_FILE).read_text(encoding="utf-8"))
        counts = np.load(directory / COUNTS_FILE)
        index = cls(chunks, k1=k1, b=b)
        index._bm25 = _from_counts(vocab, counts, k1, b)
        return index
