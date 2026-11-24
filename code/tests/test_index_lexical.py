"""Тесты лексического индекса (Шаг 2): токенизация и build → save → load."""

from __future__ import annotations

from pathlib import Path

import pytest

from semsearch.core.chunk import CodeChunk
from semsearch.core.index_lexical import BM25Index, tokenize


def _chunks() -> list[CodeChunk]:
    return [
        CodeChunk("f1", "def get_status_text(self):\n    return self.status"),
        CodeChunk("f2", "def sum_numbers(items):\n    return sum(items)"),
        CodeChunk("f3", "def parse_xml_document(text):\n    return XMLParser(text)"),
    ]


def test_tokenize_splits_snake_case() -> None:
    assert "status" in tokenize("get_status_text") and "text" in tokenize("get_status_text")


def test_tokenize_splits_camel_case_and_lowercases() -> None:
    assert tokenize("XMLParser") == ["xml", "parser"]
    assert tokenize("getStatusText") == ["get", "status", "text"]


def test_search_finds_relevant_chunk() -> None:
    index = BM25Index(_chunks()).build()
    assert index.search("sum of items", 1) == ["f2"]


def test_empty_query_raises() -> None:
    index = BM25Index(_chunks()).build()
    with pytest.raises(ValueError, match="пустой запрос"):
        index.search("   ")


def test_empty_corpus_raises() -> None:
    with pytest.raises(ValueError, match="пустому корпусу"):
        BM25Index([]).build()


def test_k_larger_than_corpus_returns_all() -> None:
    index = BM25Index(_chunks()).build()
    assert len(index.search("status", 100)) == 3


def test_search_before_build_raises() -> None:
    with pytest.raises(RuntimeError, match="не построен"):
        BM25Index(_chunks()).search("status")


def test_save_load_same_results(tmp_path: Path) -> None:
    index = BM25Index(_chunks()).build()
    before = index.search_many(["status text", "parse xml", "sum items"], 3)
    index.save(tmp_path)
    loaded = BM25Index.load(tmp_path)
    assert loaded.search_many(["status text", "parse xml", "sum items"], 3) == before
