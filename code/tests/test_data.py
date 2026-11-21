"""Тесты модулей подготовки данных (Шаг 1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from semsearch.config import DataConfig, data_dir
from semsearch.data.prepare import (
    DocstringRemovalError,
    check_no_overlap,
    extract_query,
    normalize_code,
    normalize_query,
    read_jsonl,
    remove_docstring,
    word_count,
)
from semsearch.data.translate import DIRECTIONS, CachedTranslator, TranslationCache

CFG = DataConfig()


# --------------------------------------------------------------------------
# Удаление docstring через AST
# --------------------------------------------------------------------------


def test_remove_docstring_simple() -> None:
    code = 'def add(a, b):\n    """Складывает a и b."""\n    return a + b\n'
    result = remove_docstring(code)
    assert result.had_docstring
    assert not result.used_pass
    assert "Складывает" not in result.code
    assert "return a + b" in result.code
    compile(result.code, "<test>", "exec")


def test_remove_docstring_method() -> None:
    code = (
        "class Point:\n"
        "    def norm(self):\n"
        '        """Возвращает длину вектора."""\n'
        "        return (self.x ** 2 + self.y ** 2) ** 0.5\n"
    )
    result = remove_docstring(code)
    assert "Возвращает длину" not in result.code
    assert "class Point:" in result.code
    assert "self.x ** 2" in result.code
    compile(result.code, "<test>", "exec")


def test_remove_docstring_async() -> None:
    code = (
        "async def fetch(url):\n"
        '    """Получает страницу по url."""\n'
        "    async with client.get(url) as response:\n"
        "        return await response.text()\n"
    )
    result = remove_docstring(code)
    assert "Получает страницу" not in result.code
    assert "async with" in result.code
    compile(result.code, "<test>", "exec")


def test_remove_docstring_keeps_decorator() -> None:
    code = '@staticmethod\ndef parse(text):\n    """Парсит текст."""\n    return text.split()\n'
    result = remove_docstring(code)
    assert result.code.startswith("@staticmethod\n")
    assert "Парсит текст" not in result.code
    compile(result.code, "<test>", "exec")


def test_remove_docstring_without_docstring() -> None:
    code = "def add(a, b):\n    return a + b\n"
    result = remove_docstring(code)
    assert not result.had_docstring
    assert result.code == code


def test_remove_docstring_syntax_error() -> None:
    with pytest.raises(DocstringRemovalError):
        remove_docstring("def broken(:\n    pass\n")


def test_remove_docstring_only_docstring_body() -> None:
    code = 'def noop():\n    """Ничего не делает."""\n'
    result = remove_docstring(code)
    assert result.used_pass
    assert "pass" in result.code
    assert "Ничего не делает" not in result.code
    compile(result.code, "<test>", "exec")


def test_remove_docstring_keeps_comments_and_line_numbers() -> None:
    code = (
        "def add(a, b):\n"
        "    # складывает\n"
        '    """Складывает a и b."""\n'
        "    return a + b  # результат\n"
    )
    result = remove_docstring(code)
    assert "# складывает" in result.code
    assert "# результат" in result.code
    assert len(result.code.splitlines()) == len(code.splitlines())


def test_remove_docstring_multiline_keeps_line_numbers() -> None:
    code = (
        "def add(a, b):\n"
        '    """Складывает a и b.\n'
        "    Возвращает сумму.\n"
        '    """\n'
        "    return a + b\n"
    )
    result = remove_docstring(code)
    assert len(result.code.splitlines()) == len(code.splitlines())
    assert "Возвращает сумму" not in result.code
    compile(result.code, "<test>", "exec")


# --------------------------------------------------------------------------
# Нормализация и извлечение запроса
# --------------------------------------------------------------------------


def test_normalize_code() -> None:
    assert normalize_code("def  Foo( a ):\n    return a\n") == "def foo( a ): return a"


def test_normalize_query() -> None:
    assert normalize_query("Get, the user's data!") == "get the user s data"
    assert normalize_query("  A   b  ") == "a b"


def test_extract_query_takes_first_paragraph() -> None:
    docstring = "First line of doc.\n\nSecond paragraph is ignored."
    assert extract_query(docstring, 30) == "First line of doc."


def test_extract_query_cuts_by_word_limit() -> None:
    docstring = " ".join(f"word{i}" for i in range(50))
    query = extract_query(docstring, 10)
    assert word_count(query) == 10
    assert query.startswith("word0 word1")


def test_extract_query_dedents() -> None:
    docstring = "    Returns the value.\n    More text."
    assert extract_query(docstring, 30) == "Returns the value. More text."


# --------------------------------------------------------------------------
# Проверки на подготовленных данных
# --------------------------------------------------------------------------


def _prepared_files() -> dict[str, Path]:
    root = data_dir()
    return {
        name: root / f"{name}.jsonl"
        for name in ("train", "val_pool", "val_queries", "test_pool", "test_queries")
    }


requires_data = pytest.mark.skipif(
    not (data_dir() / "train.jsonl").exists(),
    reason="данные не подготовлены: запустите semsearch.data.prepare",
)


@requires_data
def test_query_not_in_code() -> None:
    files = _prepared_files()
    for name in ("train", "val_pool", "test_pool"):
        for row in read_jsonl(files[name]):
            assert normalize_query(row["query_en"]) not in normalize_code(row["code"]), row["id"]


@requires_data
def test_no_split_overlap() -> None:
    files = _prepared_files()
    train = read_jsonl(files["train"])
    val = read_jsonl(files["val_pool"])
    test = read_jsonl(files["test_pool"])

    def keys(rows: list[dict[str, object]]) -> tuple[set[str], set[str], set[str]]:
        return (
            {str(row["id"]) for row in rows},
            {normalize_code(str(row["code"])) for row in rows},
            {normalize_query(str(row["query_en"])) for row in rows},
        )

    train_keys = keys(train)
    val_keys = keys(val)
    test_keys = keys(test)
    for left, right in ((train_keys, val_keys), (train_keys, test_keys), (val_keys, test_keys)):
        assert not left[0] & right[0]
        assert not left[1] & right[1]
        assert not left[2] & right[2]


@requires_data
def test_queries_are_subset_of_corpus() -> None:
    files = _prepared_files()
    for query_name, pool_name in (("val_queries", "val_pool"), ("test_queries", "test_pool")):
        pool_ids = {row["id"] for row in read_jsonl(files[pool_name])}
        query_ids = {row["id"] for row in read_jsonl(files[query_name])}
        assert query_ids <= pool_ids


@requires_data
def test_check_no_overlap_raises_on_duplicates() -> None:
    from semsearch.data.prepare import Pair

    item = Pair(id="train-0", query="q", code="c", func_name="f", url="u", repo="r")
    other = Pair(id="val-0", query="q", code="c", func_name="f", url="u", repo="r")
    with pytest.raises(AssertionError):
        check_no_overlap([item], [other], [])


MANUAL_QUERIES = Path(__file__).resolve().parent.parent / "data" / "manual_queries.jsonl"


def _manual_rows() -> list[dict[str, str]]:
    """Ручные запросы из файла."""
    lines = MANUAL_QUERIES.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_manual_queries_are_valid() -> None:
    """Ручные запросы: jsonl с полями id и query_ru, запросы непустые."""
    rows = _manual_rows()
    assert rows
    for row in rows:
        assert set(row) == {"id", "query_ru"}
        assert row["id"]
        assert row["query_ru"].strip()
        assert any("а" <= ch.lower() <= "я" or ch in "ёЁ" for ch in row["query_ru"])


@requires_data
def test_manual_queries_reference_test_corpus() -> None:
    """id ручных запросов есть в тест-корпусе."""
    pool_ids = {row["id"] for row in read_jsonl(_prepared_files()["test_pool"])}
    for row in _manual_rows():
        assert row["id"] in pool_ids, row["id"]


# --------------------------------------------------------------------------
# Кэш перевода
# --------------------------------------------------------------------------


class FakeBackend:
    """Фейковый переводчик: считает вызовы модели."""

    def __init__(self) -> None:
        self.calls = 0

    def translate(self, texts: list[str]) -> list[str]:
        """Вернуть верхний регистр и посчитать вызов."""
        self.calls += len(texts)
        return [text.upper() for text in texts]


def test_translation_cache_avoids_model_on_repeat(tmp_path: Path) -> None:
    backend = FakeBackend()
    cache = TranslationCache(tmp_path / "cache.jsonl", "fake-model")
    translator = CachedTranslator(lambda: backend, cache)

    first, from_cache, from_model = translator.translate(["привет", "пока"])
    assert from_model == 2
    assert from_cache == 0
    assert first == ["ПРИВЕТ", "ПОКА"]
    assert backend.calls == 2

    second, from_cache, from_model = translator.translate(["привет", "пока"])
    assert from_model == 0
    assert from_cache == 2
    assert backend.calls == 2
    assert second == first


def test_translation_cache_persisted_on_disk(tmp_path: Path) -> None:
    backend = FakeBackend()
    path = tmp_path / "cache.jsonl"
    translator = CachedTranslator(lambda: backend, TranslationCache(path, "fake-model"))
    translator.translate(["один"])
    saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(saved) == 1
    assert saved[0]["translation"] == "ОДИН"

    resumed = CachedTranslator(lambda: FakeBackend(), TranslationCache(path, "fake-model"))
    translations, from_cache, from_model = resumed.translate(["один"])
    assert from_model == 0
    assert from_cache == 1
    assert translations == ["ОДИН"]


def test_translation_model_not_loaded_when_cache_is_full(tmp_path: Path) -> None:
    """Если всё переведено раньше, модель вообще не создаётся."""
    path = tmp_path / "cache.jsonl"
    CachedTranslator(lambda: FakeBackend(), TranslationCache(path, "fake-model")).translate(["два"])

    def explode() -> FakeBackend:
        raise AssertionError("модель не должна создаваться при полном кэше")

    resumed = CachedTranslator(explode, TranslationCache(path, "fake-model"))
    translations, from_cache, from_model = resumed.translate(["два"])
    assert (from_cache, from_model) == (1, 0)
    assert translations == ["ДВА"]


class TagBackend:
    """Фейковый переводчик: добавляет метку модели к тексту."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.calls = 0

    def translate(self, texts: list[str]) -> list[str]:
        """Вернуть текст с меткой и посчитать вызов."""
        self.calls += len(texts)
        return [f"{self.tag}:{text}" for text in texts]


def test_translation_cache_does_not_mix_models(tmp_path: Path) -> None:
    """Переводы разных моделей не подменяют друг друга даже в одном файле кэша."""
    path = tmp_path / "cache.jsonl"
    opus = CachedTranslator(lambda: TagBackend("opus"), TranslationCache(path, "opus-model"))
    nllb = CachedTranslator(lambda: TagBackend("nllb"), TranslationCache(path, "nllb-model"))

    assert opus.translate(["hello"])[0] == ["opus:hello"]
    assert nllb.translate(["hello"])[0] == ["nllb:hello"]


def test_nllb_cache_id_depends_on_direction() -> None:
    """У NLLB модель одна на оба направления — ключ кэша обязан различать их."""
    assert DIRECTIONS["en2ru"].cache_id == "Helsinki-NLP/opus-mt-en-ru"
    assert DIRECTIONS["nllb-en2ru"].cache_id != DIRECTIONS["nllb-ru2en"].cache_id
