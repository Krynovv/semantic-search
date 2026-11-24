"""Тесты прогона оценки (Шаг 2): наборы запросов согласованы с корпусом."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from semsearch.eval.run_eval import prepare_queries


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.fixture
def fake_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Мини-датасет: 3 функции, 2 EN-запроса, 2 RU-перевода, 2 обратных."""
    pool = _write(
        tmp_path / "test_pool.jsonl",
        [{"id": f"test-{i}", "code": "def f(): pass"} for i in range(3)],
    )
    monkeypatch.setattr("semsearch.eval.run_eval.data_dir", lambda: tmp_path)
    assert pool.exists()
    return {
        "en": _write(
            tmp_path / "test_queries.jsonl",
            [{"id": "test-0", "query_en": "a"}, {"id": "test-1", "query_en": "b"}],
        ),
        "ru": _write(
            tmp_path / "ru.jsonl",
            [{"id": "test-0", "query_ru": "а"}, {"id": "test-1", "query_ru": "б"}],
        ),
        "back": _write(
            tmp_path / "back.jsonl",
            [{"id": "test-0", "query_en_back": "A"}, {"id": "test-1", "query_en_back": "B"}],
        ),
    }


def test_variants_share_the_same_relevant_ids(fake_data: dict[str, Path]) -> None:
    sets = prepare_queries(None, fake_data["ru"], fake_data["back"])
    assert sets["en"][0] == sets["ru_translated"][0] == sets["ru_translated+mt_en"][0]


def test_partial_ru_keeps_en_on_same_ids(fake_data: dict[str, Path], tmp_path: Path) -> None:
    # RU-перевод неполный: EN урезается до покрытых id, RU-вариант считается целиком
    partial = _write(tmp_path / "ru_partial.jsonl", [{"id": "test-1", "query_ru": "б"}])
    sets = prepare_queries(None, partial, fake_data["back"])
    assert sets["en"][0] == ["test-1"]
    assert sets["ru_translated"] == (["test-1"], ["б"])
    # обратный перевод не покрывает id test-1, поэтому вариант не считается


def test_unknown_id_raises(fake_data: dict[str, Path]) -> None:
    # EN-запросы заменены на id, которого нет в корпусе
    broken = {"id": "nope", "query_en": "a"}
    fake_data["en"].write_text(json.dumps(broken) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="test_pool"):
        prepare_queries(None, fake_data["ru"], fake_data["back"])
