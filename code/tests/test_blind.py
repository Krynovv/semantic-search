"""Тесты слепого сравнения переводов (Шаг 1)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from semsearch.data.blind import make_blind_file, score_blind, sign_test_p


def _write_csv(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    """Записать csv в utf-8-sig (как это делает blind.py)."""
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Прочитать csv в utf-8-sig."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_make_blind_file_swaps_variants_and_keeps_key(tmp_path: Path) -> None:
    """В слепом файле переводы переставлены, а ключ восстанавливает соответствие."""
    opus_csv = tmp_path / "opus.csv"
    _write_csv(
        opus_csv,
        ["query_en", "query_ru", "rating"],
        [["q1", "ru1", "3"], ["q2", "ru2", ""]],
    )
    nllb_jsonl = tmp_path / "nllb.jsonl"
    nllb_jsonl.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in [
                {"query_en": "q1", "query_ru": "nllb1"},
                {"query_en": "q2", "query_ru": "nllb2"},
            ]
        ),
        encoding="utf-8",
    )
    out_csv = tmp_path / "blind.csv"
    key_csv = tmp_path / "key.csv"

    assert make_blind_file(opus_csv, nllb_jsonl, out_csv, key_csv, seed=42) == 2

    blind = _read_csv(out_csv)
    key = {row["id"]: row["a_is"] for row in _read_csv(key_csv)}
    assert [row["id"] for row in blind] == ["1", "2"]
    expected = {"1": ("ru1", "nllb1"), "2": ("ru2", "nllb2")}
    for row in blind:
        opus_ru, nllb_ru = expected[row["id"]]
        assert row["variant_a"] == (opus_ru if key[row["id"]] == "opus" else nllb_ru)
        assert row["variant_b"] == (nllb_ru if key[row["id"]] == "opus" else opus_ru)
        assert row["rating_a"] == ""
        assert row["rating_b"] == ""


def test_make_blind_file_is_reproducible(tmp_path: Path) -> None:
    """Один и тот же seed — тот же ключ соответствия."""
    opus_csv = tmp_path / "opus.csv"
    rows = [[f"q{i}", f"ru{i}", ""] for i in range(10)]
    _write_csv(opus_csv, ["query_en", "query_ru", "rating"], rows)
    nllb_jsonl = tmp_path / "nllb.jsonl"
    nllb_jsonl.write_text(
        "\n".join(
            json.dumps({"query_en": f"q{i}", "query_ru": f"nllb{i}"}) for i in range(10)
        ),
        encoding="utf-8",
    )
    keys = []
    for name in ("a", "b"):
        key_csv = tmp_path / f"key_{name}.csv"
        make_blind_file(opus_csv, nllb_jsonl, tmp_path / f"blind_{name}.csv", key_csv, seed=42)
        keys.append(_read_csv(key_csv))
    assert keys[0] == keys[1]


def test_make_blind_file_requires_translation_for_every_query(tmp_path: Path) -> None:
    """Нет перевода NLLB для запроса — ошибка, а не пустая ячейка."""
    opus_csv = tmp_path / "opus.csv"
    _write_csv(opus_csv, ["query_en", "query_ru", "rating"], [["q1", "ru1", ""]])
    nllb_jsonl = tmp_path / "nllb.jsonl"
    nllb_jsonl.write_text(json.dumps({"query_en": "other", "query_ru": "nllb"}), encoding="utf-8")
    with pytest.raises(KeyError):
        make_blind_file(opus_csv, nllb_jsonl, tmp_path / "blind.csv", tmp_path / "key.csv")


def test_score_blind_on_manual_example(tmp_path: Path) -> None:
    """Сведение оценок на ручном примере (4 строки, ответ известен заранее)."""
    key_csv = tmp_path / "key.csv"
    _write_csv(
        key_csv,
        ["id", "a_is"],
        [["1", "opus"], ["2", "nllb"], ["3", "opus"], ["4", "nllb"]],
    )
    blind_csv = tmp_path / "blind.csv"
    _write_csv(
        blind_csv,
        ["id", "query_en", "variant_a", "variant_b", "rating_a", "rating_b"],
        [
            ["1", "q1", "a", "b", "2", "4"],
            ["2", "q2", "a", "b", "5", "3"],
            ["3", "q3", "a", "b", "3", "3"],
            ["4", "q4", "a", "b", "5", "2"],
        ],
    )

    summary = score_blind(blind_csv, key_csv)

    assert summary["rows"] == 4
    assert summary["opus_mean"] == pytest.approx((2 + 3 + 3 + 2) / 4)
    assert summary["nllb_mean"] == pytest.approx((4 + 5 + 3 + 5) / 4)
    assert summary["opus_distribution"] == {1: 0, 2: 2, 3: 2, 4: 0, 5: 0}
    assert summary["nllb_distribution"] == {1: 0, 2: 0, 3: 1, 4: 1, 5: 2}
    assert (summary["nllb_better"], summary["nllb_worse"], summary["equal"]) == (3, 0, 1)
    assert summary["p_value"] == pytest.approx(0.25)


def test_score_blind_requires_filled_ratings(tmp_path: Path) -> None:
    """Незаполненная оценка — ошибка, а не молчаливое «0»."""
    key_csv = tmp_path / "key.csv"
    _write_csv(key_csv, ["id", "a_is"], [["1", "opus"]])
    blind_csv = tmp_path / "blind.csv"
    _write_csv(
        blind_csv,
        ["id", "query_en", "variant_a", "variant_b", "rating_a", "rating_b"],
        [["1", "q1", "a", "b", "", ""]],
    )
    with pytest.raises(ValueError):
        score_blind(blind_csv, key_csv)


def test_sign_test_p() -> None:
    """Точный тест знаков: симметричен, на равных долях даёт 1.0."""
    assert sign_test_p(3, 0) == pytest.approx(0.25)
    assert sign_test_p(0, 3) == pytest.approx(0.25)
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(5, 5) == pytest.approx(1.0)
