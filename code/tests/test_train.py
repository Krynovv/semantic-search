"""Тесты подготовки данных дообучения (Шаг 3).

Само обучение локально не запускается: на этой машине тренер
transformers/accelerate падает после первого шага (research-log.md,
2026-11-23), обучение идёт в Colab. Здесь проверяется подготовка к нему:
конфиг, сборка пар в формате sentence-transformers и загрузка test-набора.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from semsearch.train.finetune import TrainConfig, load_jsonl, prepare_training_data, to_dataset


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def train_path(tmp_path: Path) -> Path:
    """100 синтетических пар: EN- и RU-запрос → код функции."""
    rows = [
        {
            "id": f"train-{i}",
            "query_en": f"returns status text number {i}",
            "query_ru": f"возвращает текст статуса номер {i}",
            "code": f"def status_text_{i}(self):\n    return 'ok {i}'",
        }
        for i in range(100)
    ]
    return _write(tmp_path / "train_with_ru.jsonl", rows)


def test_config_validates() -> None:
    cfg = TrainConfig(experiment="mix-en")
    assert (cfg.lr, cfg.batch_size, cfg.epochs, cfg.warmup_steps, cfg.seed) == (
        1e-5,
        32,
        3,
        500,
        42,
    )
    with pytest.raises(ValueError, match="experiment"):
        TrainConfig(experiment="en-only")
    with pytest.raises(ValueError, match="epochs"):
        TrainConfig(epochs=0)


def test_prepare_training_data_ru_only(train_path: Path) -> None:
    examples = prepare_training_data(train_path, "ru-only")
    assert len(examples) == 100
    assert examples[0].texts == [
        "возвращает текст статуса номер 0",
        "def status_text_0(self):\n    return 'ok 0'",
    ]


def test_prepare_training_data_mix_en(train_path: Path) -> None:
    examples = prepare_training_data(train_path, "mix-en")
    assert len(examples) == 200
    # пара одного кода: сначала RU-запрос, потом EN-запрос к тому же коду
    assert [example.texts for example in examples[40:44]] == [
        ["возвращает текст статуса номер 20", "def status_text_20(self):\n    return 'ok 20'"],
        ["returns status text number 20", "def status_text_20(self):\n    return 'ok 20'"],
        ["возвращает текст статуса номер 21", "def status_text_21(self):\n    return 'ok 21'"],
        ["returns status text number 21", "def status_text_21(self):\n    return 'ok 21'"],
    ]


def test_to_dataset_prefixes_and_shuffles(train_path: Path) -> None:
    examples = prepare_training_data(train_path, "mix-en")
    data = to_dataset(examples)
    assert sorted(data) == ["anchor", "positive"]
    assert len(data["anchor"]) == len(data["positive"]) == 200
    assert all(text.startswith("query: ") for text in data["anchor"])
    assert all(text.startswith("passage: ") for text in data["positive"])
    # пары целы: query: <запрос i> ↔ passage: <код i>, но порядок не файловый
    numbers = [int(anchor.rsplit(" ", 1)[-1]) for anchor in data["anchor"]]
    assert numbers != sorted(numbers)
    for anchor, positive in zip(data["anchor"], data["positive"], strict=True):
        number = anchor.rsplit(" ", 1)[-1]
        assert positive == f"passage: def status_text_{number}(self):\n    return 'ok {number}'"


def test_prepare_training_data_limit_keeps_pairs(train_path: Path) -> None:
    assert len(prepare_training_data(train_path, "ru-only", limit=10)) == 10
    assert len(prepare_training_data(train_path, "mix-en", limit=10)) == 20
    with pytest.raises(ValueError, match="experiment"):
        prepare_training_data(train_path, "both")


def test_load_jsonl_skips_blank_lines(train_path: Path) -> None:
    with train_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert len(load_jsonl(train_path)) == 100
