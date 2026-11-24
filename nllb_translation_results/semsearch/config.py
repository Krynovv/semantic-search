"""Пути и параметры (dataclass-конфиги) пакета semsearch.

Корень данных, кэша и моделей задаётся переменной окружения SEMSEARCH_HOME
(по умолчанию <корень кода>/var). Каталог кэша можно вынести отдельно
переменной SEMSEARCH_CACHE_DIR (например, на Google Drive в Colab).
Все пути строятся через pathlib.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
"""Корень кода: kursovaya-tex/code."""


def semsearch_home() -> Path:
    """Корень для данных, кэша и моделей (SEMSEARCH_HOME или <code>/var)."""
    env = os.environ.get("SEMSEARCH_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return CODE_ROOT / "var"


def data_dir() -> Path:
    """Каталог с подготовленными датасетами (jsonl)."""
    return semsearch_home() / "data"


def cache_dir() -> Path:
    """Каталог для дискового кэша переводов (SEMSEARCH_CACHE_DIR или <home>/cache)."""
    env = os.environ.get("SEMSEARCH_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return semsearch_home() / "cache"


def results_dir() -> Path:
    """Каталог с результатами курсовой (kursovaya-tex/results)."""
    return CODE_ROOT.parent / "results"


@dataclass(frozen=True)
class DataConfig:
    """Параметры подготовки данных (Шаг 1)."""

    train_size: int = 100_000
    val_queries: int = 2_000
    val_pool_size: int = 5_000
    test_pool_size: int = 5_000
    test_queries: int = 1_000
    min_docstring_words: int = 3
    max_code_tokens: int = 512
    max_query_words: int = 30
    tokenizer_name: str = "intfloat/multilingual-e5-base"


@dataclass(frozen=True)
class TranslateConfig:
    """Параметры машинного перевода запросов (EN->RU и RU->EN)."""

    en_ru_model: str = "Helsinki-NLP/opus-mt-en-ru"
    ru_en_model: str = "Helsinki-NLP/opus-mt-ru-en"
    batch_size: int = 16
    max_input_tokens: int = 256
    max_new_tokens: int = 256
    num_beams: int = 4
    dtype: str = "float32"
