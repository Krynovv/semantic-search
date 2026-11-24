"""Документ индекса — фрагмент кода, который ищется по запросу.

Для оценки (Шаг 2) фрагмент — одна функция из корпуса; id совпадает с id записи
датасета, поэтому пары «запрос → правильный ответ» берутся прямо из jsonl.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CodeChunk:
    """Фрагмент кода: идентификатор и текст."""

    id: str
    code: str
