"""Слепое сравнение переводов двух моделей (Шаг 1).

`make` собирает слепой файл для оценки человеком: в каждой строке два перевода
(opus-mt и NLLB) без указания, какой из них какой, и отдельный ключ
соответствия — его открывают только после того, как оценки проставлены.

`score` сводит заполненные оценки: средние баллы, распределение 1–5, число
строк «лучше/хуже/равно» и p-значение точного парного теста знаков.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from semsearch.data.translate import read_jsonl

BLIND_COLUMNS = ["id", "query_en", "variant_a", "variant_b", "rating_a", "rating_b"]
KEY_COLUMNS = ["id", "a_is"]
OPUS = "opus"
NLLB = "nllb"


def read_rows(path: Path) -> list[dict[str, str]]:
    """Прочитать csv (utf-8-sig — так файл открывается в Excel/LibreOffice)."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Записать csv в utf-8-sig."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def make_blind_file(
    opus_csv: Path,
    nllb_jsonl: Path,
    out_csv: Path,
    key_csv: Path,
    seed: int = 42,
) -> int:
    """Собрать слепой файл оценки и ключ соответствия.

    opus_csv — csv с колонками query_en/query_ru (оценки из него не берутся);
    nllb_jsonl — jsonl с полями query_en и query_ru (вывод translate.py).
    """
    opus = read_rows(opus_csv)
    nllb = {str(row["query_en"]): str(row["query_ru"]) for row in read_jsonl(nllb_jsonl)}
    rng = random.Random(seed)
    blind_rows: list[list[Any]] = []
    key_rows: list[list[Any]] = []
    for index, row in enumerate(opus, start=1):
        query_en = row["query_en"]
        if query_en not in nllb:
            raise KeyError(f"нет перевода NLLB для запроса: {query_en}")
        a_is_opus = rng.random() < 0.5
        opus_ru, nllb_ru = row["query_ru"], nllb[query_en]
        variant_a, variant_b = (opus_ru, nllb_ru) if a_is_opus else (nllb_ru, opus_ru)
        blind_rows.append([index, query_en, variant_a, variant_b, "", ""])
        key_rows.append([index, OPUS if a_is_opus else NLLB])
    write_rows(out_csv, BLIND_COLUMNS, blind_rows)
    write_rows(key_csv, KEY_COLUMNS, key_rows)
    return len(blind_rows)


def sign_test_p(better: int, worse: int) -> float:
    """Двусторонний точный тест знаков (точный биномиальный).

    scipy в окружении нет, поэтому сумма по биномиальному распределению
    считается вручную: P(|X - n/2| >= |k - n/2|).
    """
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2**n)
    return min(1.0, 2 * tail)


def _rating(value: str, row_id: str) -> int:
    """Разобрать оценку из ячейки (1–5)."""
    try:
        rating = int(str(value).strip())
    except ValueError as error:
        raise ValueError(f"строка {row_id}: оценка не заполнена или не число: {value!r}") from error
    if not 1 <= rating <= 5:
        raise ValueError(f"строка {row_id}: оценка вне шкалы 1–5: {rating}")
    return rating


def score_blind(blind_csv: Path, key_csv: Path) -> dict[str, Any]:
    """Свести заполненные оценки: средние, распределение, парный тест знаков."""
    key = {row["id"]: row["a_is"] for row in read_rows(key_csv)}
    ratings: dict[str, list[int]] = {OPUS: [], NLLB: []}
    better = worse = equal = 0
    for row in read_rows(blind_csv):
        row_id = row["id"]
        if row_id not in key:
            raise KeyError(f"нет ключа для строки {row_id}")
        rating_a = _rating(row["rating_a"], row_id)
        rating_b = _rating(row["rating_b"], row_id)
        if key[row_id] == OPUS:
            opus_rating, nllb_rating = rating_a, rating_b
        else:
            opus_rating, nllb_rating = rating_b, rating_a
        ratings[OPUS].append(opus_rating)
        ratings[NLLB].append(nllb_rating)
        if nllb_rating > opus_rating:
            better += 1
        elif nllb_rating < opus_rating:
            worse += 1
        else:
            equal += 1
    count = len(ratings[OPUS])
    return {
        "rows": count,
        "opus_mean": sum(ratings[OPUS]) / count if count else 0.0,
        "nllb_mean": sum(ratings[NLLB]) / count if count else 0.0,
        "opus_distribution": {score: ratings[OPUS].count(score) for score in range(1, 6)},
        "nllb_distribution": {score: ratings[NLLB].count(score) for score in range(1, 6)},
        "nllb_better": better,
        "nllb_worse": worse,
        "equal": equal,
        "p_value": sign_test_p(better, worse),
    }


def format_report(summary: dict[str, Any]) -> str:
    """Текстовый отчёт по сведению оценок."""
    lines = [
        f"Оценено строк: {summary['rows']}",
        f"opus-mt: средняя {summary['opus_mean']:.2f}, "
        f"распределение {summary['opus_distribution']}",
        f"NLLB-600M: средняя {summary['nllb_mean']:.2f}, "
        f"распределение {summary['nllb_distribution']}",
        f"NLLB лучше: {summary['nllb_better']}, хуже: {summary['nllb_worse']}, "
        f"равно: {summary['equal']}",
        f"p (точный тест знаков, двусторонний): {summary['p_value']:.4f}",
    ]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """Парсер CLI: make / score."""
    parser = argparse.ArgumentParser(description="Слепое сравнение переводов (Шаг 1)")
    commands = parser.add_subparsers(dest="command", required=True)

    make_cmd = commands.add_parser("make", help="собрать слепой файл и ключ")
    make_cmd.add_argument("--opus-csv", type=Path, required=True, help="csv переводов opus-mt")
    make_cmd.add_argument("--nllb-jsonl", type=Path, required=True, help="jsonl переводов NLLB")
    make_cmd.add_argument("--out-csv", type=Path, required=True, help="слепой файл для оценки")
    make_cmd.add_argument("--key-csv", type=Path, required=True, help="ключ соответствия")
    make_cmd.add_argument("--seed", type=int, default=42)

    score_cmd = commands.add_parser("score", help="свести заполненные оценки")
    score_cmd.add_argument("--blind-csv", type=Path, required=True, help="заполненный слепой файл")
    score_cmd.add_argument("--key-csv", type=Path, required=True, help="ключ соответствия")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI."""
    args = _build_parser().parse_args(argv)
    if args.command == "make":
        rows = make_blind_file(
            opus_csv=args.opus_csv,
            nllb_jsonl=args.nllb_jsonl,
            out_csv=args.out_csv,
            key_csv=args.key_csv,
            seed=args.seed,
        )
        print(f"Слепой файл: {args.out_csv} ({rows} строк), ключ: {args.key_csv}")
        return 0
    summary = score_blind(blind_csv=args.blind_csv, key_csv=args.key_csv)
    print(format_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
