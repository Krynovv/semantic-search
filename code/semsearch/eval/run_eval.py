"""Прогон базовых конфигураций поиска на test-сплите (Шаг 2).

Файл в `results/` (csv, utf-8-sig):
  config,method,language,split,n_queries,corpus_size,MRR@10,Recall@1,Recall@5,Recall@10,nDCG@10

language — язык запроса: `en` (исходный docstring), `ru_translated` (машинный
перевод EN→RU), `ru_manual` (10 ручных запросов). Строка с language
`ru_translated+mt_en` — BM25 по запросу, переведённому EN→RU→EN: для векторной
модели этот вариант не считается.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from semsearch.config import CODE_ROOT, EvalConfig, data_dir, results_dir
from semsearch.core.chunk import CodeChunk
from semsearch.core.index_dense import DenseIndex, Embedder
from semsearch.core.index_lexical import BM25Index
from semsearch.core.metrics import METRICS, evaluate

RESULTS_FILE = "baselines.csv"
COLUMNS = ["config", "method", "language", "split", "n_queries", "corpus_size", *METRICS]
TEST = "test"
RU_MANUAL = "ru_manual"
DENSE_VARIANTS = ("en", "ru_translated", RU_MANUAL)
BM25_VARIANTS = (*DENSE_VARIANTS, "ru_translated+mt_en")
K = 10
"""Глубина выдачи: MRR@10, Recall@10 и nDCG@10 считаются по первым 10."""


class Searcher(Protocol):
    """Минимальный интерфейс индекса для оценки."""

    def search_many(self, queries: Sequence[str], k: int) -> list[list[str]]:
        """Выдачи для списка запросов."""
        ...


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Прочитать jsonl."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_corpus() -> list[CodeChunk]:
    """Корпус кандидатов test: N = 5000 функций."""
    return [CodeChunk(row["id"], row["code"]) for row in load_jsonl(data_dir() / "test_pool.jsonl")]


def prepare_queries(
    limit: int | None,
    ru_path: Path | None = None,
    back_path: Path | None = None,
) -> dict[str, tuple[list[str], list[str]]]:
    """Наборы запросов: имя варианта → (id правильных ответов, тексты запросов).

    RU-варианты берутся из test_queries_ru.jsonl (перевод EN→RU), обратный —
    из test_queries_en_back.jsonl; пути можно переопределить (--ru-queries,
    --back-queries), например на NLLB-переводы из var/data/step2/. Вариант
    считается, только если его id совпадают с EN-набором целиком: иначе методы
    сравнивались бы на разных запросах.
    """
    pool_ids = {row["id"] for row in load_jsonl(data_dir() / "test_pool.jsonl")}
    en_rows = load_jsonl(data_dir() / "test_queries.jsonl")
    ru_rows = load_jsonl(ru_path or data_dir() / "test_queries_ru.jsonl")
    back_rows = load_jsonl(back_path or data_dir() / "test_queries_en_back.jsonl")
    if limit is not None:
        en_rows, ru_rows, back_rows = en_rows[:limit], ru_rows[:limit], back_rows[:limit]
    for name, rows in (("en", en_rows), ("ru", ru_rows), ("back", back_rows)):
        if missing := [row["id"] for row in rows if row["id"] not in pool_ids]:
            raise ValueError(f"{name}: id нет в test_pool.jsonl, например {missing[0]}")

    en_ids = {row["id"] for row in en_rows}
    ru_ids = {row["id"] for row in ru_rows}
    if not ru_ids:
        print("внимание: RU-файл пуст — RU-варианты не считаются")
    elif ru_ids != en_ids:
        # RU-перевод на диске неполный: EN считаем на тех же id, что и RU,
        # иначе методы с русским запросом сравнивались бы с EN на разных наборах
        print(f"внимание: RU-перевод покрывает {len(ru_ids)} из {len(en_ids)} запросов — "
              "EN тоже считается только на покрытых id")
        en_rows = [row for row in en_rows if row["id"] in ru_ids]
        en_ids = ru_ids
    if not en_rows:
        raise ValueError("нет запросов: EN и RU-наборы не пересекаются по id")

    sets: dict[str, tuple[list[str], list[str]]] = {
        "en": ([row["id"] for row in en_rows], [row["query_en"] for row in en_rows]),
    }
    if ru_ids == en_ids:
        sets["ru_translated"] = (
            [row["id"] for row in ru_rows],
            [row["query_ru"] for row in ru_rows],
        )
    if {row["id"] for row in back_rows} == en_ids:
        sets["ru_translated+mt_en"] = (
            [row["id"] for row in back_rows],
            [row["query_en_back"] for row in back_rows],
        )

    manual = [
        row
        for row in load_jsonl(CODE_ROOT / "data" / "manual_queries.jsonl")
        if row["id"] in pool_ids
    ]
    sets[RU_MANUAL] = ([row["id"] for row in manual], [row["query_ru"] for row in manual])
    return sets


def run_config(
    config: str,
    method: str,
    variants: Sequence[str],
    sets: dict[str, tuple[list[str], list[str]]],
    searcher: Searcher,
    corpus_size: int,
) -> list[dict[str, Any]]:
    """Метрики одного индекса по выбранным наборам запросов."""
    rows: list[dict[str, Any]] = []
    for name in variants:
        if name not in sets:
            continue
        relevant, queries = sets[name]
        values = evaluate(searcher.search_many(queries, K), relevant, K)
        language, _, suffix = name.partition("+")
        rows.append(
            {
                "config": f"{config}+{suffix}" if suffix else config,
                "method": method,
                "language": language,
                "split": TEST,
                "n_queries": len(queries),
                "corpus_size": corpus_size,
                **values,
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Записать таблицу метрик (utf-8-sig — открывается в Excel)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row[key]) for key in COLUMNS})


def _fmt(value: Any) -> Any:
    """Метрики — 4 знака после запятой."""
    return f"{value:.4f}" if isinstance(value, float) else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Базовые конфигурации поиска на test-сплите")
    parser.add_argument("--method", choices=["bm25", "dense", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="отладка: первые N запросов")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--ru-queries",
        type=Path,
        default=None,
        help="перевод запросов EN→RU (по умолчанию test_queries_ru.jsonl)",
    )
    parser.add_argument(
        "--back-queries",
        type=Path,
        default=None,
        help="перевод RU→EN (по умолчанию test_queries_en_back.jsonl)",
    )
    parser.add_argument("--output", type=Path, default=results_dir() / RESULTS_FILE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Прогон выбранных методов и запись csv + печать таблицы."""
    args = build_parser().parse_args(argv)
    cfg = EvalConfig(device=args.device, batch_size=args.batch_size)
    sets = prepare_queries(args.limit, args.ru_queries, args.back_queries)
    corpus = load_corpus()

    rows: list[dict[str, Any]] = []
    if args.method in ("bm25", "all"):
        index = BM25Index(corpus, k1=cfg.bm25_k1, b=cfg.bm25_b)
        rows += run_config("bm25", "bm25", BM25_VARIANTS, sets, index.build(), len(corpus))
    if args.method in ("dense", "all"):
        dense = DenseIndex(Embedder(cfg))
        rows += run_config("e5", "dense", DENSE_VARIANTS, sets, dense.build(corpus), len(corpus))

    write_csv(args.output, rows)
    _print(rows)
    print(f"\nзаписано: {args.output}")
    return 0


def _print(rows: Sequence[dict[str, Any]]) -> None:
    print(f"{'config':22} {'method':6} {'N':>5} " + " ".join(f"{m:>9}" for m in METRICS))
    for row in rows:
        print(
            f"{row['config']:22} {row['method']:6} {row['n_queries']:>5} "
            + " ".join(f"{row[m]:>9.4f}" for m in METRICS)
        )


if __name__ == "__main__":
    raise SystemExit(main())
