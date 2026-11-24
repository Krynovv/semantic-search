"""Оценка дообученной модели (Шаг 3).

Тот же путь, что у baseline'ов Шага 2 (`Embedder` + `DenseIndex` + `evaluate`),
поэтому метрики «до» и «после» сравнимы напрямую. Один раз строится индекс
test-корпуса (`results/test_pool.jsonl`, 5000 функций), потом по нему
оцениваются три набора запросов:

- EN — `query_en` из файла запросов;
- RU — NLLB-перевод из `test_queries_nllb_ru_en_1000.jsonl` рядом с ним;
- ручные — `manual_queries.jsonl` (написаны автором без взгляда на код).

В Colab модель оценивается прямо в памяти во время обучения (`evaluate`
параметром `model=`), в CLI — уже сохранённая (`model_path`).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from semsearch.config import EvalConfig
from semsearch.core.chunk import CodeChunk
from semsearch.core.index_dense import DenseIndex, Embedder
from semsearch.core.metrics import evaluate

RU_QUERIES_FILE = "test_queries_nllb_ru_en_1000.jsonl"
DEFAULT_MANUAL_QUERIES = Path(__file__).resolve().parents[3] / "results" / "manual_queries.jsonl"
SOURCES = ("en", "ru", "manual")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Прочитать jsonl."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_model(model: str, device: str) -> SentenceTransformer:
    """Модель из каталога или хаба с длиной входа e5 (512)."""
    loaded = SentenceTransformer(model, device=device)
    loaded.max_seq_length = EvalConfig().max_seq_length
    return loaded


def load_test_queries(queries_path: Path) -> list[tuple[str, str, str]]:
    """(источник, запрос, id ответа) для EN и RU; RU берётся из NLLB-файла."""
    queries = load_jsonl(queries_path)
    ru_path = queries_path.parent / RU_QUERIES_FILE
    ru_by_id = {row["id"]: row["query_ru"] for row in load_jsonl(ru_path)}
    triples = [("en", row["query_en"], row["id"]) for row in queries]
    triples += [("ru", ru_by_id[row["id"]], row["id"]) for row in queries]
    return triples


class EmbeddingEvaluator:
    """MRR@10 и Recall@10 на EN, RU и ручных запросах по test-корпусу."""

    def __init__(
        self, queries_path: Path, pool_path: Path, manual_queries_path: Path | None = None
    ) -> None:
        self.cfg = EvalConfig()
        manual_path = manual_queries_path or DEFAULT_MANUAL_QUERIES
        manual = [("manual", row["query_ru"], row["id"]) for row in load_jsonl(manual_path)]
        self.triples = load_test_queries(queries_path) + manual
        self.corpus = [CodeChunk(row["id"], row["code"]) for row in load_jsonl(pool_path)]
        known = {chunk.id for chunk in self.corpus}
        if missing := [answer for _, _, answer in self.triples if answer not in known]:
            raise ValueError(f"ответа нет в корпусе, например {missing[0]}")

    def __call__(self, model: SentenceTransformer) -> dict[str, float]:
        """Метрики по индексу, построенному на этой модели."""
        index = DenseIndex(Embedder(self.cfg, model=model)).build(self.corpus)
        rankings = index.search_many([query for _, query, _ in self.triples], self.cfg.k)
        metrics: dict[str, float] = {}
        for source in SOURCES:
            pairs = [
                (rank, answer)
                for rank, (name, _, answer) in zip(rankings, self.triples, strict=True)
                if name == source
            ]
            values = evaluate([rank for rank, _ in pairs], [a for _, a in pairs], self.cfg.k)
            metrics[f"mrr_{source}"] = values["MRR@10"]
            metrics[f"recall_{source}"] = values["Recall@10"]
        return metrics


def evaluate_model(
    model_path: Path, queries_path: Path, pool_path: Path, manual_queries_path: Path | None = None
) -> dict[str, float]:
    """Оценить сохранённую модель (CLI: ~1 ч на CPU на 5000 функций)."""
    cfg = replace(EvalConfig(), model_name=str(model_path))
    model = resolve_model(str(model_path), cfg.device)
    return EmbeddingEvaluator(queries_path, pool_path, manual_queries_path)(model)
