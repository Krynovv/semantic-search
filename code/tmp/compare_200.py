"""Метрики на первых 200 id для сравнения с прогоном Шага 2 (n=200)."""
from __future__ import annotations

import json
from pathlib import Path

from semsearch.config import EvalConfig
from semsearch.core.chunk import CodeChunk
from semsearch.core.index_dense import DenseIndex, Embedder
from semsearch.core.index_lexical import BM25Index
from semsearch.core.metrics import METRICS, evaluate

DATA = Path("var/data")
K = 10


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


ids = [r["id"] for r in load(DATA / "step2/test_queries_200en.jsonl")]
en = {r["id"]: r for r in load(DATA / "test_queries.jsonl")}
ru = {r["id"]: r for r in load(DATA / "test_queries_nllb_ru_1000.jsonl")}
back = {r["id"]: r for r in load(DATA / "test_queries_nllb_ru_en_1000.jsonl")}
sets = {
    "en": [en[i]["query_en"] for i in ids],
    "ru_translated": [ru[i]["query_ru"] for i in ids],
    "ru_translated+mt_en": [back[i]["query_en_back"] for i in ids],
}
corpus = [CodeChunk(r["id"], r["code"]) for r in load(DATA / "test_pool.jsonl")]
cfg = EvalConfig(device="cpu", batch_size=32)
indexes = {"bm25": BM25Index(corpus).build(), "e5": DenseIndex(Embedder(cfg)).build(corpus)}
for method, index in indexes.items():
    for name, queries in sets.items():
        if method == "e5" and name.endswith("+mt_en"):
            continue
        values = evaluate(index.search_many(queries, K), ids, K)
        print(f"{method:5}{name:22}" + " ".join(f"{values[m]:8.4f}" for m in METRICS))