"""Проверка двух выходных файлов NLLB против test_queries.jsonl."""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path("var/data")
QUERIES = DATA / "test_queries.jsonl"
RU = DATA / "test_queries_nllb_ru_1000.jsonl"
RU_EN = DATA / "test_queries_nllb_ru_en_1000.jsonl"


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


queries = load(QUERIES)
ru = load(RU)
ru_en = load(RU_EN)

for path, rows, extra in ((RU, ru, "query_ru"), (RU_EN, ru_en, "query_en_back")):
    errors = []
    if len(rows) != 1000:
        errors.append(f"строк {len(rows)}")
    if [r["id"] for r in rows] != [q["id"] for q in queries]:
        errors.append("id/порядок не совпадают с test_queries.jsonl")
    if [r["query_en"] for r in rows] != [q["query_en"] for q in queries]:
        errors.append("query_en изменён")
    if any(not r[extra].strip() for r in rows):
        errors.append(f"пустые {extra}")
    if len({r[extra] for r in rows}) < 900:
        errors.append(f"подозрительно мало разных {extra}")
    print(f"{path}: {'OK' if not errors else errors}")
