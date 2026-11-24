"""Собрать test_queries_nllb_ru_1000.jsonl и _ru_en_1000.jsonl из кэша NLLB.

EN→RU берётся из кэша прогона Colab (3000 записей, тест покрыт полностью).
RU→EN в том кэше пустой (2000 строк "text": ""), поэтому переводится заново
через translate_file — существующий файл кэша он не трогает (другие ключи).
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

from semsearch.config import TranslateConfig
from semsearch.data.translate import DIRECTIONS, read_jsonl, translate_file, write_jsonl

DATA = Path("var/data")
QUERIES = DATA / "test_queries.jsonl"
OUT_RU = DATA / "test_queries_nllb_ru_1000.jsonl"
OUT_RU_EN = DATA / "test_queries_nllb_ru_en_1000.jsonl"
COLAB_EN_RU = (
    Path("../nllb_translation_results/cache/cache/translation/"
         "facebook__nllb-200-distilled-600M-eng_Latn-rus_Cyrl.jsonl")
)


def en_ru_map() -> dict[str, str]:
    return {record["text"]: record["translation"] for record in read_jsonl(COLAB_EN_RU)}


def main() -> None:
    queries = read_jsonl(QUERIES)
    en_ru = en_ru_map()
    missing = [q["id"] for q in queries if q["query_en"] not in en_ru]
    if missing:
        sys.exit(f"нет EN→RU перевода для {len(missing)} запросов: {missing[:5]}")

    rows = [
        {"id": q["id"], "query_en": q["query_en"], "query_ru": en_ru[q["query_en"]]}
        for q in queries
    ]
    write_jsonl(OUT_RU, rows)
    print(f"{OUT_RU}: {len(rows)} строк", flush=True)

    summary = translate_file(
        input_path=OUT_RU,
        output_path=OUT_RU_EN,
        field="query_ru",
        out_field="query_en_back",
        spec=DIRECTIONS["nllb-ru2en"],
        limit=None,
        device="cpu",
        cfg=TranslateConfig(),
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
