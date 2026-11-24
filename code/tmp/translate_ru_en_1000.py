"""RU→EN для 1000 новых запросов с записью кэша после каждого батча.

Переводы пишет сам чекпойнт-цикл: при обрыве повторный запуск продолжит
с места, а прочитанные из кэша строки не переводятся повторно.
Параметры декодирования — как в Seq2SeqBackend (4 луча, 256 токенов).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from semsearch.data.translate import TranslationCache

SRC = "rus_Cyrl"
TGT = "eng_Latn"
MODEL = "facebook/nllb-200-distilled-600M"
CACHE = Path("var/cache/translation/facebook__nllb-200-distilled-600M-rus_Cyrl-eng_Latn.jsonl")
BATCH = 20
RESTART_EVERY = 200
IN = Path("var/data/test_queries_nllb_ru_1000.jsonl")
OUT = Path("var/data/test_queries_nllb_ru_en_1000.jsonl")


def main() -> None:
    rows = [json.loads(line) for line in IN.open(encoding="utf-8")]
    texts = list(dict.fromkeys(row["query_ru"] for row in rows))
    cache = TranslationCache(CACHE, f"{MODEL}-{SRC}-{TGT}")
    known = cache.load()
    pending = [t for t in texts if cache.key(t) not in known]
    print(f"текстов {len(texts)}, в кэше {len(texts) - len(pending)}, перевести {len(pending)}",
          flush=True)
    if pending:
        tok = AutoTokenizer.from_pretrained(MODEL, src_lang=SRC)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
        bos = tok.convert_tokens_to_ids(TGT)
        # на загруженной машине 4 потока дают 0.14 запр./с против 0.10 на 8 (tmp/bench_nllb.py)
        torch.set_num_threads(4)
        started = time.perf_counter()
        with torch.no_grad():
            for start in range(0, len(pending), BATCH):
                chunk = pending[start : start + BATCH]
                enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                          max_length=256)
                gen = model.generate(**enc, num_beams=4, max_new_tokens=256,
                                     forced_bos_token_id=bos)
                out = tok.batch_decode(gen, skip_special_tokens=True)
                cache.append([(cache.key(t), t, o) for t, o in zip(chunk, out, strict=True)])
                done = start + len(chunk)
                if done % 160 == 0 or done == len(pending):
                    rate = done / (time.perf_counter() - started)
                    print(f"{done}/{len(pending)}, {rate:.2f} запр./с, "
                          f"осталось ~{(len(pending) - done) / rate / 60:.0f} мин", flush=True)
                # машина живёт у предела swap: длинный цикл вытесняется и его убивают,
                # поэтому выходим сами, а обёртка перезапускает процесс с кэша
                if done >= RESTART_EVERY:
                    print(f"пауза после {done} — перезапуск из кэша", flush=True)
                    return
        del model
    known = cache.load()
    missing = [row["id"] for row in rows if cache.key(row["query_ru"]) not in known]
    if missing:
        sys.exit(f"нет перевода для {len(missing)}: {missing[:5]}")
    with OUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            full = dict(row, query_en_back=known[cache.key(row["query_ru"])])
            handle.write(json.dumps(full, ensure_ascii=False) + "\n")
    print(f"{OUT}: {len(rows)} строк", flush=True)


if __name__ == "__main__":
    main()
