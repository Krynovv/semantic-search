"""Замер декодирования NLLB RU→EN: batch x threads, сверка с эталоном backend.translate."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, ".")
from semsearch.data.translate import DIRECTIONS, Seq2SeqBackend  # noqa: E402

rows = [json.loads(line) for line in Path("var/data/test_queries_nllb_ru_1000.jsonl").open(encoding="utf-8")][:32]
texts = [r["query_ru"] for r in rows]
spec = DIRECTIONS["nllb-ru2en"]

torch.set_grad_enabled(False)
backend = Seq2SeqBackend(spec.name, device="cpu", src_lang=spec.src_lang, tgt_lang=spec.tgt_lang)
tok, model, bos = backend.tokenizer, backend.model, backend.forced_bos_token_id


def run(batch: int) -> list[str]:
    out: list[str] = []
    for start in range(0, len(texts), batch):
        chunk = texts[start : start + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256)
        gen = model.generate(**enc, num_beams=4, max_new_tokens=256, forced_bos_token_id=bos)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


started = time.perf_counter()
ref = run(16)
base = time.perf_counter() - started
print(f"batch 16, threads {torch.get_num_threads()}: {base:.1f}s ({32 / base:.2f} req/s)")
for threads in (4, 8):
    torch.set_num_threads(threads)
    for batch in (32, 64):
        started = time.perf_counter()
        out = run(batch)
        dt = time.perf_counter() - started
        same = sum(a == b for a, b in zip(out, ref, strict=True))
        print(f"threads {threads}, batch {batch}: {dt:.1f}s ({32 / dt:.2f} req/s), "
              f"same as ref {same}/32")
print(ref[0][:120])
