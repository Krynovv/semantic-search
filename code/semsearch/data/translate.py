"""Перевод запросов EN->RU и RU->EN (Шаг 1).

Переводчик — локальная модель Helsinki-NLP/opus-mt-en-ru (или opus-mt-ru-en);
для слепого сравнения переводчиков — facebook/nllb-200-distilled-600M.
Переводы кэшируются на диске (ключ — хэш текста и имени модели), поэтому
прерванный запуск продолжается с места обрыва: повторный вызов не обращается
к модели для уже переведённых текстов.

Замечание для отчёта: последовательность EN->RU->EN — круговой перевод.
Для BM25 по запросу RU->EN она даёт оптимистичную (завышенную) оценку,
поэтому выводы о гибридном поиске делаются на ручных запросах.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from semsearch.config import TranslateConfig, cache_dir


@dataclass(frozen=True)
class ModelSpec:
    """Модель перевода: имя и языковые коды.

    lang-коды нужны только NLLB: у Marian (opus-mt) язык задан самой моделью,
    у NLLB модель одна на все языки, поэтому язык передаётся токенизатору.
    """

    name: str
    src_lang: str | None = None
    tgt_lang: str | None = None

    @property
    def cache_id(self) -> str:
        """Идентификатор для кэша: у NLLB модель одна, а направлений два."""
        if self.src_lang is None:
            return self.name
        return f"{self.name}-{self.src_lang}-{self.tgt_lang}"


DIRECTIONS = {
    "en2ru": ModelSpec("Helsinki-NLP/opus-mt-en-ru"),
    "ru2en": ModelSpec("Helsinki-NLP/opus-mt-ru-en"),
    "nllb-en2ru": ModelSpec("facebook/nllb-200-distilled-600M", "eng_Latn", "rus_Cyrl"),
    "nllb-ru2en": ModelSpec("facebook/nllb-200-distilled-600M", "rus_Cyrl", "eng_Latn"),
}


class TranslationBackend(Protocol):
    """Минимальный интерфейс переводчика (для подмены в тестах)."""

    def translate(self, texts: list[str]) -> list[str]:
        """Перевести список текстов."""
        ...


@dataclass(frozen=True)
class TranslationCache:
    """Дисковый кэш переводов (jsonl: key/text/translation).

    model_id — имя модели (для NLLB с языковыми кодами): входит в ключ,
    поэтому переводы разных переводчиков не смешиваются.
    """

    path: Path
    model_id: str

    def key(self, text: str) -> str:
        """Ключ кэша: хэш текста вместе с именем модели."""
        raw = f"{self.model_id}\x00{text}".encode()
        return hashlib.sha256(raw).hexdigest()

    def load(self) -> dict[str, str]:
        """Загрузить кэш из файла."""
        data: dict[str, str] = {}
        if not self.path.exists():
            return data
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                data[record["key"]] = record["translation"]
        return data

    def append(self, records: Sequence[tuple[str, str, str]]) -> None:
        """Дописать переводы в файл кэша (с flush — можно прервать и продолжить)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for key, text, translation in records:
                record = {"key": key, "text": text, "translation": translation}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


def _is_float16(dtype: Any) -> bool:
    """dtype задан как float16 — строкой "float16" или объектом torch.float16."""
    return dtype == "float16" or getattr(dtype, "__name__", "") == "float16"


class Seq2SeqBackend:
    """Обёртка над seq2seq-моделью HuggingFace (Marian/opus-mt и NLLB)."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 16,
        num_beams: int = 4,
        max_input_tokens: int = 256,
        max_new_tokens: int = 256,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
        dtype: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if device == "cuda" and not torch.cuda.is_available():
            print("[translate] CUDA недоступна — работаю на CPU")
            device = "cpu"
        self.device = device
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        # язык NLLB передаётся токенизатору, язык перевода — в forced_bos_token_id
        langs: dict[str, str] = {}
        if src_lang is not None:
            langs["src_lang"] = src_lang
        if tgt_lang is not None:
            langs["tgt_lang"] = tgt_lang
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **langs)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(device)
        # fp16 на T4 даёт ~2× ускорения; на CPU половина операций float16 не поддержана
        if _is_float16(dtype):
            if device == "cuda":
                self.model.half()
            else:
                print("[translate] float16 на CPU не поддерживается — оставляю float32")
        self.model.eval()
        self.forced_bos_token_id = (
            self.tokenizer.convert_tokens_to_ids(tgt_lang) if tgt_lang is not None else None
        )

    def translate(self, texts: list[str]) -> list[str]:
        """Перевести список текстов батчами.

        Тексты сортируются по длине: в батче из строк похожей длины меньше
        padding-вычислений. `order` восстанавливает порядок входа.
        """
        import torch

        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        results: list[str | None] = [None] * len(texts)
        for start in range(0, len(order), self.batch_size):
            indices = order[start : start + self.batch_size]
            encoded = self.tokenizer(
                [texts[index] for index in indices],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
            ).to(self.device)
            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    num_beams=self.num_beams,
                    max_new_tokens=self.max_new_tokens,
                    forced_bos_token_id=self.forced_bos_token_id,
                )
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            for index, translation in zip(indices, decoded, strict=True):
                results[index] = translation
        return [item if item is not None else "" for item in results]


@dataclass
class CachedTranslator:
    """Переводчик с дисковым кэшем: модель вызывается только для новых текстов.

    Модель создаётся лениво (backend_factory) — если всё уже переведено и лежит
    в кэше, запускать модель не нужно вовсе.
    """

    backend_factory: Callable[[], TranslationBackend]
    cache: TranslationCache
    backend: TranslationBackend | None = None

    def _get_backend(self) -> TranslationBackend:
        """Создать модель при первом обращении."""
        if self.backend is None:
            self.backend = self.backend_factory()
        return self.backend

    def translate(self, texts: list[str], flush_every: int = 1500) -> tuple[list[str], int, int]:
        """Перевести тексты. Возвращает (переводы, из кэша, переведено моделью).

        Новые переводы пишутся в кэш порциями по flush_every (последняя порция —
        остаток): при обрыве сессии Colab теряется не больше одной порции.
        """
        cached = self.cache.load()
        results: list[str | None] = [None] * len(texts)
        misses: list[tuple[int, str, str]] = []
        for index, text in enumerate(texts):
            key = self.cache.key(text)
            hit = cached.get(key)
            if hit is None:
                misses.append((index, key, text))
            else:
                results[index] = hit
        for start in range(0, len(misses), flush_every):
            chunk = misses[start : start + flush_every]
            translations = self._get_backend().translate([text for _, _, text in chunk])
            records: list[tuple[str, str, str]] = []
            for (index, key, text), translation in zip(chunk, translations, strict=True):
                results[index] = translation
                records.append((key, text, translation))
            self.cache.append(records)
        done = [item if item is not None else "" for item in results]
        return done, len(texts) - len(misses), len(misses)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Прочитать jsonl."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Записать jsonl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def cache_path(cache_id: str) -> Path:
    """Путь к файлу кэша модели (cache_id — имя модели + направление)."""
    slug = cache_id.replace("/", "__")
    return cache_dir() / "translation" / f"{slug}.jsonl"


def export_sample_csv(path: Path, pairs: Sequence[tuple[str, str]], count: int, seed: int) -> int:
    """Выгрузить случайные пары (query_en, query_ru) в csv с пустой оценкой."""
    rng = random.Random(seed)
    chosen = rng.sample(list(pairs), min(count, len(pairs)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_en", "query_ru", "rating"])
        for query_en, query_ru in chosen:
            writer.writerow([query_en, query_ru, ""])
    return len(chosen)


def translate_file(
    input_path: Path,
    output_path: Path,
    field: str,
    out_field: str,
    spec: ModelSpec,
    limit: int | None,
    device: str,
    cfg: TranslateConfig,
    sample_csv: Path | None = None,
    sample_count: int = 100,
    seed: int = 42,
    flush_every: int = 1500,
    benchmark_limit: int | None = None,
) -> dict[str, Any]:
    """Переводит поле jsonl-файла и записывает результат в новый jsonl.

    benchmark_limit — перевести только первые N запросов (оценка скорости на GPU),
    в summary добавляется поле "benchmark": true.
    """
    rows = read_jsonl(input_path)
    if limit is not None:
        rows = rows[:limit]
    if benchmark_limit is not None:
        rows = rows[:benchmark_limit]
    texts = [str(row.get(field, "")) for row in rows]
    load_holder: dict[str, float] = {}

    def make_backend() -> TranslationBackend:
        started = time.perf_counter()
        backend = Seq2SeqBackend(
            spec.name,
            device=device,
            batch_size=cfg.batch_size,
            num_beams=cfg.num_beams,
            max_input_tokens=cfg.max_input_tokens,
            max_new_tokens=cfg.max_new_tokens,
            src_lang=spec.src_lang,
            tgt_lang=spec.tgt_lang,
            dtype=cfg.dtype,
        )
        load_holder["sec"] = time.perf_counter() - started
        return backend

    translator = CachedTranslator(
        make_backend,
        TranslationCache(cache_path(spec.cache_id), spec.cache_id),
    )

    started = time.perf_counter()
    translations, from_cache, from_model = translator.translate(texts, flush_every=flush_every)
    elapsed = time.perf_counter() - started

    for row, translation in zip(rows, translations, strict=True):
        row[out_field] = translation
    write_jsonl(output_path, rows)

    sample_written = 0
    if sample_csv is not None:
        pairs = [
            (str(row.get(field, "")), translation)
            for row, translation in zip(rows, translations, strict=True)
        ]
        sample_written = export_sample_csv(sample_csv, pairs, sample_count, seed)

    per_second = len(texts) / elapsed if elapsed > 0 else 0.0
    summary: dict[str, Any] = {
        "model": spec.name,
        "src_lang": spec.src_lang,
        "tgt_lang": spec.tgt_lang,
        "device": device,
        "input": str(input_path),
        "output": str(output_path),
        "sentences": len(texts),
        "from_cache": from_cache,
        "translated_by_model": from_model,
        "model_load_sec": round(load_holder.get("sec", 0.0), 2),
        "elapsed_sec": round(elapsed, 2),
        "sentences_per_sec": round(per_second, 2),
        "sample_csv": str(sample_csv) if sample_csv else None,
        "sample_rows": sample_written,
    }
    if benchmark_limit is not None:
        summary["benchmark"] = True
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Перевод запросов EN<->RU (Шаг 1)")
    parser.add_argument("--input", type=Path, required=True, help="исходный jsonl")
    parser.add_argument("--output", type=Path, required=True, help="jsonl с переведённым полем")
    parser.add_argument("--field", required=True, help="поле с исходным текстом")
    parser.add_argument("--out-field", required=True, help="имя поля с переводом")
    parser.add_argument("--direction", choices=sorted(DIRECTIONS), required=True)
    parser.add_argument("--model", default=None, help="переопределить модель перевода (имя HF)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--sample-csv", type=Path, default=None, help="csv из 100 случайных пар")
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--flush-every", type=int, default=1500, help="сброс кэша каждые N запросов"
    )
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument(
        "--benchmark", type=int, default=None, help="режим бенчмарка: перевести N запросов"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI."""
    args = _build_parser().parse_args(argv)
    cfg = replace(TranslateConfig(), dtype=args.dtype)
    if args.batch_size is not None or args.num_beams is not None:
        cfg = replace(
            cfg,
            batch_size=args.batch_size if args.batch_size is not None else cfg.batch_size,
            num_beams=args.num_beams if args.num_beams is not None else cfg.num_beams,
        )
    spec = DIRECTIONS[args.direction]
    if args.model:
        spec = replace(spec, name=args.model)
    summary = translate_file(
        input_path=args.input,
        output_path=args.output,
        field=args.field,
        out_field=args.out_field,
        spec=spec,
        limit=args.limit,
        device=args.device,
        cfg=cfg,
        sample_csv=args.sample_csv,
        sample_count=args.sample_count,
        seed=args.seed,
        flush_every=args.flush_every,
        benchmark_limit=args.benchmark,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
