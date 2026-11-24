"""Шаг 1: подготовка данных CodeSearchNet (Python).

Что делает модуль:
1. Загружает подмножество python датасета ``code-search-net/code_search_net``
   (через loading script, при недоступности — parquet-файлы напрямую).
2. Очищает примеры: требует docstring (>= 3 слов), выделяет запрос,
   удаляет docstring из кода через ``ast``, отбрасывает код длиннее
   512 токенов (токенизатор intfloat/multilingual-e5-base).
3. Дедуплицирует по нормализованному коду и по нормализованному запросу
   (глобально, между всеми сплитами — иначе тест «утёк» бы в train).
4. Формирует сплиты: train (пары), val и test (корпус кандидатов + запросы
   из этого корпуса).

Файлы результата (jsonl, поля id/query_en/code/func_name/url/repo):
``train.jsonl``, ``val_pool.jsonl``, ``val_queries.jsonl``,
``test_pool.jsonl``, ``test_queries.jsonl``.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import random
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semsearch.config import DataConfig, data_dir, results_dir

DATASET_ID = "code-search-net/code_search_net"
CONFIG_NAME = "python"
SPLITS = ("train", "validation", "test")
"""Официальные сплиты CodeSearchNet (validation, а не val)."""

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
"""Знаки препинания для нормализации запроса."""


# --------------------------------------------------------------------------
# Нормализация
# --------------------------------------------------------------------------


def normalize_code(code: str) -> str:
    """Нормализация кода для дедупликации: нижний регистр + схлопнутые пробелы."""
    return " ".join(code.lower().split())


def normalize_query(text: str) -> str:
    """Нормализация запроса: нижний регистр, без пунктуации, схлопнутые пробелы."""
    return " ".join(_PUNCT_RE.sub(" ", text.lower()).split())


def extract_query(docstring: str, max_words: int) -> str:
    """Запрос из docstring: первый абзац (до первой пустой строки), обрезанный по словам."""
    text = inspect.cleandoc(docstring.replace("\r\n", "\n"))
    paragraph = text.split("\n\n", 1)[0]
    words = paragraph.split()
    return " ".join(words[:max_words])


def word_count(text: str) -> int:
    """Число слов в тексте."""
    return len(text.split())


# --------------------------------------------------------------------------
# Удаление docstring из кода через AST
# --------------------------------------------------------------------------


class DocstringRemovalError(ValueError):
    """Код не удалось разобрать или в нём нет функции."""


@dataclass(frozen=True)
class RemovalResult:
    """Результат удаления docstring."""

    code: str
    had_docstring: bool
    used_pass: bool
    """True, если тело функции состояло только из docstring и пришлось заменить его на pass."""


def _byte_line_offsets(data: bytes) -> list[int]:
    """Байтовые смещения начала строк (ast.col_offset — это байты, не символы)."""
    offsets = [0]
    for line in data.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _find_function(tree: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Первая функция в дереве (в CSN код — это одна функция)."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    for candidate in ast.walk(tree):
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return candidate
    raise DocstringRemovalError("в коде нет функции")


def remove_docstring(code: str) -> RemovalResult:
    """Удаляет docstring функции, сохраняя комментарии и форматирование.

    Работает по байтовым смещениям ``ast`` (без ``ast.unparse``), поэтому
    исходный вид кода не меняется, кроме самого docstring.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise DocstringRemovalError(f"код не разбирается: {exc}") from exc

    func = _find_function(tree)
    if not func.body:
        raise DocstringRemovalError("у функции пустое тело")
    first = func.body[0]
    if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)):
        return RemovalResult(code=code, had_docstring=False, used_pass=False)
    literal = first.value
    if not isinstance(literal.value, str):
        return RemovalResult(code=code, had_docstring=False, used_pass=False)

    data = code.encode("utf-8")
    offsets = _byte_line_offsets(data)
    if literal.lineno is None or literal.col_offset is None:
        raise DocstringRemovalError("у docstring нет позиций в исходном коде")
    if literal.end_lineno is None or literal.end_col_offset is None:
        raise DocstringRemovalError("у docstring нет конечных позиций в исходном коде")
    start = offsets[literal.lineno - 1] + literal.col_offset
    end = offsets[literal.end_lineno - 1] + literal.end_col_offset
    segment = data[start:end]

    # Число строк внутри docstring сохраняем, чтобы не поехали номера строк.
    newlines = segment.count(b"\n")
    body_only_docstring = len(func.body) == 1
    replacement = b"pass" if body_only_docstring else b""
    new_code = data[:start] + replacement + b"\n" * newlines + data[end:]

    try:
        ast.parse(new_code.decode("utf-8"))
    except SyntaxError as exc:  # pragma: no cover - защитная ветка
        raise DocstringRemovalError(f"после удаления docstring код не разбирается: {exc}") from exc
    return RemovalResult(
        code=new_code.decode("utf-8"),
        had_docstring=True,
        used_pass=body_only_docstring,
    )


# --------------------------------------------------------------------------
# Очистка
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """Очищенная пара «запрос → функция»."""

    id: str
    query: str
    code: str
    func_name: str
    url: str
    repo: str

    def to_json(self) -> dict[str, str]:
        """Представление для jsonl."""
        return {
            "id": self.id,
            "query_en": self.query,
            "code": self.code,
            "func_name": self.func_name,
            "url": self.url,
            "repo": self.repo,
        }


@dataclass
class Stats:
    """Счётчики отсева по каждому фильтру."""

    counts: dict[str, int]

    def bump(self, name: str, n: int = 1) -> None:
        """Увеличить счётчик фильтра."""
        self.counts[name] = self.counts.get(name, 0) + n


def _row_field(row: dict[str, Any], *names: str) -> str:
    """Первое непустое поле строки из набора возможных имён."""
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def filter_rows(
    rows: Iterable[dict[str, Any]],
    split: str,
    cfg: DataConfig,
    stats: Stats,
) -> list[Pair]:
    """Дешёвые фильтры: docstring, длина запроса, разбор AST, утечка запроса в код."""
    items: list[Pair] = []
    for index, row in enumerate(rows):
        item_id = f"{split}-{index}"
        docstring = _row_field(row, "func_documentation_string", "docstring")
        if not docstring:
            stats.bump("no_docstring")
            continue
        if word_count(docstring) < cfg.min_docstring_words:
            stats.bump("short_docstring")
            continue
        query = extract_query(docstring, cfg.max_query_words)
        if word_count(query) < cfg.min_docstring_words:
            stats.bump("short_query")
            continue
        code = _row_field(row, "func_code_string", "code")
        if not code.strip():
            stats.bump("empty_code")
            continue
        try:
            removal = remove_docstring(code)
        except DocstringRemovalError:
            stats.bump("unparsable_code")
            continue
        if removal.used_pass:
            stats.bump("docstring_only_body")
        if normalize_query(query) in normalize_code(removal.code):
            stats.bump("query_in_code")
            continue
        items.append(
            Pair(
                id=item_id,
                query=query,
                code=removal.code,
                func_name=_row_field(row, "func_name", "function_name", "name"),
                url=_row_field(row, "func_code_url", "url"),
                repo=_row_field(row, "repository_name", "repo"),
            )
        )
    return items


def dedup_and_limit(
    items: Sequence[Pair],
    cfg: DataConfig,
    stats: Stats,
    seen_code: set[str],
    seen_query: set[str],
) -> list[Pair]:
    """Дедупликация по коду и запросу (глобально между сплитами) + фильтр длины кода."""
    kept: list[Pair] = []
    for item in items:
        code_key = normalize_code(item.code)
        if code_key in seen_code:
            stats.bump("dup_code")
            continue
        query_key = normalize_query(item.query)
        if query_key in seen_query:
            stats.bump("dup_query")
            continue
        seen_code.add(code_key)
        seen_query.add(query_key)
        kept.append(item)

    return _filter_by_token_length(kept, cfg, stats)


def _filter_by_token_length(items: Sequence[Pair], cfg: DataConfig, stats: Stats) -> list[Pair]:
    """Отбрасывает код длиннее max_code_tokens токенов токенизатора модели."""
    if not items:
        return []
    from transformers import AutoTokenizer  # локальный импорт: тяжёлая зависимость

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    kept: list[Pair] = []
    batch = 256
    for start in range(0, len(items), batch):
        chunk = items[start : start + batch]
        encoded = tokenizer(
            [item.code for item in chunk],
            add_special_tokens=False,
        )["input_ids"]
        for item, ids in zip(chunk, encoded, strict=True):
            if len(ids) > cfg.max_code_tokens:
                stats.bump("too_long")
                continue
            kept.append(item)
    return kept


# --------------------------------------------------------------------------
# Загрузка датасета
# --------------------------------------------------------------------------


def load_code_search_net(loader: str) -> dict[str, Any]:
    """Загружает датасет. loader: auto | hub | parquet."""
    from datasets import load_dataset

    if loader in ("auto", "hub"):
        try:
            dataset = load_dataset(DATASET_ID, CONFIG_NAME)
            print(f"[prepare] загрузка через {DATASET_ID}/{CONFIG_NAME} (hub) — удалось")
            return {name: dataset[name] for name in SPLITS if name in dataset}
        except Exception as exc:  # noqa: BLE001 — причина логируется и попадает в отчёт
            if loader == "hub":
                raise
            print(f"[prepare] hub-загрузка не удалась ({exc!r}); пробую parquet")
    files = {
        name: f"hf://datasets/{DATASET_ID}/{CONFIG_NAME}/{name}-00000-of-00001.parquet"
        for name in SPLITS
    }
    dataset = load_dataset("parquet", data_files=files)
    print("[prepare] загрузка через parquet-файлы — удалось")
    return {name: dataset[name] for name in SPLITS if name in dataset}


def _rows(dataset: Any, limit: int | None) -> Iterator[dict[str, Any]]:
    """Итерация по строкам датасета с ограничением по количеству."""
    if limit is not None and limit < len(dataset):
        dataset = dataset.select(range(limit))
    for row in dataset:
        yield dict(row)


# --------------------------------------------------------------------------
# Сборка сплитов
# --------------------------------------------------------------------------


def write_jsonl(path: Path, pairs: Iterable[Pair]) -> int:
    """Записывает пары в jsonl, возвращает количество записей."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair.to_json(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Читает jsonl в список словарей."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare(
    cfg: DataConfig,
    limit: int | None,
    loader: str,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Готовит все сплиты и возвращает сводку (счётчики + размеры)."""
    dataset = load_code_search_net(loader)
    stats = Stats(counts={})
    seen_code: set[str] = set()
    seen_query: set[str] = set()

    cleaned: dict[str, list[Pair]] = {}
    for split in SPLITS:
        raw = filter_rows(_rows(dataset[split], limit), split, cfg, stats)
        cleaned[split] = dedup_and_limit(raw, cfg, stats, seen_code, seen_query)
        print(f"[prepare] {split}: осталось {len(cleaned[split])} пар")

    rng = random.Random(seed)
    for split in SPLITS:
        rng.shuffle(cleaned[split])

    train = cleaned["train"][: cfg.train_size]
    val_pool = cleaned["validation"][: cfg.val_pool_size]
    val_queries = val_pool[: cfg.val_queries]
    test_pool = cleaned["test"][: cfg.test_pool_size]
    test_queries = test_pool[: cfg.test_queries]

    out_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "train.jsonl": write_jsonl(out_dir / "train.jsonl", train),
        "val_pool.jsonl": write_jsonl(out_dir / "val_pool.jsonl", val_pool),
        "val_queries.jsonl": write_jsonl(out_dir / "val_queries.jsonl", val_queries),
        "test_pool.jsonl": write_jsonl(out_dir / "test_pool.jsonl", test_pool),
        "test_queries.jsonl": write_jsonl(out_dir / "test_queries.jsonl", test_queries),
    }

    check_no_overlap(train, val_pool, test_pool)
    summary = {
        "dataset": DATASET_ID,
        "config": CONFIG_NAME,
        "loader": loader,
        "limit": limit,
        "seed": seed,
        "filters": stats.counts,
        "written": written,
        "corpus_sizes": {
            "train_pairs": len(train),
            "val_pool": len(val_pool),
            "val_queries": len(val_queries),
            "test_pool": len(test_pool),
            "test_queries": len(test_queries),
        },
    }
    (out_dir / "stats.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def check_no_overlap(
    train: Sequence[Pair],
    val_pool: Sequence[Pair],
    test_pool: Sequence[Pair],
) -> None:
    """Проверяет, что сплиты не пересекаются по id, коду и запросу."""

    def keys(items: Sequence[Pair]) -> tuple[set[str], set[str], set[str]]:
        return (
            {item.id for item in items},
            {normalize_code(item.code) for item in items},
            {normalize_query(item.query) for item in items},
        )

    train_ids, train_code, train_query = keys(train)
    val_ids, val_code, val_query = keys(val_pool)
    test_ids, test_code, test_query = keys(test_pool)
    for name, left, right in (
        ("id", train_ids, val_ids),
        ("id", train_ids, test_ids),
        ("id", val_ids, test_ids),
        ("code", train_code, val_code),
        ("code", train_code, test_code),
        ("code", val_code, test_code),
        ("query", train_query, val_query),
        ("query", train_query, test_query),
        ("query", val_query, test_query),
    ):
        overlap = left & right
        if overlap:
            example = next(iter(overlap))
            raise AssertionError(
                f"пересечение сплитов по {name}: {len(overlap)} (пример: {example!r})"
            )


# --------------------------------------------------------------------------
# Пул функций для ручных запросов
# --------------------------------------------------------------------------


def export_manual_pool(pool_path: Path, out_path: Path, count: int, seed: int) -> int:
    """Выгружает случайные функции тест-корпуса в markdown для ручных запросов."""
    rows = read_jsonl(pool_path)
    rng = random.Random(seed)
    chosen = rng.sample(rows, min(count, len(rows)))
    lines = [
        "# Пул функций для ручных запросов",
        "",
        f"Источник: `{pool_path}` (тест-корпус, {len(rows)} функций).",
        f"Выбрано {len(chosen)} случайных функций (seed={seed}).",
        "",
        "Для каждой функции автор пишет русский запрос в `data/manual_queries.jsonl`:",
        "```json",
        '{"id": "<id функции>", "query_ru": "<запрос на русском>"}',
        "```",
        "",
    ]
    for row in chosen:
        lines.append(f"## {row['id']} — {row['func_name'] or '(без имени)'}")
        lines.append("")
        lines.append(f"- url: {row['url']}")
        lines.append(f"- repo: {row['repo']}")
        lines.append("")
        lines.append("```python")
        lines.append(row["code"].rstrip())
        lines.append("```")
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return len(chosen)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Подготовка данных CodeSearchNet (Шаг 1)")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="собрать train/val/test")
    prep.add_argument("--limit", type=int, default=None, help="строк из каждого сплита")
    prep.add_argument("--loader", choices=("auto", "hub", "parquet"), default="auto")
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--out", type=Path, default=None, help="каталог jsonl ($SEMSEARCH_HOME/data)")

    pool = sub.add_parser("export-manual-pool", help="выгрузить пул функций для ручных запросов")
    pool.add_argument("--pool", type=Path, default=None, help="jsonl тест-корпуса")
    pool.add_argument("--count", type=int, default=150)
    pool.add_argument("--seed", type=int, default=42)
    pool.add_argument("--out", type=Path, default=None, help="markdown-файл пула")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI."""
    args = _build_parser().parse_args(argv)
    cfg = DataConfig()
    if args.command == "prepare":
        out_dir = args.out or data_dir()
        summary = prepare(cfg, args.limit, args.loader, args.seed, out_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    pool_path = args.pool or (data_dir() / "test_pool.jsonl")
    out_path = args.out or (results_dir() / "manual_pool.md")
    count = export_manual_pool(pool_path, out_path, args.count, args.seed)
    print(f"[prepare] в {out_path} выгружено {count} функций")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
