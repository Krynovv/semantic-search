"""Проверка и слияние ручных запросов (задание 03). Только чтение данных, без моделей."""

import json
import sys
from collections import Counter
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
NEW = CODE.parent / "results" / "manual_queries.jsonl"
OLD = CODE / "data" / "manual_queries.jsonl"
POOL = CODE / "var" / "data" / "test_pool.jsonl"


def load(path):
    rows, errors = [], []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append((n, "json", str(exc)))
            continue
        if set(row) != {"id", "query_ru"} or not isinstance(row["id"], str) or not isinstance(row["query_ru"], str):
            errors.append((n, "fields", row))
            continue
        rows.append(row)
    return rows, errors


def has_cyr(text):
    return any("а" <= c.lower() <= "я" or c in "ёЁ" for c in text)


def report():
    new_rows, new_err = load(NEW)
    old_rows, old_err = load(OLD)
    print(f"новых строк: {len(new_rows)}, ошибок JSON/полей: {len(new_err)} {new_err}")
    print(f"старых строк: {len(old_rows)}, ошибок: {len(old_err)} {old_err}")
    new_ids = [r["id"] for r in new_rows]
    pool_ids = {json.loads(line)["id"] for line in POOL.read_text(encoding="utf-8").splitlines() if line.strip()}
    print(f"id в корпусе test_pool: {len(pool_ids)}")
    print("дубли новых:", {k: v for k, v in Counter(new_ids).items() if v > 1})
    print("дубли старых:", {k: v for k, v in Counter(r["id"] for r in old_rows).items() if v > 1})
    print("новые id не в корпусе:", [i for i in new_ids if i not in pool_ids])
    print("старые id не в корпусе:", [r["id"] for r in old_rows if r["id"] not in pool_ids])
    print("пересечение новых и старых:", sorted(set(new_ids) & {r["id"] for r in old_rows}))
    print("< 3 слов:", [(r["id"], r["query_ru"]) for r in new_rows if len(r["query_ru"].split()) < 3])
    print("пустые:", [(r["id"], r["query_ru"]) for r in new_rows if not r["query_ru"].strip()])
    print("без кириллицы:", [r["id"] for r in new_rows if not has_cyr(r["query_ru"])])
    print("новые id:", new_ids)


def merge():
    """Старые 10 + новые, кроме дублей старых по id (решение исследователя)."""
    new_rows, _ = load(NEW)
    old_rows, _ = load(OLD)
    old_ids = {r["id"] for r in old_rows}
    dropped = [r["id"] for r in new_rows if r["id"] in old_ids]
    out = old_rows + [r for r in new_rows if r["id"] not in old_ids]
    OLD.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(f"отброшено дублей старых: {dropped}")
    print(f"записано {len(out)} строк в {OLD}")


if __name__ == "__main__":
    merge() if "merge" in sys.argv else report()