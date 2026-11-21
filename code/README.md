# semsearch — семантический поиск кода по русским запросам

Практическая часть курсовой: подготовка данных (Шаг 1) для гибридного семантического
поиска программного кода на Python по запросам на русском языке.

Исходный план: `kursovaya-tex/task/task-01-practice (3).md`. Решения по ходу работы —
в `DECISIONS.md`, итоги шага — в `kursovaya-tex/results/step1-report.md`.

## Окружение

Python 3.11+, зависимости зафиксированы в `requirements.txt` (собраны под Python 3.12,
Linux x86_64, CPU-версия torch).

```bash
cd kursovaya-tex/code
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt
```

Пакеты ставятся только внутрь `.venv` (системный Python не трогаем).

## Пути

Корень данных, кэша и моделей — переменная окружения `SEMSEARCH_HOME`
(по умолчанию `kursovaya-tex/code/var`, то есть вне git). Пути внутри кода строятся
через `pathlib`, жёстких `/home/...` и `/content/...` нет.

```bash
export SEMSEARCH_HOME=/путь/к/хранилищу     # опционально
```

## Шаг 1: данные

### Сборка train/val/test

```bash
.venv/bin/python -m semsearch.data.prepare prepare                 # полный прогон
.venv/bin/python -m semsearch.data.prepare prepare --limit 20000   # уменьшенный прогон (отладка)
```

Что происходит: загрузка CodeSearchNet (python) → очистка → удаление docstring из кода
через AST → дедупликация по коду и по запросу → формирование сплитов. Файлы результата
(в `$SEMSEARCH_HOME/data/`):

| файл | содержимое |
|---|---|
| `train.jsonl` | пары «запрос → функция» для дообучения |
| `val_pool.jsonl` | корпус кандидатов val (N = 5000 функций) |
| `val_queries.jsonl` | 2000 запросов из этого корпуса |
| `test_pool.jsonl` | корпус кандидатов test (N = 5000 функций) |
| `test_queries.jsonl` | 1000 запросов из этого корпуса |
| `stats.json` | счётчики отсева по каждому фильтру |

Поля jsonl: `id`, `query_en`, `code`, `func_name`, `url`, `repo`.
В `code` docstring уже удалён, поэтому запрос не встречается в коде (проверяется тестом).

### Пул функций для ручных запросов

```bash
.venv/bin/python -m semsearch.data.prepare export-manual-pool --count 150
```

Пишет `kursovaya-tex/results/manual_pool.md` — 150 случайных функций тест-корпуса
(id, имя, код без docstring), чтобы автор мог писать к ним русские запросы.

### Ручные запросы

Файл `data/manual_queries.jsonl`, одна запись на строку:

```json
{"id": "test-1234", "query_ru": "функция, которая считает сумму элементов списка"}
```

`id` — функция из `test_pool.jsonl`. Цель — 100 запросов (минимум 50) от разных функций;
в файле уже есть 10 примеров (сделаны на функциях из локального пула).

### Перевод запросов EN→RU и RU→EN

Локальные модели `Helsinki-NLP/opus-mt-en-ru` и `Helsinki-NLP/opus-mt-ru-en`.
Переводы кэшируются на диске (`$SEMSEARCH_HOME/cache/translation/*.jsonl`), поэтому
прерванный запуск продолжается с места обрыва.

```bash
# локально, CPU, на 200 запросах (пример из отчёта)
.venv/bin/python -m semsearch.data.translate \
  --input $SEMSEARCH_HOME/data/test_queries.jsonl \
  --output $SEMSEARCH_HOME/data/test_queries_ru.jsonl \
  --field query_en --out-field query_ru \
  --direction en2ru --limit 200 --device cpu \
  --sample-csv ../results/translation_sample.csv --sample-count 100

# обратный перевод RU→EN (для варианта BM25)
.venv/bin/python -m semsearch.data.translate \
  --input $SEMSEARCH_HOME/data/test_queries_ru.jsonl \
  --output $SEMSEARCH_HOME/data/test_queries_en_back.jsonl \
  --field query_ru --out-field query_en_back \
  --direction ru2en --limit 200 --device cpu
```

Направления `--direction`: `en2ru`, `ru2en` (opus-mt) и `nllb-en2ru`, `nllb-ru2en`
(`facebook/nllb-200-distilled-600M` — язык передаётся токенизатору, у этой модели
в кэш попадает ещё и направление, иначе EN→RU и RU→EN смешались бы).

Полный прогон (100k+ запросов) локально **не выполняется** — он идёт в Colab на GPU:

```bash
# в Colab: те же команды, но --device cuda и без --limit
```

### Проверки качества

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy
```

Тесты `tests/test_data.py`: удаление docstring (простая функция, метод, `async`,
декоратор, без docstring, синтаксическая ошибка), нормализация, отсутствие запроса
в коде, отсутствие пересечений сплитов, работа кэша перевода (на фейковой модели).
Тесты `tests/test_blind.py`: сборка слепого файла, соответствие по ключу, сведение
оценок на ручном примере, точный тест знаков.

### Слебое сравнение переводчиков (opus-mt против NLLB)

Оценщик не должен знать, какой перевод где — поэтому в `results/translation_blind.csv`
два перевода переставлены случайно (seed 42), а соответствие лежит отдельно
в `results/translation_blind_key.csv` (её открывают только после оценки).

```bash
# 1) собрать слепой файл и ключ
.venv/bin/python -m semsearch.data.blind make \
  --opus-csv ../results/translation_sample.csv \
  --nllb-jsonl tmp/queries_100_nllb.jsonl \
  --out-csv ../results/translation_blind.csv \
  --key-csv ../results/translation_blind_key.csv

# 2) оценщик проставляет rating_a и rating_b в translation_blind.csv (школа 1–5)

# 3) свести оценки (после заполнения)
.venv/bin/python -m semsearch.data.blind score \
  --blind-csv ../results/translation_blind.csv \
  --key-csv ../results/translation_blind_key.csv
```
