# Шаг 1 — Данные

Отчёт по Шагу 1 практической части курсовой («алгоритм гибридного семантического поиска
программного кода по запросам на русском языке»).
План: `kursovaya-tex/task/task-01-practice (3).md`. Решения: `kursovaya-tex/code/DECISIONS.md`.
Дата прогона: 2026-11-15. Окружение: Arch Linux, Python 3.12.15 (uv-venv), CPU 8 ядер, без GPU.

---

## Сделано

Код: `kursovaya-tex/code/` (пакет `semsearch/`).

| файл | что внутри |
|---|---|
| `semsearch/config.py` | `DataConfig`, `TranslateConfig` (dataclass), пути через `pathlib`, `SEMSEARCH_HOME` |
| `semsearch/data/prepare.py` | загрузка CodeSearchNet, очистка, удаление docstring через AST, дедупликация, сплиты, выгрузка пула для ручных запросов |
| `semsearch/data/translate.py` | перевод EN→RU и RU→EN (opus-mt), дисковый кэш, возобновление после обрыва |
| `tests/test_data.py` | 23 теста: удаление docstring, нормализация, инварианты данных, кэш перевода |
| `pyproject.toml` | настройки ruff / mypy / pytest |
| `requirements.txt` | 60 зафиксированных версий (Python 3.12, CPU-сборка torch) |
| `README.md`, `DECISIONS.md`, `.env.example`, `.gitignore` | документация, решения, пример переменных окружения |
| `data/manual_queries.jsonl` | 10 ручных русских запросов на реальных функциях тест-корпуса |

Результаты: `kursovaya-tex/results/manual_pool.md` (150 функций тест-корпуса для автора),
`kursovaya-tex/results/translation_sample.csv` (100 пар EN→RU с пустой колонкой `rating`).

Ключевые функции `prepare.py`:
`remove_docstring` (AST, вырезание по байтовым смещениям), `extract_query`, `normalize_code`,
`normalize_query`, `filter_rows`, `dedup_and_limit`, `_filter_by_token_length`, `prepare`,
`check_no_overlap`, `export_manual_pool`.

---

## Как запустить

Все команды — из `kursovaya-tex/code/`. Окружение — `code/.venv` (создаётся первой командой),
данные — в `code/var/data` (путь меняется переменной `SEMSEARCH_HOME`).

```bash
# 0. окружение (один раз)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements.txt

# 1. данные (полный прогон: 457 461 строка, 7 минут на CPU)
.venv/bin/python -m semsearch.data.prepare prepare
#    уменьшенный вариант для отладки:
.venv/bin/python -m semsearch.data.prepare prepare --limit 20000

# 2. пул функций для ручных запросов (150 штук)
.venv/bin/python -m semsearch.data.prepare export-manual-pool --count 150

# 3. перевод EN→RU (локально, CPU, 200 запросов)
.venv/bin/python -m semsearch.data.translate \
  --input var/data/test_queries.jsonl \
  --output var/data/test_queries_ru.jsonl \
  --field query_en --out-field query_ru \
  --direction en2ru --limit 200 --device cpu \
  --sample-csv ../results/translation_sample.csv --sample-count 100

# 4. обратный перевод RU→EN (для варианта BM25)
.venv/bin/python -m semsearch.data.translate \
  --input var/data/test_queries_ru.jsonl \
  --output var/data/test_queries_en_back.jsonl \
  --field query_ru --out-field query_en_back \
  --direction ru2en --limit 200 --device cpu

# 5. проверки
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy
```

Полный перевод 100 000 обучающих запросов локально **не запускался** — команда для Colab
описана в `code/README.md` (те же команды, но `--device cuda` и без `--limit`).

---

## Результаты

### Способ загрузки датасета

`load_dataset("code-search-net/code_search_net", "python")` в `datasets 5.2.0` — **работает**:
датасет лежит в parquet, `datasets` сам сгенерировал сплиты. Loading script в `datasets ≥ 3.0`
не поддерживается, но он и не нужен. Ветка `--loader parquet` (прямое чтение
`python/{train,validation,test}-00000-of-00001.parquet`) реализована как запасная.
Размеры официальных сплитов: train 412 178, validation 23 107, test 22 176 — всего **457 461** строка.

### Сколько отсеял каждый фильтр (полный прогон, 457 461 строка)

| фильтр | отсеяно | доля |
|---|---:|---:|
| `too_long` — код длиннее 512 токенов `multilingual-e5-base` | 33 795 | 7.39 % |
| `dup_query` — такой же нормализованный запрос уже встречался (глобально по сплитам) | 33 303 | 7.28 % |
| `short_docstring` — в docstring меньше 3 слов | 7 788 | 1.70 % |
| `short_query` — в извлечённом запросе меньше 3 слов | 6 134 | 1.34 % |
| `unparsable_code` — код не разбирается `ast.parse` | 4 367 | 0.95 % |
| `query_in_code` — запрос всё ещё встречается в коде (утечка) | 389 | 0.09 % |
| `dup_code` — такой же нормализованный код уже встречался | 12 | 0.003 % |
| **итого отсеяно** | **85 788** | **18.76 %** |
| **осталось после очистки** | **371 673** | **81.24 %** |

Отдельно: `docstring_only_body = 95` — столько функций имели тело, состоящее только из
docstring; для них на место docstring вставлен `pass`, иначе получился бы невалидный Python.

Счётчик `no_docstring` не сработал ни разу: в CodeSearchNet поле `func_documentation_string`
непустое всегда (проверено на 5000 строк выборки). При этом docstring **в коде** есть у 99.7 %
функций — именно его и вырезает AST-шаг.

### Размеры сплитов (файлы в `$SEMSEARCH_HOME/data/`, по умолчанию `code/var/data/`)

| файл | строк | содержимое |
|---|---:|---|
| `train.jsonl` | 100 000 | пары «запрос → функция» (лимит `DataConfig.train_size`; после очистки доступно 334 948) |
| `val_pool.jsonl` | 5 000 | корпус кандидатов val (N_val = 5000) |
| `val_queries.jsonl` | 2 000 | запросы val, взятые из этого корпуса |
| `test_pool.jsonl` | 5 000 | корпус кандидатов test (N_test = 5000) |
| `test_queries.jsonl` | 1 000 | запросы test, взятые из этого корпуса |

Поля: `id`, `query_en`, `code`, `func_name`, `url`, `repo`. Время сборки: **416 с** (7 минут) на CPU.

Характеристики данных: длина запроса — медиана 9 слов, среднее 11.2, максимум 30 (ограничение
`max_query_words`); длина кода — медиана 30 слов, среднее 42.2. Уникальных репозиториев:
9 701 в train, 512 в test-корпусе.

### Проверки инвариантов (выполнены кодом и тестами)

- сплиты не пересекаются по `id`, по нормализованному коду и по нормализованному запросу
  (дедупликация глобальная, плюс `check_no_overlap` после сборки — бросает `AssertionError`);
- ни в одной записи `train/val_pool/test_pool` запрос не встречается в коде
  (проверено на всех 110 000 записей);
- запросы val/test — подмножество своего корпуса по `id`;
- docstring в коде не остался: `docstring-маркеры` (`"""`, `'''`) не встречаются ни в одной записи;
- официальные сплиты CodeSearchNet разделены **по репозиториям**: общих репозиториев между
  train/validation/test нет (0 при 12 231 / 679 / 680 репозиториях) — утечки на уровне проекта нет.

### Перевод (локально, CPU, 200 запросов из тест-набора)

| направление | переведено моделью | из кэша | загрузка модели | перевод | скорость |
|---|---:|---:|---:|---:|---:|
| EN→RU (`opus-mt-en-ru`) | 200 | 0 | 10.03 с | 290.84 с | 0.69 предл./с |
| RU→EN (`opus-mt-ru-en`) | 200 | 0 | 10.3 с | 219.68 с | 0.91 предл./с |

Возобновление проверено повторным запуском той же команды: `from_cache: 200`,
`translated_by_model: 0`, `model_load_sec: 0.0`, `elapsed_sec: 0.0` — модель не загружалась вообще.

**Оценка (не измерение)**: 100 000 обучающих запросов на этом CPU ≈ 100 000 / 0.69 ≈ 145 000 с ≈ **40 часов**.
На GPU Colab (T4/L4) ожидается ускорение в 15–40 раз, то есть ориентировочно **1–3 часа** —
проверить это нужно до того, как строить на этом датасете дообучение.

### Качество перевода (первые строки `results/translation_sample.csv`, оценка — за автором)

| EN | RU (opus-mt) | что не так |
|---|---|---|
| Log-joint probability given a state's log-probability and momentum. | Вероятность логарифмического совпадения с учетом вероятности логарифма и динамики. | `joint` → «совпадения» — смысловая ошибка |
| Create a union (and also shift if desired) of all input `Schedule`s. | Создать союз (а также, при желании, сменить) всех входных данных. | имя типа `Schedule` потеряно |
| Resolves an option value into options. | Определяет стоимость опциона в опционах. | `option` (параметр) переведено как «опцион» |
| `Convert XML to URL List.` | `Переименовать XML в URL List.` | `convert` → «переименовать» |

Обратный перевод RU→EN теряет технические термины сильнее:
`Перезагрузить функцию печати по умолчанию…` → `Reboot the default printing function…`.
Колонка `rating` в `translation_sample.csv` пустая — её должен заполнить человек (100 пар).

### Реальный вывод проверок

```
$ .venv/bin/python -m pytest
.......................                                                  [100%]
23 passed in 4.49s

$ .venv/bin/python -m ruff check .
All checks passed!

$ .venv/bin/python -m ruff format --check .
8 files already formatted

$ .venv/bin/python -m mypy
Success: no issues found in 6 source files
```

---

## Решения

Все решения — в `kursovaya-tex/code/DECISIONS.md`:

1. **Способ загрузки датасета** — `load_dataset(..., "python")` работает, parquet оставлен запасным (§1).
2. **Формат запроса** — первый абзац docstring, обрезка до 30 слов (§2).
3. **Удаление docstring** — AST, вырезание по байтовым смещениям, `ast.unparse` не используется,
   при пустом теле — `pass` (§3).
4. **Фильтр 512 токенов** — токенизатор `intfloat/multilingual-e5-base`, считается по коду без docstring (§4).
5. **Дедупликация** — по нормализованному коду и по нормализованному запросу, глобально между сплитами (§5).
6. **Состав сплитов** — официальные сплиты CodeSearchNet, корпус кандидатов N = 5000, запросы из него (§6).
7. **Модели перевода** — `Helsinki-NLP/opus-mt-en-ru` и `opus-mt-ru-en` (NLLB отвергнут как избыточный) (§7).
8. **Круговой перевод** — EN→RU→EN даёт оптимистичную оценку для BM25 по RU→EN; выводы о гибриде
   делаются по ручным запросам (§8).
9. **Пути и окружение** — `SEMSEARCH_HOME`, uv-venv, зафиксированные версии (§9).
10. **Протокола эмбеддера в Шаге 1 нет** — правило №9 AGENTS.md (§10).

---

## Проблемы / вопросы

1. **Полный перевод train не сделан** — локально это ~40 часов на CPU. Нужен прогон в Colab на GPU
   перед тем, как считать, что датасет готов к дообучению.
2. **Качество перевода не оценено человеком** — `results/translation_sample.csv` готов (100 пар),
   колонка `rating` пустая. Это главный риск для качества всего RU-датасета. Если оценка окажется
   низкой, план Б: NLLB-200 (`facebook/nllb-200-distilled-600M`) или LLM через API.
3. **Ручных запросов 10 из 100** — нужны ещё ≥ 40 (цель 100) от автора. Пул на 150 функций
   выгружен в `results/manual_pool.md`; формат записи — `{"id": "<id>", "query_ru": "<запрос>"}`.
4. **Обратный перевод искажает термины** — «Reboot the default printing function» вместо
   «перегрузка функции print». Для BM25 по RU→EN это дополнительно (сверх кругового перевода)
   занижает лексическое совпадение. Нужно решить, что считать эталоном этого варианта.
5. **Код курсовой и результаты вне git** — корневой `.gitignore` репозитория исключает
   `kursovaya-*/code/*` и `kursovaya-*/results/*`. Ничего не коммитилось и не пушилось, как и
   требовалось, но перед сдачей решается: изменить `.gitignore` или сдавать архивом.
6. **`ast.parse` на чужом коде шумит** — CSN содержит код с `invalid escape sequence`
   (SyntaxWarning на каждый такой файл). На результат не влияет, но при большом прогоне лог
   засоряется; при необходимости глушится `warnings`.

---

## Что нужно разобрать студенту перед защитой

1. **Удаление docstring через AST** — почему нельзя `ast.unparse` (теряет комментарии и
   форматирование), почему смещения байтовые, а не символьные (`ast.col_offset` считает байты
   UTF-8), что делать с телом функции, состоящим только из docstring (`pass`), и почему
   номера строк сохраняются. Нужно уметь объяснить своими словами, без кода на экране.
2. **Нормализация** — зачем приводить код и запрос к каноническому виду перед дедупликацией,
   почему дедупликация запроса **глобальная** (иначе одинаковый docstring уедет из train в test
   и тест станет бессмысленным), почему «у каждого запроса ровно один правильный ответ»
   обеспечивается именно так.
3. **Кэш перевода и возобновление** — как устроен ключ (хэш текста + имя модели), почему
   переводы пишутся в jsonl по ходу (а не в конце), почему модель не загружается, если всё
   уже переведено, и как это даёт устойчивость к обрыву сессии в Colab.
4. **Смещение оценки из-за кругового перевода** — почему BM25 по RU→EN на машинных переводах
   даёт верхнюю оценку, а не честную метрику, и почему выводы о гибриде должны опираться
   на ручные запросы.
5. **Числа в отчёте** — любую цифру из таблиц выше нужно уметь воспроизвести командой,
   которая приведена в разделе «Как запустить».
