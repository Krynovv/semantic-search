# Шаг 1б — слепое сравнение переводчиков opus-mt и NLLB

Дата: 2026-11-15. Код: `kursovaya-tex/code` (пакет `semsearch`).

## Что сделано

1. **В `semsearch/data/translate.py` добавлена модель NLLB** — минимальным изменением:
   - `ModelSpec` (имя модели + языковые коды) и словарь `DIRECTIONS` с направлениями
     `en2ru`, `ru2en` (opus-mt) и `nllb-en2ru`, `nllb-ru2en` (NLLB-600M);
   - бэкенд переименован `MarianBackend` → `Seq2SeqBackend` и получил два необязательных
     параметра `src_lang` / `tgt_lang`: язык передаётся токенизатору
     (`AutoTokenizer.from_pretrained(..., src_lang=..., tgt_lang=...)`), а язык перевода —
     в `forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang)`;
   - ключ кэша по-прежнему «хэш текста + имя модели»: у opus-mt имя модели не изменилось,
     у NLLB в идентификатор кэша добавлено направление (`ModelSpec.cache_id`) — иначе
     EN→RU и RU→EN одной и той же моделью NLLB перезатирали бы друг друга в кэше.
   Остальной код (CLI, кэш, формат вывода) не менялся.
2. **100 запросов из `results/translation_sample.csv` переведены моделью NLLB** на CPU,
   beams = 4, batch = 16 (те же параметры, что у opus-mt).
3. **Собран слепой файл** `results/translation_blind.csv` (100 строк: `id`, `query_en`,
   `variant_a`, `variant_b`, `rating_a`, `rating_b`; оценки пустые) и отдельный ключ
   `results/translation_blind_key.csv` (`id`, `a_is`). Кодировка utf-8-sig — файл
   открывается в Excel/LibreOffice без «кракозябр».
4. **Новый модуль `semsearch/data/blind.py`**: `make` — сборка слепого файла и ключа
   (порядок вариантов задаётся `random.Random(42)`, в `variant_a` opus-mt попадает
   ровно в 50 строках из 100), `score` — сведение заполненных оценок: средние баллы,
   распределение 1–5, число строк «NLLB лучше / хуже / равно» и p-значение точного
   двустороннего **теста знаков**. scipy в окружении нет, поэтому биномиальная сумма
   посчитана вручную (`math.comb`, 5 строк).
5. **Тесты** (`tests/test_blind.py` + 2 теста в `tests/test_data.py`): кэш разных моделей
   не смешивается, ключ NLLB различает направления, сборка слепого файла восстанавливается
   по ключу, сведение оценок проверено на ручном примере (4 строки, ответ известен заранее),
   тест знаков симметричен.
6. Запись о решении — в `code/DECISIONS.md` §11 (пункт §7 «Модели перевода» дополнен
   ссылкой на пересмотр решения).

## Команды

Всё выполняется из `kursovaya-tex/code`, venv — `.venv`.

```bash
# 1) перевод 100 запросов моделью NLLB (первый запуск скачивает модель ~2,5 ГБ)
.venv/bin/python -m semsearch.data.translate \
  --input tmp/queries_100.jsonl --output tmp/queries_100_nllb.jsonl \
  --field query_en --out-field query_ru \
  --direction nllb-en2ru --device cpu

# 2) сборка слепого файла и ключа
.venv/bin/python -m semsearch.data.blind make \
  --opus-csv ../results/translation_sample.csv \
  --nllb-jsonl tmp/queries_100_nllb.jsonl \
  --out-csv ../results/translation_blind.csv \
  --key-csv ../results/translation_blind_key.csv

# 3) после заполнения оценок оценщиком — сведение
.venv/bin/python -m semsearch.data.blind score \
  --blind-csv ../results/translation_blind.csv \
  --key-csv ../results/translation_blind_key.csv

# 4) проверки
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy
.venv/bin/python -m pytest
```

`tmp/queries_100.jsonl` — те же 100 `query_en` из `results/translation_sample.csv`
(порядок сохранён), собранные в jsonl, потому что `translate.py` работает с jsonl.

## Время перевода 100 запросов NLLB

CPU (torch 2.15.0+cpu), batch = 16, beams = 4, `max_new_tokens` = 256:

| величина | значение |
|---|---|
| скачивание модели + загрузка весов (`model_load_sec`) | 207 с (веса 2,46 ГБ) |
| перевод 100 запросов (`elapsed_sec`) | **1164 с ≈ 19,4 мин** |
| скорость | 0,086 запроса/с (11,6 с на запрос) |
| весь запуск (`time`) | 19 мин 28 с |

Для сравнения: opus-mt на тех же 100 запросах — 0,69 запроса/с (≈ 1,45 с на запрос),
то есть **NLLB-600M на CPU примерно в 8 раз медленнее**. Если NLLB выиграет оценку,
полный прогон (100 000 запросов) на CPU нереален (≈ 13 суток) — нужен GPU (Colab),
как и планировалось для остальных тяжёлых шагов.

Замечания по ходу: transformers предупреждает, что у NLLB в `generation_config` задан
`max_length=200`; наш `max_new_tokens=256` имеет приоритет (это одинаково для обеих
моделей, сравнение остаётся честным). Модель лицензирована **cc-by-nc-4.0**
(некоммерческая) — для учебной курсовой допустимо, для промышленного применения нет.

## 5 примеров пар opus-mt / NLLB (иллюстрация, без вывода о качестве)

Взяты из готового слепого файла; порядок вариантов в нём случайный, поэтому «opus» и
«NLLB» подписаны здесь явно, а оценок и итогового вердикта здесь нет.

| # | query_en | opus-mt | NLLB-600M |
|---|---|---|---|
| 1 | Log-joint probability given a state's log-probability and momentum. | Вероятность логарифмического совпадения с учетом вероятности логарифма и динамики. | Лог-соединенная вероятность с учетом лог-вероятности и импульса состояния. |
| 6 | Return dict describing the context of this package | Задикт возврата, описывающий контекст этого пакета | Возвратный текст, описывающий контекст данного пакета |
| 10 | Save the current database into the inactive-db.json file. | Сохранить текущую базу данных в файл неактивного db.json. | Сохранить текущую базу данных в файле inactive-db.json. |
| 11 | Adds argument for config to existing argparser | Добавляет аргумент для конига к существующему argparser | Добавляет аргумент для конфигурации к существующему argparser |
| 13 | Returns index transforms for ``name``. | Индекс доходности преобразуется для "имя". | Индекс возврата преобразуется для ``name``. |

Видно, что ошибки двух переводчиков разной природы: opus-mt ломает идентификаторы и
имена файлов (`inactive-db.json` → «неактивного db.json», ``name`` → «имя», `config` →
«конига»), NLLB термины и разметку сохраняет чаще, но переводит их описательно
(`dict` → «текст», `option` → «опцион», `joint` → «соединенная»).

## Проверки кода

```
$ .venv/bin/python -m ruff check .
All checks passed!

$ .venv/bin/python -m mypy
Success: no issues found in 8 source files

$ .venv/bin/python -m pytest
...............................                                          [100%]
31 passed in 4.79s
```

(Было 23 теста, стало 31: +6 в `tests/test_blind.py`, +2 в `tests/test_data.py`.)

## Что дальше (шаг для студента)

1. Открыть `results/translation_blind.csv` и проставить оценки 1–5 в `rating_a` и `rating_b`
   для всех 100 строк (критерий шкалы стоит записать — в прошлый раз он не был зафиксирован).
2. **`results/translation_blind_key.csv` до конца оценки не открывать** — иначе сравнение
   перестаёт быть слепым.
3. После заполнения оценок — команда `blind score` (п. 3 выше): она сама сопоставит
   варианты с ключом и посчитает средние, распределение и p-значение.
4. Оценки в `results/translation_sample.csv` (средняя по opus-mt) в отчёт не включены:
   отчёт не делает выводов до заполнения слепой оценки.

## Ограничения

- Сравнение одностороннее по числу оценок: 100 пар, один оценщик → выводы о разнице
  нужно читать вместе с p-значением, а не только со средними.
- Ключ `translation_blind_key.csv` лежит рядом со слепым файлом; «не открывать» — это
  договорённость, а не техническая защита.
- Перевод NLLB сделан только для EN→RU (этого требует задача); обратное направление
  `nllb-ru2en` в коде поддержано, но не измерялось.
- Вариант LLM-API как третий переводчик не проверялся (нужен ключ, цена не подтверждена).
