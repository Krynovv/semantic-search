# Шаг 1в — полный перевод запросов NLLB-600M в Colab (ноутбук)

Дата: 2026-11-20. Ноутбук: `kursovaya-tex/notebooks/translate_nllb.ipynb`.
Основание: слепое сравнение (`results/step1b-report.md`) — NLLB-600M μ=3,36 против 2,67
у opus-mt, p<0,0001; на CPU NLLB даёт 0,086 запроса/с, то есть 100k ≈ 13 суток.

## Что делает ноутбук

| шаг | что происходит |
|---|---|
| 1. Установка | `transformers>=5.0`, `sentencepiece`, `tqdm`, свежий `setuptools`; torch в Colab уже с CUDA |
| 2. GPU | проверка `torch.cuda.is_available()`, имя GPU, свободная/полная память |
| 3. Загрузка | диалог выбора файлов (`semsearch.zip`, `csn_prepared.tar.gz` или архив прошлого прогона); распаковка в `/content/cache/data/` и `/content/semsearch/`, проверка, что jsonl на месте |
| 4. Модель | `Seq2SeqBackend` из `semsearch.data.translate` (NLLB-600M, `cuda`, batch 16, beams 4), замер времени загрузки; модель держится в памяти и переиспользуется |
| 5–6. train | 100k: `query_en` → `query_ru` (`nllb-en2ru`), затем `query_ru` → `query_en_back` (`nllb-ru2en`) |
| 7. val/test | `val_queries` (2k) и `test_queries` (1k) — оба направления |
| 8. Отчёт | `/content/results/translation_report.json`: GPU, время загрузки модели, по каждому файлу — N, из кэша, переведено моделью, время, запр./с; таблица и размеры файлов печатаются |
| 9. Сохранение | `nllb_translation_results.zip` (`cache/`, `results/`, `semsearch/`) скачивается на компьютер |

Код не переписан: ноутбук распаковывает копию `kursovaya-tex/code/semsearch/` (архив `semsearch.zip`, вложенная
папка `kursovaya-tex/` тоже работает) и вызывает
`translate_file` из `semsearch/data/translate.py`. Направление NLLB — это `DIRECTIONS`
(`nllb-en2ru` / `nllb-ru2en`, язык — в токенизатор, целевой — в `forced_bos_token_id`).
Кэш и возобновление — готовые `TranslationCache` / `CachedTranslator`: ключ = sha256
(модель + направление + текст), запись дописывается после каждого файла с flush.

Единственное, что ноутбук делает сам, — не даёт перезагружать 2,5 ГБ весов 6 раз:
на время вызова `translate_file` конструктор `Seq2SeqBackend` в модуле подменяется на
уже загруженный бэкенд (две строки вокруг вызова, функция `translate` в ячейке 4).

Формат выхода: копия входного jsonl (`id`, `query_en`, `code`, `func_name`, `url`, `repo`)
плюс `query_ru` и `query_en_back`.

## Как запускать

```bash
# 1) подготовить архивы (из корня проекта study-ensemble)
cd kursovaya-tex/code
zip -r ../notebooks/semsearch.zip semsearch
tar czf ../notebooks/csn_prepared.tar.gz -C var/data train.jsonl val_queries.jsonl test_queries.jsonl
```

2. Colab → File → Upload notebook → `translate_nllb.ipynb`; Runtime → Change runtime type → **T4 GPU**.
3. Запустить ячейку «Загрузка файлов» и выбрать оба архива; затем Runtime → Run all.
4. В конце скачивается `nllb_translation_results.zip` → положить в `kursovaya-tex/results/`.
5. Продолжение после обрыва: загрузить этот архив в ту же ячейку загрузки — данные и кэш
   восстановятся, пересчитаются только непереведённые тексты.

## Оценочное время

| этап | T4 (оценка) |
|---|---|
| скачивание и загрузка NLLB-600M (~2,5 ГБ) | 1–2 мин |
| train EN→RU, 100k (batch 16, beams 4) | ~1,5–2,5 ч |
| train RU→EN, 100k | ~1,5–2,5 ч |
| val + test, 6k текстов в обе стороны | < 10 мин |
| **всего** | **3–5 ч** |

Оценка получена пересчётом замера CPU (0,086 запроса/с) на ускорение GPU ~50–100 раз;
на T4 **не измерено** — первая же прогонная строка отчёта даст настоящие запр./с.
Бесплатная сессия Colab живёт 3–4 ч, поэтому ячейку «Сохранить результаты» стоит
запускать вручную в середине прогона, а при обрыве — продолжать из кэша.

## Проверено локально (без Colab, на CPU)

Скрипт `tmp/run_notebook_local.py` исполняет все код-ячейки ноутбука на уменьшенных данных
(20/5/3 строки), с подменёнными `google.colab.files`, `torch.cuda` и переводчиком
(детерминированная заглушка вместо NLLB):

- цепочка проходит целиком: распаковка архивов → перевод 6 прогонов → `translation_report.json` → архив;
- в выходных jsonl исходные поля сохранены, добавлены `query_ru` и `query_en_back`;
- два кэш-файла — отдельно EN→RU и RU→EN (`nllb-200-distilled-600M-eng_Latn-rus_Cyrl.jsonl`,
  `...-rus_Cyrl-eng_Latn.jsonl`), направления не смешиваются;
- повторный запуск с нуля, когда загружен только `nllb_translation_results.zip` (имитация
  обрыва сессии): все 6 прогонов берутся из кэша (0 обращений к модели) — продолжение работает.

Реальный GPU-прогон NLLB в Colab не выполнялся (в этой среде нет GPU) — первый запуск
делает студент; настоящие цифры времени появятся в `translation_report.json`.

Ограничения:
- кэш и jsonl в Colab живут в `/content/`, поэтому при обрыве их надо сохранить заранее
  (архив), иначе прогон начнётся с нуля;
- RU→EN — круговой перевод (BM25 по нему даёт завышенную оценку, `DECISIONS.md` §8);
- лицензия NLLB cc-by-nc-4.0 (учебное использование допустимо).
