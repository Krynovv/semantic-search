# Модели (обновляет worker после каждого обучения)

## ru-only
- Путь (содержимое не читать): `mnt/proj/proj/study/data/semsearch_download/ru-only`
- База: multilingual-e5-base, дообучение на CodeSearchNet Python (457k)
- Данные оценки: корпус test N=5000, 1000 запросов
- Как получена: команда запуска из `code/`, коммит `<hash>`
- Параметры: epochs=?, batch=?, lr=?, max_len=?, seed=?
- Железо и время: Colab, 200 мин
- Метрики: MRR@10=?, Recall@1/5/10=?, nDCG@10=?
- Сравнение с базовой моделью (без дообучения): ?
- Файл с метриками: `results/metrics_ru-only.json`
- Проверка: `du -sh <путь>` = ?, загрузка прошла: да/нет (дата)

## mix-en
- Путь (содержимое не читать): `mnt/proj/proj/study/data/semsearch_download/mix-en`
- База: multilingual-e5-base, дообучение на CodeSearchNet Python (457k)
- Данные оценки: корпус test N=5000, 1000 запросов
- Как получена: команда запуска из `code/`, коммит `<hash>`
- Параметры: epochs=?, batch=?, lr=?, max_len=?, seed=?
- Железо и время: Colab, 120 мин
- Метрики: MRR@10=?, Recall@1/5/10=?, nDCG@10=?
- Сравнение с базовой моделью (без дообучения): ?
- Файл с метриками: `results/metrics_mix-en.json`
- Проверка: `du -sh <путь>` = ?, загрузка прошла: да/нет (дата)
