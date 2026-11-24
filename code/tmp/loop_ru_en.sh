#!/bin/zsh
# Перезапускать перевод RU→EN, пока не останется непереведённых текстов.
CODE=/mnt/proj/proj/study/study-ensemble/kursovaya-tex/code
cd "$CODE" || exit 1
for attempt in {1..40}; do
  PYTHONPATH=. .venv/bin/python "$CODE/tmp/translate_ru_en_1000.py" >> "$CODE/tmp/loop_ru_en.log" 2>&1
  if grep -q "test_queries_nllb_ru_en_1000.jsonl: 1000 строк" "$CODE/tmp/loop_ru_en.log"; then
    echo "done after $attempt attempt(s)" >> "$CODE/tmp/loop_ru_en.log"
    exit 0
  fi
  sleep 5
done
echo "gave up" >> "$CODE/tmp/loop_ru_en.log"
exit 1
