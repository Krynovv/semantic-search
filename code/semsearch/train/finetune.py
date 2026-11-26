"""Дообучение multilingual-e5-base контрастивным loss (Шаг 3).

Обучение идёт в Colab (`notebooks/train_colab.ipynb`): на локальной машине
тренер transformers/accelerate молча выходит после первого шага
(research-log.md, 2026-11-23). Эксперименты различаются только тем, какой
текст стоит в anchor'е:

- `ru-only` — anchor = query_ru (перевод NLLB), 100k пар;
- `mix-en`  — 50k пар с anchor = query_ru и 50k с anchor = query_en
  (те же коды, другой язык запроса): проверка, снижает ли второй язык
  забывание английского при том же числе пар.

Остальное одинаково: MultipleNegativesRankingLoss (in-batch негативы),
префиксы e5 `query: ` / `passage: `, число шагов (epochs × пар / batch).
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer import losses
from transformers import TrainerCallback

from semsearch.config import EvalConfig
from semsearch.train.evaluate import EmbeddingEvaluator, resolve_model

EXPERIMENTS = ("ru-only", "mix-en")


@dataclass(frozen=True)
class TrainConfig:
    """Параметры дообучения: одинаковые для эксперимента (а) и (б)."""

    lr: float = 1e-5
    batch_size: int = 8
    epochs: int = 2
    warmup_steps: int = 500
    max_seq_length: int = 256
    seed: int = 42
    experiment: str = "ru-only"
    eval_every: int = 500
    checkpoint_every: int = 500
    save_dir: Path = Path("models") / "ru-only"
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.experiment not in EXPERIMENTS:
            raise ValueError(f"experiment: {self.experiment!r}, ожидалось {EXPERIMENTS}")
        if min(self.lr, self.batch_size, self.epochs) <= 0:
            raise ValueError("lr, batch_size и epochs должны быть > 0")

    def to_dict(self) -> dict[str, Any]:
        """Конфиг для сохранения рядом с моделью и логом."""
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Прочитать jsonl."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_training_data(
    train_path: Path, experiment: str, limit: int | None = None
) -> list[InputExample]:
    """Пары (запрос, код) для выбранного эксперимента.

    `limit` — первые N пар исходного файла (отладка на маленьком наборе);
    для mix-en это N/2 пар на язык.
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(f"experiment: {experiment!r}, ожидалось {EXPERIMENTS}")
    rows = load_jsonl(train_path)
    if limit is not None:
        rows = rows[:limit]
    examples = []
    for row in rows:
        queries: Sequence[str] = [row["query_ru"]]
        if experiment == "mix-en":
            queries = [row["query_ru"], row["query_en"]]
        for query in queries:
            examples.append(InputExample(texts=[query, row["code"]]))
    return examples


def to_dataset(examples: Sequence[InputExample], seed: int = 0) -> dict[str, list[str]]:
    """Словарь для `datasets.Dataset.from_dict`: колонки anchor и positive.

    Тексты с префиксами e5. Перемешивать обязательно: в файле RU- и EN-пары
    одного кода идут подряд, и в одном батче позитив одного примера
    оказывается позитивом другого — loss штрафует правильный ответ.
    """
    cfg = EvalConfig()
    pairs = [
        (cfg.query_prefix + texts[0], cfg.passage_prefix + texts[1])
        for texts in (ex.texts or [] for ex in examples)
    ]
    random.Random(seed).shuffle(pairs)
    return {
        "anchor": [anchor for anchor, _ in pairs],
        "positive": [positive for _, positive in pairs],
    }


class _Trainer(SentenceTransformerTrainer):
    """Тренер без карточки модели: нужны только веса.

    Штатный тренер при каждом сохранении собирает карточку (кодирует
    запросы по одному) и пишет README.md — лишние минуты на чекпоинт.
    """

    def add_model_card_callback(self, default_args_dict: dict[str, Any]) -> None:
        del default_args_dict

    def _save(self, output_dir: str | None = None, state_dict: Any = None) -> None:
        del state_dict
        self.model.save_pretrained(output_dir or str(self.args.output_dir), create_model_card=False)


class _CsvLogger:
    """Лог обучения: строка `step=0` (до обучения), затем каждые N шагов."""

    columns = ("step", "loss", "mrr_en", "recall_en", "mrr_ru", "recall_ru")

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: list[dict[str, float | int | None]] = []

    def add(self, step: int, **metrics: float | int | None) -> None:
        self.rows.append({"step": step, **{c: metrics.get(c) for c in self.columns[1:]}})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.columns))
            writer.writeheader()
            writer.writerows(self.rows)


class _ValLogger(TrainerCallback):
    """Каждые `every` шагов: MRR@10/Recall@10 на EN и RU по test-корпусу и loss."""

    def __init__(
        self,
        evaluator: EmbeddingEvaluator,
        logger: _CsvLogger,
        every: int,
        model: SentenceTransformer,
    ) -> None:
        self.evaluator = evaluator
        self.logger = logger
        self.every = every
        self.model = model
        self.substep_loss: float | None = None

    # def on_substep_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        
    #     del args, control, kwargs
    #     self.substep_loss = state.loss

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, control, kwargs
        if state.global_step % self.every != 0:
            return
        
        recent_loss = None
        for log in reversed(state.log_history):
            if "loss" in log:
                recent_loss = log["loss"]
                break
            elif "train_loss" in log:
                recent_loss = log["train_loss"]
                break
        
        metrics = self.evaluator(self.model)
                
        # Записываем в CSV
        self.logger.add(state.global_step, loss=recent_loss, **metrics)
                
        # Форматируем вывод для консоли
        loss_str = f"{recent_loss:.4f}" if recent_loss is not None else "N/A"
        print(f"[шаг {state.global_step}] loss={loss_str}, {metrics}", flush=True)


def train_finetune(
    config: TrainConfig,
    train_path: Path,
    queries_path: Path,
    pool_path: Path,
    manual_queries_path: Path | None = None,
    limit: int | None = None,
) -> dict[str, float]:
    """Обучить, сохранить модель в `config.save_dir`; вернуть итоговые метрики.

    Лог `train_log_<experiment>.csv` — в родительском каталоге `save_dir`,
    `train_config.json` — рядом с моделью.
    """
    torch.manual_seed(config.seed)
    data = to_dataset(prepare_training_data(train_path, config.experiment, limit), config.seed)
    model = resolve_model(EvalConfig().model_name, config.device)
    model.max_seq_length = config.max_seq_length

    evaluator = EmbeddingEvaluator(queries_path, pool_path, manual_queries_path)
    logger = _CsvLogger(config.save_dir.parent / f"train_log_{config.experiment}.csv")
    logger.add(0, **evaluator(model))

    args = SentenceTransformerTrainingArguments(
        output_dir=str(config.save_dir / "checkpoints"),
        learning_rate=config.lr,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=4,
        fp16=True,
        fp16_full_eval=False,
        # activation_checkpointing=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_bnb_8bit",
        num_train_epochs=config.epochs,
        warmup_steps=config.warmup_steps,
        logging_steps=config.eval_every,
        save_strategy="steps",
        save_steps=config.checkpoint_every,
        save_total_limit=2,  # чекпоинт ~1.1 ГБ: нужен для возобновления, старые не нужны
        load_best_model_at_end=False,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )
    trainer = _Trainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_dict(data),
        loss=losses.MultipleNegativesRankingLoss(model),
        callbacks=[_ValLogger(evaluator, logger, config.eval_every, model)],
    )
    trainer.train()
    model.save_pretrained(str(config.save_dir), create_model_card=False)
    (config.save_dir / "train_config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    history = trainer.state.log_history
    final_loss = next((row["loss"] for row in reversed(history) if "loss" in row), None)
    step = trainer.state.global_step
    print(f"обучение закончено, шаг {step}; строю индекс для итоговой оценки", flush=True)
    final = evaluator(model)
    logger.add(step, loss=final_loss, **final)
    return final
