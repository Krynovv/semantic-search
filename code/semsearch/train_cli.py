"""CLI Шага 3: дообучение (Colab) и оценка сохранённой модели.

    # Colab или GPU-машина
    python -m semsearch.train_cli finetune --experiment mix-en --train-data ... \\
        --queries ... --pool ... --manual ... --save-dir models/mix-en

    # локально, CPU (~1 ч на 5000 функций)
    python -m semsearch.train_cli evaluate --model models/mix-en \\
        --queries var/data/test_queries_nllb_ru_1000.jsonl \\
        --pool var/data/test_pool.jsonl --manual ../results/manual_queries.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semsearch.config import data_dir
from semsearch.train.evaluate import DEFAULT_MANUAL_QUERIES, evaluate_model
from semsearch.train.finetune import TrainConfig, train_finetune


def _finetune(args: argparse.Namespace) -> None:
    config = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        experiment=args.experiment,
        eval_every=args.eval_every,
        save_dir=args.save_dir,
        device=args.device,
    )
    metrics = train_finetune(config, args.train_data, args.queries, args.pool, args.manual)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / f"metrics_{args.experiment}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False))


def _evaluate(args: argparse.Namespace) -> None:
    metrics = evaluate_model(args.model, args.queries, args.pool, args.manual)
    print(json.dumps(metrics, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    """Разбор командной строки."""
    parser = argparse.ArgumentParser(prog="python -m semsearch.train_cli")
    commands = parser.add_subparsers(dest="command", required=True)

    finetune = commands.add_parser("finetune", help="дообучить e5-base")
    finetune.add_argument("--experiment", choices=("ru-only", "mix-en"), default="ru-only")
    finetune.add_argument("--train-data", type=Path, default=data_dir() / "train_with_ru.jsonl")
    queries = data_dir() / "test_queries_nllb_ru_1000.jsonl"
    finetune.add_argument("--queries", type=Path, default=queries)
    finetune.add_argument("--pool", type=Path, default=data_dir() / "test_pool.jsonl")
    finetune.add_argument("--manual", type=Path, default=DEFAULT_MANUAL_QUERIES)
    finetune.add_argument("--save-dir", type=Path, default=Path("models") / "ru-only")
    finetune.add_argument("--epochs", type=int, default=3)
    finetune.add_argument("--batch-size", type=int, default=32)
    finetune.add_argument("--lr", type=float, default=1e-5)
    finetune.add_argument("--eval-every", type=int, default=500)
    finetune.add_argument("--device", default="cuda")
    finetune.set_defaults(run=_finetune)

    evaluate = commands.add_parser("evaluate", help="оценить сохранённую модель")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--queries", type=Path, default=queries)
    evaluate.add_argument("--pool", type=Path, default=data_dir() / "test_pool.jsonl")
    evaluate.add_argument("--manual", type=Path, default=DEFAULT_MANUAL_QUERIES)
    evaluate.set_defaults(run=_evaluate)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
