"""Тесты метрик ранжирования (Шаг 2): ответы известны заранее."""

from __future__ import annotations

import math

import pytest

from semsearch.core.metrics import METRICS, evaluate, ndcg_at_k, recall_at_k, reciprocal_rank


def test_reciprocal_rank_first_position() -> None:
    assert reciprocal_rank(["a", "b"], "a", 10) == 1.0


def test_reciprocal_rank_third_position() -> None:
    assert reciprocal_rank(["a", "b", "c"], "c", 10) == pytest.approx(1 / 3)


def test_reciprocal_rank_outside_k_is_zero() -> None:
    ranking = ["a", "b", "c", "d"]
    assert reciprocal_rank(ranking, "d", 3) == 0.0


def test_reciprocal_rank_missing_document_is_miss() -> None:
    # BM25 на русском тексте не находит ответ — это промах, а не исключение
    assert reciprocal_rank(["a", "b"], "z", 10) == 0.0


def test_recall_boundaries() -> None:
    ranking = ["x", "y", "z", "answer"]
    assert recall_at_k(ranking, "answer", 1) == 0.0
    assert recall_at_k(ranking, "answer", 5) == 1.0
    assert recall_at_k(ranking, "answer", 3) == 0.0


def test_ndcg_matches_hand_computation() -> None:
    assert ndcg_at_k(["a", "b", "c"], "a", 10) == 1.0
    assert ndcg_at_k(["a", "b", "c"], "c", 10) == pytest.approx(1 / math.log2(4))
    assert ndcg_at_k(["a"], "z", 10) == 0.0


def test_evaluate_average_on_known_example() -> None:
    # 4 запроса: позиции ответа 1, 2, 5 и промах
    rankings = [["r1", "x"], ["x", "r2"], ["a", "b", "c", "d", "r5"], ["x", "y"]]
    relevant = ["r1", "r2", "r5", "miss"]
    metrics = evaluate(rankings, relevant, k=10)
    assert metrics["MRR@10"] == pytest.approx((1 + 0.5 + 0.2 + 0) / 4)
    assert metrics["Recall@1"] == pytest.approx(0.25)
    assert metrics["Recall@5"] == pytest.approx(0.75)
    assert metrics["Recall@10"] == pytest.approx(0.75)
    assert metrics["nDCG@10"] == pytest.approx((1 + 1 / math.log2(3) + 1 / math.log2(6)) / 4)


def test_evaluate_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="совпадать"):
        evaluate([["a"]], ["a", "b"])


def test_evaluate_rejects_empty_set() -> None:
    with pytest.raises(ValueError, match="пустой"):
        evaluate([], [])


def test_metric_names() -> None:
    metrics = evaluate([["a"]], ["a"])
    assert tuple(metrics) == METRICS
