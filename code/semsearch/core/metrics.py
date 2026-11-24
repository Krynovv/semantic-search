"""Метрики ранжирования: MRR@k, Recall@k, nDCG@k (Шаг 2).

Ранжирование — список id документов от лучшего к худшему. Релевантность
бинарная (у каждого запроса ровно один правильный ответ), поэтому nDCG@k
для попадания на позиции r (1-based) равна 1 / log2(r + 1). Правильный ответ,
поставленный за пределами выдачи, — это промах (0), а не ошибка: так BM25
на русском тексте даёт честный ноль.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

DEFAULT_K = 10


def reciprocal_rank(ranking: Sequence[str], relevant: str, k: int = DEFAULT_K) -> float:
    """MRR для одного запроса: 1/позиция, если ответ в первых k, иначе 0."""
    for rank, doc_id in enumerate(ranking[:k], start=1):
        if doc_id == relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranking: Sequence[str], relevant: str, k: int = DEFAULT_K) -> float:
    """Recall@k: один правильный ответ — 1.0, если он в первых k, иначе 0."""
    return 1.0 if relevant in ranking[:k] else 0.0


def ndcg_at_k(ranking: Sequence[str], relevant: str, k: int = DEFAULT_K) -> float:
    """nDCG@k при бинарной релевантности (идеальный DCG — попадание на 1-е место)."""
    for rank, doc_id in enumerate(ranking[:k], start=1):
        if doc_id == relevant:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def evaluate(
    rankings: Sequence[Sequence[str]], relevant: Sequence[str], k: int | None = None
) -> dict[str, float]:
    """Средние метрики по всем запросам: MRR@k, Recall@1/5/10, nDCG@k.

    k — глубина выдачи (по умолчанию 10). Если глубина меньше 10, Recall@10
    совпадает с Recall по фактической глубине; для отчёта всегда k = 10.
    """
    if len(rankings) != len(relevant):
        raise ValueError("число выдач и правильных ответов должно совпадать")
    if not rankings:
        raise ValueError("пустой набор запросов")
    depth = DEFAULT_K if k is None else k
    pairs = list(zip(rankings, relevant, strict=True))
    n = len(pairs)

    def mean(func: Callable[[Sequence[str], str, int], float], top: int) -> float:
        return sum(func(ranking, rel, top) for ranking, rel in pairs) / n

    return {
        f"MRR@{depth}": mean(reciprocal_rank, depth),
        "Recall@1": mean(recall_at_k, 1),
        "Recall@5": mean(recall_at_k, 5),
        "Recall@10": mean(recall_at_k, 10),
        f"nDCG@{depth}": mean(ndcg_at_k, depth),
    }


METRICS = ("MRR@10", "Recall@1", "Recall@5", "Recall@10", "nDCG@10")
