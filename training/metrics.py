"""Binary detection metrics with counts and Wilson 95% score intervals."""
from __future__ import annotations

import math

Z95 = 1.959963984540054  # two-sided 95% normal quantile


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion k/n; None when n == 0."""
    if n == 0:
        return None
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def rate(k: int, n: int) -> dict:
    ci = wilson(k, n)
    return {"value": (k / n) if n else None, "numerator": k, "denominator": n,
            "ci95": list(ci) if ci else None, "ci_method": "wilson"}


def confusion(labels: list[int], preds: list[int]) -> dict:
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def summarize(labels: list[int], preds: list[int]) -> dict:
    c = confusion(labels, preds)
    return {
        "n": len(labels), "positives": c["tp"] + c["fn"], "negatives": c["fp"] + c["tn"], "confusion": c,
        "precision": rate(c["tp"], c["tp"] + c["fp"]),
        "recall": rate(c["tp"], c["tp"] + c["fn"]),
        "fpr": rate(c["fp"], c["fp"] + c["tn"]),
    }
