"""Metric aggregation for conditional pipeline logs."""

from __future__ import annotations

from collections import defaultdict
import math

import numpy as np


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _safe_float(value):
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_accept(value) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "accept", "accepted", "true", "yes"}


def _is_reject(value) -> bool:
    text = str(value).strip().lower()
    return text in {"0", "reject", "rejected", "false", "no"}


def _roc_points(labels: list[int], scores: list[float]):
    if len(set(labels)) < 2:
        return [], []
    order = np.argsort(-np.asarray(scores))
    labels_arr = np.asarray(labels)[order]
    pos = max(1, int(np.sum(labels_arr == 1)))
    neg = max(1, int(np.sum(labels_arr == 0)))

    tpr = [0.0]
    fpr = [0.0]
    tp = 0
    fp = 0
    for label in labels_arr:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / pos)
        fpr.append(fp / neg)
    return fpr, tpr


def _auc(labels: list[int], scores: list[float]) -> float:
    fpr, tpr = _roc_points(labels, scores)
    if not fpr:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    area = 0.0
    for i in range(1, len(fpr)):
        area += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
    return float(area)


def _eer(labels: list[int], scores: list[float]) -> float:
    fpr, tpr = _roc_points(labels, scores)
    if not fpr:
        return 0.0
    fnr = 1.0 - np.asarray(tpr)
    fpr_arr = np.asarray(fpr)
    idx = int(np.nanargmin(np.abs(fnr - fpr_arr)))
    return float((fnr[idx] + fpr_arr[idx]) / 2.0)


def compute_group_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n_samples": 0,
            "n_active": 0,
            "n_deferred": 0,
            "FRR": 0.0,
            "FAR": 0.0,
            "FRR_active": 0.0,
            "FAR_active": 0.0,
            "FRR_with_defer_as_failure_for_genuine": 0.0,
            "EER": 0.0,
            "AUC": 0.0,
            "defer_rate": 0.0,
            "automation_rate": 0.0,
            "fast_path_rate": 0.0,
            "robust_path_rate": 0.0,
            "latency_mean": 0.0,
            "latency_std": 0.0,
            "latency_p95": 0.0,
            "ram_peak_mb": 0.0,
        }

    n = len(rows)
    deferred = [row for row in rows if _to_bool(row.get("deferred", False))]
    active = [row for row in rows if not _to_bool(row.get("deferred", False))]
    active_n = len(active)

    labels = [int(row["is_genuine"]) for row in active]
    decisions = [row.get("decision") for row in active if str(row.get("decision")) != "defer"]
    scores = [
        _safe_float(row.get("similarity_score"))
        for row in active
        if _safe_float(row.get("similarity_score")) is not None
    ]

    labels_arr = np.asarray(labels, dtype=int) if labels else np.asarray([], dtype=int)
    accept_arr = np.asarray([_is_accept(decision) for decision in decisions], dtype=bool)
    reject_arr = np.asarray([_is_reject(decision) for decision in decisions], dtype=bool)

    if active_n and len(labels_arr) == len(accept_arr):
        genuine = labels_arr == 1
        impostor = labels_arr == 0
        frr_active = float(np.sum(genuine & reject_arr) / max(1, np.sum(genuine)))
        far_active = float(np.sum(impostor & accept_arr) / max(1, np.sum(impostor)))
    else:
        frr_active = 0.0
        far_active = 0.0

    genuine_total = [row for row in rows if int(row.get("is_genuine", 0)) == 1]
    genuine_deferred = [row for row in genuine_total if _to_bool(row.get("deferred", False))]
    genuine_rejected = [
        row
        for row in genuine_total
        if not _to_bool(row.get("deferred", False)) and _is_reject(row.get("decision"))
    ]
    frr_defer_failure = (len(genuine_rejected) + len(genuine_deferred)) / max(1, len(genuine_total))

    latencies = [_safe_float(row.get("latency_ms")) for row in rows]
    latencies = [x for x in latencies if x is not None and not math.isnan(x)]
    ram_values = [_safe_float(row.get("ram_mb")) for row in rows]
    ram_values = [x for x in ram_values if x is not None and not math.isnan(x)]

    fast = sum(1 for row in active if row.get("selected_path") == "fast")
    robust = sum(1 for row in active if row.get("selected_path") == "robust")

    return {
        "n_samples": n,
        "n_active": active_n,
        "n_deferred": len(deferred),
        "FRR": frr_active,
        "FAR": far_active,
        "FRR_active": frr_active,
        "FAR_active": far_active,
        "FRR_with_defer_as_failure_for_genuine": float(frr_defer_failure),
        "EER": _eer(labels, scores) if len(labels) == len(scores) else 0.0,
        "AUC": _auc(labels, scores) if len(labels) == len(scores) else 0.0,
        "defer_rate": len(deferred) / max(1, n),
        "automation_rate": active_n / max(1, n),
        "fast_path_rate": fast / max(1, active_n),
        "robust_path_rate": robust / max(1, active_n),
        "latency_mean": float(np.mean(latencies)) if latencies else 0.0,
        "latency_std": float(np.std(latencies)) if latencies else 0.0,
        "latency_p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "ram_peak_mb": float(np.max(ram_values)) if ram_values else 0.0,
    }


def summarize_by_method(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["method_name"], str(row.get("far_budget", "")))].append(row)
    return [
        {
            "method_name": method,
            "far_budget": far_budget,
            **compute_group_metrics(group),
        }
        for (method, far_budget), group in sorted(groups.items())
    ]


def summarize_by_condition(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["method_name"],
                str(row.get("far_budget", "")),
                row.get("condition_bin", "unknown"),
            )
        ].append(row)
    return [
        {
            "method_name": method,
            "far_budget": far_budget,
            "condition_bin": condition,
            **compute_group_metrics(group),
        }
        for (method, far_budget, condition), group in sorted(groups.items())
    ]


def summarize_by_far_budget(rows: list[dict]) -> list[dict]:
    budget_rows = [row for row in rows if str(row.get("far_budget", "")).strip() not in {"", "None"}]
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in budget_rows:
        method = row["method_name"]
        far_budget = str(row.get("far_budget", ""))
        groups[(method, far_budget, "overall")].append(row)
        groups[(method, far_budget, row.get("condition_bin", "unknown"))].append(row)

    return [
        {
            "method_name": method,
            "far_budget": far_budget,
            "condition_bin": condition,
            **compute_group_metrics(group),
        }
        for (method, far_budget, condition), group in sorted(groups.items())
    ]


def latency_summary(rows: list[dict]) -> list[dict]:
    return [
        {
            "method_name": row["method_name"],
            "far_budget": row.get("far_budget", ""),
            "latency_mean": row["latency_mean"],
            "latency_std": row["latency_std"],
            "latency_p95": row["latency_p95"],
            "ram_peak_mb": row["ram_peak_mb"],
            "defer_rate": row["defer_rate"],
            "automation_rate": row["automation_rate"],
            "fast_path_rate": row["fast_path_rate"],
            "robust_path_rate": row["robust_path_rate"],
        }
        for row in summarize_by_method(rows)
    ]
