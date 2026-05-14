"""Summarize AWS edge-constrained simulation benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def profile_limits(profile_name: str, profile_config: dict) -> tuple[str, str]:
    cpu_limit = str(profile_config.get("cpu_limit") or "")
    memory_limit = str(profile_config.get("memory_limit") or "")
    if cpu_limit and memory_limit:
        return cpu_limit, memory_limit

    match = re.match(r"edge_(\d+)cpu_(\d+)(mb|gb)$", profile_name)
    if match:
        cpu_limit = cpu_limit or match.group(1)
        memory_limit = memory_limit or f"{match.group(2)}{match.group(3)}"
    return cpu_limit, memory_limit


def by_method(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        method = row.get("method_name")
        if method:
            result[method] = row
    return result


def by_method_budget(rows: list[dict]) -> dict[tuple[str, str], dict]:
    result = {}
    for row in rows:
        method = row.get("method_name")
        if method:
            result[(method, str(row.get("far_budget", "")))] = row
    return result


def dark_condition_map(rows: list[dict]) -> dict[tuple[str, str], dict]:
    result = {}
    for row in rows:
        if row.get("condition_bin") == "dark" and row.get("method_name"):
            result[(row["method_name"], str(row.get("far_budget", "")))] = row
    return result


def first_env_summary(base_dir: Path) -> dict:
    candidates = sorted(base_dir.glob("*/aws_edge_env.json"))
    candidates.append(base_dir / "aws_edge_env.json")
    for path in candidates:
        data = read_json(path)
        if data:
            return data
    return {}


def build_summaries(base_dir: Path) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    combined_rows: list[dict] = []
    latency_rows: list[dict] = []
    far_budget_rows: list[dict] = []
    calibration_rows: list[dict] = []

    for profile_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        profile = profile_dir.name
        config = read_json(profile_dir / "config_used.json")
        profile_config = read_json(profile_dir / "profile_config.json")
        cpu_limit, memory_limit = profile_limits(profile, profile_config)

        method_rows = by_method_budget(read_csv(profile_dir / "summary_by_method.csv"))
        latency_by_method = by_method_budget(read_csv(profile_dir / "latency_summary.csv"))
        dark_rows = dark_condition_map(read_csv(profile_dir / "summary_by_condition.csv"))
        n_pairs = config.get("n_test_pairs", config.get("n_pairs", ""))

        for (method_name, far_budget), method_row in method_rows.items():
            latency_row = latency_by_method.get((method_name, far_budget), {})
            dark_row = dark_rows.get((method_name, far_budget), {})
            row = {
                "profile": profile,
                "cpu_limit": cpu_limit,
                "memory_limit": memory_limit,
                "dataset": config.get("dataset", ""),
                "n_pairs": n_pairs,
                "method_name": method_name,
                "far_budget": far_budget,
                "FRR": method_row.get("FRR", ""),
                "FAR": method_row.get("FAR", ""),
                "FRR_active": method_row.get("FRR_active", method_row.get("FRR", "")),
                "FAR_active": method_row.get("FAR_active", method_row.get("FAR", "")),
                "FRR_with_defer_as_failure_for_genuine": method_row.get(
                    "FRR_with_defer_as_failure_for_genuine", ""
                ),
                "FRR_dark": dark_row.get("FRR", ""),
                "FAR_dark": dark_row.get("FAR", ""),
                "FRR_dark_active": dark_row.get("FRR_active", dark_row.get("FRR", "")),
                "FAR_dark_active": dark_row.get("FAR_active", dark_row.get("FAR", "")),
                "latency_mean": latency_row.get("latency_mean", method_row.get("latency_mean", "")),
                "latency_p95": latency_row.get("latency_p95", method_row.get("latency_p95", "")),
                "ram_peak_mb": latency_row.get("ram_peak_mb", method_row.get("ram_peak_mb", "")),
                "defer_rate": latency_row.get("defer_rate", method_row.get("defer_rate", "")),
                "automation_rate": method_row.get("automation_rate", latency_row.get("automation_rate", "")),
                "robust_path_rate": latency_row.get(
                    "robust_path_rate",
                    method_row.get("robust_path_rate", ""),
                ),
                "fast_path_rate": latency_row.get("fast_path_rate", method_row.get("fast_path_rate", "")),
            }
            combined_rows.append(row)
            latency_rows.append(
                {
                    "profile": profile,
                    "cpu_limit": cpu_limit,
                    "memory_limit": memory_limit,
                    "method_name": method_name,
                    "far_budget": far_budget,
                    "latency_mean": row["latency_mean"],
                    "latency_p95": row["latency_p95"],
                    "ram_peak_mb": row["ram_peak_mb"],
                    "defer_rate": row["defer_rate"],
                    "automation_rate": row["automation_rate"],
                    "robust_path_rate": row["robust_path_rate"],
                    "fast_path_rate": row["fast_path_rate"],
                }
            )

        for row in read_csv(profile_dir / "summary_by_far_budget.csv"):
            far_budget_rows.append(
                {
                    "profile": profile,
                    "cpu_limit": cpu_limit,
                    "memory_limit": memory_limit,
                    "dataset": config.get("dataset", ""),
                    "n_pairs": n_pairs,
                    **row,
                }
            )

        for row in read_csv(profile_dir / "calibration_thresholds.csv"):
            calibration_rows.append(
                {
                    "profile": profile,
                    "cpu_limit": cpu_limit,
                    "memory_limit": memory_limit,
                    "dataset": config.get("dataset", ""),
                    "n_calibration_pairs": config.get("n_calibration_pairs", ""),
                    **row,
                }
            )

    return combined_rows, latency_rows, far_budget_rows, calibration_rows, first_env_summary(base_dir)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No completed profile outputs found._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_report(
    base_dir: Path,
    combined_rows: list[dict],
    far_budget_rows: list[dict],
    env: dict,
) -> None:
    profiles = []
    seen = set()
    for row in combined_rows:
        key = (row["profile"], row["cpu_limit"], row["memory_limit"])
        if key not in seen:
            seen.add(key)
            profiles.append(
                {
                    "profile": row["profile"],
                    "cpu_limit": row["cpu_limit"],
                    "memory_limit": row["memory_limit"],
                }
            )

    report = [
        "# AWS Edge-Constrained Simulation Benchmark",
        "",
        "This report summarizes an AWS edge-constrained simulation. It is not a physical Raspberry Pi hardware run.",
        "AWS EC2 ARM plus Docker CPU/RAM limits provide a reproducible constrained environment for comparison.",
        "",
        "## Benchmark Profiles",
        "",
        markdown_table(profiles, ["profile", "cpu_limit", "memory_limit"]),
        "",
        "## Environment Summary",
        "",
        f"- Platform: {env.get('platform', 'unknown')}",
        f"- Machine: {env.get('machine', 'unknown')}",
        f"- Python: {env.get('python_version', 'unknown')}",
        f"- CPU count: {env.get('cpu_count', 'unknown')}",
        f"- Inside Docker: {env.get('inside_docker', 'unknown')}",
        f"- Container memory limit MB: {env.get('container_memory_limit', {}).get('mb')}",
        f"- Container CPU quota: {env.get('container_cpu_quota', {}).get('cpus')}",
        "",
        "## Risk-Constrained Results",
        "",
        "Thresholds are calibrated under FAR budgets. The goal is to reduce FRR at the same FAR constraint.",
        "Deferred samples are reported explicitly and are not hidden.",
        "",
        markdown_table(
            [row for row in far_budget_rows if row.get("condition_bin") == "dark"],
            [
                "profile",
                "method_name",
                "far_budget",
                "FAR_active",
                "FRR_active",
                "FRR_with_defer_as_failure_for_genuine",
                "defer_rate",
                "automation_rate",
                "latency_p95",
                "ram_peak_mb",
            ],
        ),
        "",
        "## Per-Method Results",
        "",
        markdown_table(
            combined_rows,
            [
                "profile",
                "method_name",
                "far_budget",
                "FRR",
                "FAR",
                "FRR_dark",
                "FAR_dark",
                "latency_mean",
                "latency_p95",
                "ram_peak_mb",
                "defer_rate",
            ],
        ),
        "",
        "## Interpretation Template",
        "",
        "- Primary claim: compare FRR_dark at the same allowed FAR budget alpha.",
        "- Check actual FAR_active before interpreting FRR_active.",
        "- Report defer_rate and automation_rate together; deferring difficult samples is not the same as correctly recognizing them.",
        "- Compare latency_p95 and ram_peak_mb across AWS edge-constrained profiles.",
        "- Report all results as AWS edge-constrained simulation results, not as physical edge-board measurements.",
        "",
        "## Warning",
        "",
        "These numbers are produced under Docker CPU/RAM limits on AWS ARM hardware. Physical Raspberry Pi or Jetson validation remains separate future work.",
        "",
    ]
    (base_dir / "edge_benchmark_report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(ROOT / "outputs" / "aws_edge_benchmark"))
    args = parser.parse_args()

    base_dir = Path(args.input_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    combined_rows, latency_rows, far_budget_rows, calibration_rows, env = build_summaries(base_dir)
    combined_fields = [
        "profile",
        "cpu_limit",
        "memory_limit",
        "dataset",
        "n_pairs",
        "method_name",
        "far_budget",
        "FRR",
        "FAR",
        "FRR_active",
        "FAR_active",
        "FRR_with_defer_as_failure_for_genuine",
        "FRR_dark",
        "FAR_dark",
        "FRR_dark_active",
        "FAR_dark_active",
        "latency_mean",
        "latency_p95",
        "ram_peak_mb",
        "defer_rate",
        "automation_rate",
        "robust_path_rate",
        "fast_path_rate",
    ]
    latency_fields = [
        "profile",
        "cpu_limit",
        "memory_limit",
        "method_name",
        "far_budget",
        "latency_mean",
        "latency_p95",
        "ram_peak_mb",
        "defer_rate",
        "automation_rate",
        "robust_path_rate",
        "fast_path_rate",
    ]
    far_budget_fields = [
        "profile",
        "cpu_limit",
        "memory_limit",
        "dataset",
        "n_pairs",
        "method_name",
        "far_budget",
        "condition_bin",
        "FRR_active",
        "FAR_active",
        "FRR_with_defer_as_failure_for_genuine",
        "defer_rate",
        "automation_rate",
        "latency_mean",
        "latency_p95",
        "ram_peak_mb",
    ]
    calibration_fields = [
        "profile",
        "cpu_limit",
        "memory_limit",
        "dataset",
        "n_calibration_pairs",
        "method_name",
        "far_budget",
        "condition_bin",
        "threshold_accept",
        "threshold_reject",
        "n_calibration_genuine",
        "n_calibration_impostor",
        "calibration_FAR",
        "calibration_FRR",
        "used_global_fallback",
    ]

    write_csv(base_dir / "combined_edge_summary.csv", combined_rows, combined_fields)
    write_csv(base_dir / "combined_latency_summary.csv", latency_rows, latency_fields)
    write_csv(base_dir / "combined_far_budget_summary.csv", far_budget_rows, far_budget_fields)
    write_csv(base_dir / "combined_calibration_thresholds.csv", calibration_rows, calibration_fields)
    write_report(base_dir, combined_rows, far_budget_rows, env)

    print(f"Profiles summarized: {len(set(row['profile'] for row in combined_rows))}")
    print(f"Rows summarized: {len(combined_rows)}")
    print(f"Saved: {base_dir / 'combined_edge_summary.csv'}")
    print(f"Saved: {base_dir / 'combined_latency_summary.csv'}")
    print(f"Saved: {base_dir / 'combined_far_budget_summary.csv'}")
    print(f"Saved: {base_dir / 'combined_calibration_thresholds.csv'}")
    print(f"Saved: {base_dir / 'edge_benchmark_report.md'}")


if __name__ == "__main__":
    main()
