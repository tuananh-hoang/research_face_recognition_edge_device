"""Plot helpers for conditional pipeline outputs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def generate_plots(
    per_sample_rows: list[dict],
    summary_by_condition: list[dict],
    latency_summary: list[dict],
    output_dir: str | Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    dark_rows = {
        row["method_name"]: row
        for row in summary_by_condition
        if row.get("condition_bin") == "dark"
    }
    lat_rows = {row["method_name"]: row for row in latency_summary}

    methods = sorted(set(dark_rows) & set(lat_rows))
    if methods:
        plt.figure(figsize=(7, 5))
        for method in methods:
            x = lat_rows[method]["latency_mean"]
            y = dark_rows[method]["FRR"]
            plt.scatter(x, y, s=90)
            plt.annotate(method, (x, y), xytext=(5, 5), textcoords="offset points")
        plt.xlabel("Latency mean (ms)")
        plt.ylabel("FRR_dark")
        plt.title("FRR_dark vs latency")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / "frr_dark_vs_latency.png", dpi=160)
        plt.close()

        plt.figure(figsize=(7, 5))
        for method in methods:
            x = dark_rows[method]["FAR"]
            y = dark_rows[method]["FRR"]
            plt.scatter(x, y, s=90)
            plt.annotate(method, (x, y), xytext=(5, 5), textcoords="offset points")
        plt.xlabel("FAR_dark")
        plt.ylabel("FRR_dark")
        plt.title("FRR/FAR trade-off in dark condition")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / "frr_far_tradeoff.png", dpi=160)
        plt.close()

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in per_sample_rows:
        condition = row.get("condition_bin", "unknown")
        path = row.get("selected_path", "unknown")
        counts[(condition, path)] += 1

    conditions = ["bright", "medium", "dark"]
    paths = ["fast", "robust", "defer"]
    totals = {condition: sum(counts[(condition, path)] for path in paths) for condition in conditions}
    if sum(totals.values()) > 0:
        bottom = np.zeros(len(conditions))
        plt.figure(figsize=(8, 5))
        for path in paths:
            values = [
                counts[(condition, path)] / max(1, totals[condition])
                for condition in conditions
            ]
            plt.bar(conditions, values, bottom=bottom, label=path)
            bottom += np.asarray(values)
        plt.ylabel("Path usage percentage")
        plt.title("Path usage by condition")
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "path_usage_by_condition.png", dpi=160)
        plt.close()
