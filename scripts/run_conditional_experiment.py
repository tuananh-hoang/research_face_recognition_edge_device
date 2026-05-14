"""Run the conditional pipeline experiment from docs/plan3.md."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.conditional.evaluator import ConditionalEvaluator, MethodConfig, PairRecord
from src.conditional.metrics import (
    latency_summary,
    summarize_by_condition,
    summarize_by_method,
)
from src.conditional.plotting import generate_plots
from src.conditional.policies import AlwaysFastPolicy, AlwaysRobustPolicy, ConditionalPolicy
from src.conditional.thresholds import (
    BinSpecificThreshold,
    FixedThreshold,
    PathSpecificBinThreshold,
)


def parse_person_id(path: Path) -> str:
    match = re.match(r"(\d+)", path.stem)
    return match.group(1) if match else path.stem.split("_")[0]


def image_paths(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(folder.glob(pattern))
    return sorted(paths, key=lambda p: p.name)


def load_synthetic_pairs(seed: int = 42, n_per_condition_label: int = 500) -> list[PairRecord]:
    rng = np.random.default_rng(seed)
    records: list[PairRecord] = []

    conditions = [
        ("bright", 0.62, 0.08, 0.18, 0.10, 0.60, 0.90, 0.02, 0.10),
        ("medium", 0.52, 0.10, 0.22, 0.11, 0.30, 0.60, 0.08, 0.20),
        ("dark", 0.41, 0.12, 0.28, 0.12, 0.05, 0.30, 0.15, 0.50),
    ]

    for condition, sm, ss, dm, ds, l_low, l_high, n_low, n_high in conditions:
        for label in (1, 0):
            for idx in range(n_per_condition_label):
                sim = rng.normal(sm, ss) if label == 1 else rng.normal(dm, ds)
                sim = float(np.clip(sim, -1.0, 1.0))
                L = float(rng.uniform(l_low, l_high))
                N = float(rng.uniform(n_low, n_high))
                records.append(
                    PairRecord(
                        pair_id=f"syn_{condition}_{label}_{idx:04d}",
                        label=label,
                        person_id=f"synthetic_{label}",
                        sim=sim,
                        L=L,
                        N=N,
                        q=float(max(0.0, min(1.0, 1.0 - N))),
                        bin_id=condition,
                    )
                )
    return records


def load_custom_pairs(
    data_dir: Path,
    seed: int = 42,
    max_pairs: int | None = None,
    positives_per_person: int = 3,
    negatives_per_positive: int = 1,
) -> list[PairRecord]:
    rng = np.random.default_rng(seed)
    records: list[PairRecord] = []

    for condition in ("bright", "medium", "dark"):
        condition_dir = data_dir / condition
        if not condition_dir.exists():
            continue

        by_person: dict[str, list[Path]] = {}
        for path in image_paths(condition_dir):
            by_person.setdefault(parse_person_id(path), []).append(path)

        persons = sorted(by_person.keys())
        gallery = {pid: paths[0] for pid, paths in by_person.items() if paths}
        pair_idx = 0

        for pid in persons:
            paths = by_person.get(pid, [])
            if len(paths) < 2 or pid not in gallery:
                continue

            probes = paths[1 : 1 + positives_per_person]
            for probe_path in probes:
                records.append(
                    PairRecord(
                        pair_id=f"{condition}_same_{pair_idx:05d}",
                        label=1,
                        person_id=pid,
                        image1_path=str(gallery[pid]),
                        image2_path=str(probe_path),
                        bin_id=condition,
                    )
                )
                pair_idx += 1

                impostor_candidates = [other for other in persons if other != pid and other in gallery]
                if not impostor_candidates:
                    continue
                chosen = rng.choice(
                    impostor_candidates,
                    size=min(negatives_per_positive, len(impostor_candidates)),
                    replace=False,
                )
                for other_pid in chosen:
                    records.append(
                        PairRecord(
                            pair_id=f"{condition}_diff_{pair_idx:05d}",
                            label=0,
                            person_id=f"{other_pid}->{pid}",
                            image1_path=str(gallery[other_pid]),
                            image2_path=str(probe_path),
                            bin_id=condition,
                        )
                    )
                    pair_idx += 1

    if max_pairs is not None and max_pairs > 0 and len(records) > max_pairs:
        idx = rng.permutation(len(records))[:max_pairs]
        records = [records[i] for i in sorted(idx)]

    return records


def build_methods(args) -> dict[str, MethodConfig]:
    conditional_policy = ConditionalPolicy(
        l_defer=args.l_defer,
        l_robust=args.l_robust,
        n_robust=args.n_robust,
        q_defer=args.q_defer,
    )
    return {
        "M0": MethodConfig("M0_always_fast_fixed", AlwaysFastPolicy(), FixedThreshold()),
        "M1": MethodConfig("M1_always_fast_bin", AlwaysFastPolicy(), BinSpecificThreshold()),
        "M2": MethodConfig("M2_always_robust_bin", AlwaysRobustPolicy(), BinSpecificThreshold()),
        "M3": MethodConfig("M3_conditional_fixed", conditional_policy, FixedThreshold()),
        "M4": MethodConfig("M4_conditional_path_bin", conditional_policy, PathSpecificBinThreshold()),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary_rows: list[dict]) -> None:
    print("\nConditional pipeline summary")
    print("-" * 113)
    print(
        f"{'method':<30} | {'FRR':>7} | {'FAR':>7} | {'lat_mean':>9} | "
        f"{'lat_p95':>8} | {'RAM_MB':>8} | {'defer':>7} | {'fast':>7} | {'robust':>7}"
    )
    print("-" * 113)
    for row in summary_rows:
        print(
            f"{row['method_name']:<30} | "
            f"{row['FRR']:>6.1%} | {row['FAR']:>6.1%} | "
            f"{row['latency_mean']:>8.2f} | {row['latency_p95']:>7.2f} | "
            f"{row.get('ram_peak_mb', 0.0):>7.1f} | "
            f"{row['defer_rate']:>6.1%} | {row['fast_path_rate']:>6.1%} | "
            f"{row['robust_path_rate']:>6.1%}"
        )
    print("-" * 113)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synthetic", "custom"], default="synthetic")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "conditional_pipeline"))
    parser.add_argument("--methods", default="M0,M1,M2,M3,M4")
    parser.add_argument(
        "--face-model",
        default=None,
        help="InsightFace model pack or alias. 'mobilefacenet' maps to buffalo_sc/MBF@WebFace600K.",
    )
    parser.add_argument(
        "--face-det-size",
        default=None,
        help="Detector size, e.g. 320,320 or 256x256.",
    )
    parser.add_argument("--robust-enhancement", choices=["gamma", "clahe", "gamma+clahe", "none"], default="clahe")
    parser.add_argument("--l-defer", type=float, default=0.12)
    parser.add_argument("--l-robust", type=float, default=0.30)
    parser.add_argument("--n-robust", type=float, default=None)
    parser.add_argument("--q-defer", type=float, default=None)
    parser.add_argument("--max-pairs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-mock-embedder",
        action="store_true",
        help="Allow custom dataset runs to use RealEmbedder's mock fallback when InsightFace is missing.",
    )
    parser.add_argument(
        "--synthetic-robust-delta",
        type=float,
        default=0.0,
        help="Optional score shift for synthetic-only robust-path smoke tests.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "custom":
        records = load_custom_pairs(Path(args.data_dir), seed=args.seed, max_pairs=args.max_pairs)
        if not records:
            print("No custom image pairs found; falling back to synthetic pairs.")
            records = load_synthetic_pairs(seed=args.seed)
            args.dataset = "synthetic"
    else:
        records = load_synthetic_pairs(seed=args.seed)
        if args.max_pairs and len(records) > args.max_pairs:
            rng = np.random.default_rng(args.seed)
            idx = rng.permutation(len(records))[: args.max_pairs]
            records = [records[i] for i in sorted(idx)]

    requested = [item.strip() for item in args.methods.split(",") if item.strip()]
    method_map = build_methods(args)
    methods = [method_map[key] for key in requested if key in method_map]
    if not methods:
        raise SystemExit(f"No valid methods selected from: {sorted(method_map)}")

    embedder = None
    if args.dataset == "custom":
        from src.core.embedder import RealEmbedder
        from src.core.model_config import parse_det_size

        embedder = RealEmbedder(model_name=args.face_model, det_size=parse_det_size(args.face_det_size))
        if not getattr(embedder, "available", False) and not args.allow_mock_embedder:
            raise SystemExit(
                "InsightFace is required for custom dataset benchmarks. "
                "Install insightface/onnxruntime, or pass --allow-mock-embedder for smoke tests only."
            )

    evaluator = ConditionalEvaluator(
        embedder=embedder,
        robust_enhancement=args.robust_enhancement,
        synthetic_robust_delta=args.synthetic_robust_delta,
    )

    print("=" * 80)
    print("Running conditional pipeline experiment")
    print(f"dataset={args.dataset}  pairs={len(records)}  methods={','.join(requested)}")
    print("=" * 80)

    rows = evaluator.evaluate(records, methods)
    summary_methods = summarize_by_method(rows)
    summary_conditions = summarize_by_condition(rows)
    latency_rows = latency_summary(rows)

    per_sample_fields = [
        "image_id",
        "person_id",
        "condition_bin",
        "brightness_L",
        "noise_N",
        "det_score_q",
        "method_name",
        "selected_path",
        "enhancement_type",
        "similarity_score",
        "threshold",
        "decision",
        "is_genuine",
        "latency_ms",
        "ram_mb",
        "deferred",
        "defer_reason",
    ]

    write_csv(output_dir / "per_sample_log.csv", rows, per_sample_fields)
    write_csv(output_dir / "summary_by_method.csv", summary_methods)
    write_csv(output_dir / "summary_by_condition.csv", summary_conditions)
    write_csv(output_dir / "latency_summary.csv", latency_rows)

    config = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "data_dir": str(Path(args.data_dir).resolve()),
        "n_pairs": len(records),
        "methods": requested,
        "robust_enhancement": args.robust_enhancement,
        "l_defer": args.l_defer,
        "l_robust": args.l_robust,
        "n_robust": args.n_robust,
        "q_defer": args.q_defer,
        "synthetic_robust_delta": args.synthetic_robust_delta,
        "allow_mock_embedder": bool(args.allow_mock_embedder),
        "requested_face_model": args.face_model or "mobilefacenet",
        "resolved_face_model": getattr(embedder, "model_name", "synthetic_scores"),
        "face_model_description": getattr(embedder, "model_description", "synthetic scores; no face model loaded"),
        "face_det_size": list(getattr(embedder, "det_size", (320, 320))),
        "embedding_backend": (
            "synthetic_scores"
            if args.dataset == "synthetic"
            else ("insightface" if getattr(embedder, "available", False) else "mock")
        ),
    }
    with open(output_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    generate_plots(rows, summary_conditions, latency_rows, output_dir)
    print_summary(summary_methods)
    print(f"\nSaved conditional pipeline outputs to: {output_dir}")


if __name__ == "__main__":
    main()
