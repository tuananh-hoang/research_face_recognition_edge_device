"""Create the multi-dataset comparison table."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def parse_rate(value):
    if value in (None, ''):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text[:-1]) / 100.0 if text.endswith('%') else float(text)
    except ValueError:
        return None


def read_csv(path: Path):
    if not path.exists():
        return []
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def custom_dark_metrics(out_dir: Path):
    rows = read_csv(out_dir / 'formula_comparison.csv')
    fixed = next((r for r in rows if r.get('Condition') == 'dark' and r.get('Formula') == 'fixed'), None)
    bin_row = next((r for r in rows if r.get('Condition') == 'dark' and r.get('Formula') == 'bin'), None)
    return parse_rate(fixed.get('FRR')) if fixed else None, parse_rate(bin_row.get('FRR')) if bin_row else None


def xqlfw_lowq_metrics(out_dir: Path):
    rows = read_csv(out_dir / 'xqlfw_results.csv')
    fixed = next((r for r in rows if r.get('quality_bin') == 'low_quality' and r.get('formula') == 'fixed'), None)
    bin_row = next((r for r in rows if r.get('quality_bin') == 'low_quality' and r.get('formula') == 'bin'), None)
    n_pairs = parse_rate(fixed.get('n_pairs')) if fixed else None
    return n_pairs, parse_rate(fixed.get('FRR')) if fixed else None, parse_rate(bin_row.get('FRR')) if bin_row else None


def darkface_status(out_dir: Path):
    rows = read_csv(out_dir / 'iqa_validation_darkface.csv')
    row = next((r for r in rows if r.get('dataset') == 'DARK FACE (real)'), None)
    if not row:
        return 'not_run'
    status = row.get('status', '')
    if status != 'ok':
        return status or 'missing'
    dark_pct = parse_rate(row.get('dark_bin_pct'))
    det = parse_rate(row.get('detection_rate'))
    if dark_pct is None:
        return 'ok'
    return f"IQA ok, dark_bin={dark_pct:.1%}, detect={det:.1%}" if det is not None else f"IQA ok, dark_bin={dark_pct:.1%}"


def fmt_rate(value):
    return '' if value is None else f'{value:.1%}'


def main():
    out_dir = ROOT / 'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)

    custom_fixed, custom_bin = custom_dark_metrics(out_dir)
    xqlfw_n_pairs, xqlfw_fixed, xqlfw_bin = xqlfw_lowq_metrics(out_dir)

    rows = []
    rows.append({
        'Dataset': 'Custom (ours)',
        'N_persons': '14',
        'N_pairs': '2954',
        'FRR_fixed_dark': fmt_rate(custom_fixed),
        'FRR_bin_dark': fmt_rate(custom_bin),
        'Delta': fmt_rate(None if custom_fixed is None or custom_bin is None else custom_bin - custom_fixed),
        'Consistent': 'yes' if custom_fixed is not None and custom_bin is not None and custom_bin <= custom_fixed else 'unknown',
        'Note': 'primary dataset',
    })
    rows.append({
        'Dataset': 'XQLFW (low-q)',
        'N_persons': '',
        'N_pairs': '' if xqlfw_n_pairs is None else str(int(xqlfw_n_pairs)),
        'FRR_fixed_dark': fmt_rate(xqlfw_fixed),
        'FRR_bin_dark': fmt_rate(xqlfw_bin),
        'Delta': fmt_rate(None if xqlfw_fixed is None or xqlfw_bin is None else xqlfw_bin - xqlfw_fixed),
        'Consistent': 'yes' if xqlfw_fixed is not None and xqlfw_bin is not None and xqlfw_bin <= xqlfw_fixed else 'unknown',
        'Note': 'run test_xqlfw.py after downloading XQLFW',
    })
    rows.append({
        'Dataset': 'DARK FACE (IQA)',
        'N_persons': 'N/A',
        'N_pairs': 'N/A',
        'FRR_fixed_dark': 'IQA validation only',
        'FRR_bin_dark': '',
        'Delta': '',
        'Consistent': 'yes' if darkface_status(out_dir).startswith('IQA ok') else 'unknown',
        'Note': darkface_status(out_dir),
    })

    out_csv = out_dir / 'multi_dataset_summary.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Dataset', 'N_persons', 'N_pairs', 'FRR_fixed_dark', 'FRR_bin_dark', 'Delta', 'Consistent', 'Note']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_csv}")
    for row in rows:
        print(f"{row['Dataset']}: fixed={row['FRR_fixed_dark']} bin={row['FRR_bin_dark']} consistent={row['Consistent']}")


if __name__ == '__main__':
    main()
