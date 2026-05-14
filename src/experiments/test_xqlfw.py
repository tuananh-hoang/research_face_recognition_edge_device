"""Evaluate threshold formulas on XQLFW cross-quality pairs.

Expected layout:
  data/xqlfw/pairs.csv
or the official protocol:
  data/xqlfw/xqlfw_pairs.txt
  data/xqlfw/<aligned images>

The CSV may use either:
  img1_path,img2_path,label
or common aliases:
  path1,path2,same / image1,image2,issame
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.core.embedder import RealEmbedder
from src.core.iqa import IQAModule
from src.threshold import (
    formula_bin_specific,
    formula_fixed,
    formula_interaction,
    formula_linear,
)


FORMULAS = {
    'fixed': formula_fixed,
    'bin': formula_bin_specific,
    'linear': formula_linear,
    'interaction': formula_interaction,
}


def normalize_label(value) -> int:
    text = str(value).strip().lower()
    return 1 if text in {'1', 'true', 'same', 'yes', 'y'} else 0


def pick_column(row, candidates):
    for key in candidates:
        if key in row and row[key]:
            return row[key]
    raise KeyError(f"Missing any of columns: {candidates}")


def resolve_image(root: Path, raw_path: str, image_index=None) -> Path:
    path = Path(raw_path)
    candidate = path if path.is_absolute() else root / path
    if candidate.exists() or image_index is None:
        return candidate
    return image_index.get(path.name, candidate)


def lfw_image_name(person: str, idx: int) -> str:
    return f'{person}/{person}_{idx:04d}.jpg'


def quality_bin(L: float, N: float) -> str:
    if L > 0.6 and N < 0.1:
        return 'high_quality'
    if L < 0.3 or N > 0.3:
        return 'low_quality'
    return 'medium_quality'


def compute_eer(labels, scores) -> float:
    if len(set(labels)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def evaluate(rows, formula_func):
    if not rows:
        return {'FRR': 0.0, 'FAR': 0.0, 'EER': 0.0, 'n_pairs': 0}

    labels, scores, decisions = [], [], []
    for row in rows:
        tau = formula_func(row['bin_id'], row['L'], row['N'], row['q'])
        labels.append(row['label'])
        scores.append(row['sim'])
        decisions.append(1 if row['sim'] >= tau else 0)

    labels_arr = np.asarray(labels)
    decisions_arr = np.asarray(decisions)
    frr = float(np.sum((labels_arr == 1) & (decisions_arr == 0)) / max(1, np.sum(labels_arr == 1)))
    far = float(np.sum((labels_arr == 0) & (decisions_arr == 1)) / max(1, np.sum(labels_arr == 0)))
    return {'FRR': frr, 'FAR': far, 'EER': compute_eer(labels, scores), 'n_pairs': len(rows)}


def load_pairs_csv(dataset_dir: Path, limit: int | None):
    csv_path = dataset_dir / 'pairs.csv'
    if not csv_path.exists():
        matches = list(dataset_dir.rglob('pairs.csv'))
        csv_path = matches[0] if matches else csv_path
    if not csv_path.exists():
        return None, []

    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if limit:
        rows = rows[:limit]
    return csv_path, rows


def load_official_pairs_txt(dataset_dir: Path, limit: int | None):
    txt_path = dataset_dir / 'xqlfw_pairs.txt'
    if not txt_path.exists():
        matches = list(dataset_dir.rglob('xqlfw_pairs.txt'))
        txt_path = matches[0] if matches else txt_path
    if not txt_path.exists():
        return None, []

    lines = [
        line.strip().split()
        for line in txt_path.read_text(encoding='utf-8', errors='ignore').splitlines()
        if line.strip()
    ]
    rows = []
    for parts in lines[1:]:
        if len(parts) == 3:
            person, idx1, idx2 = parts
            rows.append({
                'img1_path': lfw_image_name(person, int(idx1)),
                'img2_path': lfw_image_name(person, int(idx2)),
                'label': '1',
            })
        elif len(parts) == 4:
            person1, idx1, person2, idx2 = parts
            rows.append({
                'img1_path': lfw_image_name(person1, int(idx1)),
                'img2_path': lfw_image_name(person2, int(idx2)),
                'label': '0',
            })
    if limit:
        rows = rows[:limit]
    return txt_path, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=str(ROOT / 'data' / 'xqlfw'))
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    dataset_dir = Path(args.data_dir)
    out_dir = ROOT / 'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'xqlfw_results.csv'
    out_json = out_dir / 'xqlfw_results.json'

    pairs_csv, raw_pairs = load_pairs_csv(dataset_dir, args.limit)
    if not raw_pairs:
        pairs_csv, raw_pairs = load_official_pairs_txt(dataset_dir, args.limit)
    if not raw_pairs:
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['quality_bin', 'formula', 'n_pairs', 'FRR', 'FAR', 'EER', 'status'])
            writer.writeheader()
            writer.writerow({'quality_bin': 'all', 'formula': 'all', 'n_pairs': 0, 'FRR': '', 'FAR': '', 'EER': '', 'status': 'missing_xqlfw_pairs'})
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump({'status': 'missing_xqlfw_pairs', 'data_dir': str(dataset_dir)}, f, indent=2)
        print(f"XQLFW pairs.csv not found under {dataset_dir}. Wrote placeholder output.")
        return

    embedder = RealEmbedder()
    iqa = IQAModule()
    image_index = {
        p.name: p
        for p in dataset_dir.rglob('*')
        if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    }
    evaluated = []

    for row in raw_pairs:
        try:
            img1_path = resolve_image(dataset_dir, pick_column(row, ['img1_path', 'path1', 'image1', 'img1']), image_index)
            img2_path = resolve_image(dataset_dir, pick_column(row, ['img2_path', 'path2', 'image2', 'img2']), image_index)
            label = normalize_label(pick_column(row, ['label', 'same', 'issame', 'target']))
        except KeyError:
            continue

        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        if img1 is None or img2 is None:
            continue

        sim, q_emb, _ = embedder.similarity(img1, img2)
        if sim is None:
            continue

        L, N, bin_id, q_iqa = iqa.compute(img2)
        evaluated.append({
            'sim': float(sim),
            'label': label,
            'L': float(L),
            'N': float(N),
            'q': float(q_emb if q_emb is not None else q_iqa),
            'bin_id': bin_id,
            'quality_bin': quality_bin(L, N),
        })

    results = []
    for qbin in ['high_quality', 'medium_quality', 'low_quality']:
        subset = [row for row in evaluated if row['quality_bin'] == qbin]
        for formula_name, formula_func in FORMULAS.items():
            metrics = evaluate(subset, formula_func)
            results.append({
                'quality_bin': qbin,
                'formula': formula_name,
                'n_pairs': metrics['n_pairs'],
                'FRR': metrics['FRR'],
                'FAR': metrics['FAR'],
                'EER': metrics['EER'],
                'status': 'ok',
            })

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['quality_bin', 'formula', 'n_pairs', 'FRR', 'FAR', 'EER', 'status'])
        writer.writeheader()
        writer.writerows(results)

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({
            'status': 'ok',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pairs_csv': str(pairs_csv),
            'n_pairs_loaded': len(raw_pairs),
            'n_pairs_evaluated': len(evaluated),
            'results': results,
        }, f, indent=2)

    low_fixed = next((r for r in results if r['quality_bin'] == 'low_quality' and r['formula'] == 'fixed'), None)
    low_bin = next((r for r in results if r['quality_bin'] == 'low_quality' and r['formula'] == 'bin'), None)
    if low_fixed and low_bin and low_fixed['n_pairs']:
        print(f"XQLFW low-quality: bin FRR delta vs fixed = {low_bin['FRR'] - low_fixed['FRR']:+.1%}")
    print(f"Saved: {out_csv}")


if __name__ == '__main__':
    main()
