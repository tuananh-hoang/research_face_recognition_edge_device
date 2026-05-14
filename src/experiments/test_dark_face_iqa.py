"""Validate IQA and detection rate on DARK FACE nighttime images."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.core.insightface_singleton import InsightFaceSingleton
from src.core.iqa import IQAModule


def image_paths(folder: Path):
    paths = []
    for pattern in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        paths.extend(folder.rglob(pattern))
    return sorted(paths, key=lambda p: str(p))


def read_boxes(image_path: Path):
    txt_candidates = [
        image_path.with_suffix('.txt'),
        image_path.parent / f'{image_path.stem}.txt',
    ]
    for txt in txt_candidates:
        if not txt.exists():
            continue
        boxes = []
        for line in txt.read_text(encoding='utf-8', errors='ignore').splitlines():
            vals = []
            for token in line.replace(',', ' ').split():
                try:
                    vals.append(float(token))
                except ValueError:
                    pass
            if len(vals) >= 4:
                x, y, w, h = vals[:4]
                boxes.append((int(x), int(y), int(w), int(h)))
        if boxes:
            return boxes
    return []


def crop_boxes(img, boxes):
    h, w = img.shape[:2]
    crops = []
    for x, y, bw, bh in boxes:
        x1 = max(0, min(w - 1, x))
        y1 = max(0, min(h - 1, y))
        x2 = max(x1 + 1, min(w, x + bw))
        y2 = max(y1 + 1, min(h, y + bh))
        crops.append(img[y1:y2, x1:x2])
    return crops


def summarize(rows, dataset_name):
    subset = [r for r in rows if r['dataset'] == dataset_name]
    if not subset:
        return {
            'dataset': dataset_name,
            'n_faces': 0,
            'mean_L': '',
            'mean_N': '',
            'dark_bin_pct': '',
            'detection_rate': '',
        }
    return {
        'dataset': dataset_name,
        'n_faces': len(subset),
        'mean_L': float(np.mean([r['L'] for r in subset])),
        'mean_N': float(np.mean([r['N'] for r in subset])),
        'dark_bin_pct': float(np.mean([r['bin_id'] == 'dark' for r in subset])),
        'detection_rate': float(np.mean([r['detected'] for r in subset])),
    }


def sample_custom(condition: str, limit: int):
    folder = ROOT / 'data' / condition
    return image_paths(folder)[:limit] if folder.exists() else []


def evaluate_images(paths, dataset_name, app, iqa, use_annotations=False):
    rows = []
    for image_path in paths:
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        boxes = read_boxes(image_path) if use_annotations else []
        crops = crop_boxes(img, boxes) if boxes else [img]
        for crop in crops:
            if crop.size == 0:
                continue
            L, N, bin_id, _q = iqa.compute(crop)
            faces = app.get(crop)
            rows.append({
                'dataset': dataset_name,
                'image': str(image_path),
                'L': float(L),
                'N': float(N),
                'bin_id': bin_id,
                'detected': bool(faces),
            })
    return rows


def plot_histograms(rows, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    datasets = sorted(set(r['dataset'] for r in rows))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for idx, dataset in enumerate(datasets):
        subset = [r for r in rows if r['dataset'] == dataset]
        if not subset:
            continue
        axes[0].hist([r['L'] for r in subset], bins=20, alpha=0.45, label=dataset, color=colors[idx % len(colors)])
        axes[1].hist([r['N'] for r in subset], bins=20, alpha=0.45, label=dataset, color=colors[idx % len(colors)])
    axes[0].set_title('Luminance distribution')
    axes[0].set_xlabel('L')
    axes[1].set_title('Noise distribution')
    axes[1].set_xlabel('N')
    for ax in axes:
        ax.set_ylabel('Count')
        ax.grid(alpha=0.25)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=str(ROOT / 'data' / 'dark_face'))
    parser.add_argument('--limit', type=int, default=200)
    args = parser.parse_args()

    out_dir = ROOT / 'outputs'
    fig_dir = out_dir / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    app = InsightFaceSingleton.get_instance()
    iqa = IQAModule()
    rows = []

    rows.extend(evaluate_images(sample_custom('bright', args.limit), 'Custom bright', app, iqa))
    rows.extend(evaluate_images(sample_custom('dark', args.limit), 'Custom dark (aug)', app, iqa))

    dark_face_dir = Path(args.data_dir)
    dark_face_paths = image_paths(dark_face_dir)[:args.limit] if dark_face_dir.exists() else []
    rows.extend(evaluate_images(dark_face_paths, 'DARK FACE (real)', app, iqa, use_annotations=True))

    summaries = [
        summarize(rows, 'Custom bright'),
        summarize(rows, 'Custom dark (aug)'),
        summarize(rows, 'DARK FACE (real)'),
    ]
    if not dark_face_paths:
        summaries[-1]['status'] = 'missing_dark_face_data'
    else:
        summaries[-1]['status'] = 'ok'
    summaries[0]['status'] = 'ok' if summaries[0]['n_faces'] else 'missing_custom_bright'
    summaries[1]['status'] = 'ok' if summaries[1]['n_faces'] else 'missing_custom_dark'

    out_csv = out_dir / 'iqa_validation_darkface.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['dataset', 'n_faces', 'mean_L', 'mean_N', 'dark_bin_pct', 'detection_rate', 'status']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    out_json = out_dir / 'iqa_validation_darkface.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dark_face_dir': str(dark_face_dir),
            'n_rows': len(rows),
            'summary': summaries,
        }, f, indent=2)

    if rows:
        plot_histograms(rows, fig_dir / 'Figure_07_IQA_validation.png')

    print(f"Saved: {out_csv}")
    if not dark_face_paths:
        print(f"DARK FACE images not found under {dark_face_dir}; custom IQA rows were still computed.")


if __name__ == '__main__':
    main()
