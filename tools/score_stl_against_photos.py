#!/usr/bin/env python3
"""Compare candidate STL silhouettes with four supplied hero screenshots."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import trimesh

CANVAS = 256
INNER = 224


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--photos', nargs='+', required=True, type=Path)
    parser.add_argument('--candidates', required=True, type=Path)
    parser.add_argument('--clip-report', type=Path)
    parser.add_argument('--hero', required=True)
    parser.add_argument('--output-json', required=True, type=Path)
    parser.add_argument('--output-csv', required=True, type=Path)
    parser.add_argument('--contact-sheet', required=True, type=Path)
    parser.add_argument('--angle-step', type=int, default=10)
    return parser.parse_args()


def fallback_foreground(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
    h, w, _ = rgb.shape
    border = np.concatenate((
        rgb[: max(2, h // 25)].reshape(-1, 3),
        rgb[-max(2, h // 25):].reshape(-1, 3),
        rgb[:, : max(2, w // 25)].reshape(-1, 3),
        rgb[:, -max(2, w // 25):].reshape(-1, 3),
    ))
    background = np.median(border, axis=0)
    distance = np.linalg.norm(rgb - background, axis=2)
    threshold = max(24.0, float(np.percentile(distance, 70)) * 0.55)
    return distance > threshold


def segment_photo(path: Path) -> np.ndarray:
    image = Image.open(path).convert('RGB')
    mask = None
    try:
        from rembg import remove
        rgba = remove(image)
        alpha = np.asarray(rgba.getchannel('A'))
        mask = alpha > 24
    except Exception as exc:
        print(f'WARNING: rembg failed for {path}: {exc}')
    if mask is None or mask.sum() < 100:
        mask = fallback_foreground(image)
    return clean_reference_mask(mask)


def clean_reference_mask(mask: np.ndarray) -> np.ndarray:
    mask = ndimage.binary_closing(mask, iterations=2)
    labels, count = ndimage.label(mask)
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        largest = max(1, int(sizes.max()))
        h, w = mask.shape
        yy, xx = np.indices(mask.shape)
        central = np.zeros_like(mask, dtype=bool)
        for label_index in range(1, count + 1):
            size = int(sizes[label_index])
            if size < max(40, largest * 0.002):
                continue
            component = labels == label_index
            cy = float(yy[component].mean())
            cx = float(xx[component].mean())
            # Keep large components and accessories near the central hero area.
            if size >= largest * 0.02 or (0.08 * w <= cx <= 0.92 * w and 0.05 * h <= cy <= 0.95 * h):
                central |= component
        mask = central if central.any() else labels == int(np.argmax(sizes))
    mask = ndimage.binary_dilation(mask, iterations=1)
    mask = ndimage.binary_fill_holes(mask)
    return normalize_mask(mask)


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if not len(coords):
        return np.zeros((CANVAS, CANVAS), dtype=bool)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = mask[y0:y1, x0:x1]
    h, w = cropped.shape
    scale = min(INNER / max(1, w), INNER / max(1, h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = Image.fromarray((cropped * 255).astype(np.uint8)).resize((nw, nh), Image.Resampling.NEAREST)
    result = np.zeros((CANVAS, CANVAS), dtype=bool)
    x = (CANVAS - nw) // 2
    y = (CANVAS - nh) // 2
    result[y:y + nh, x:x + nw] = np.asarray(resized) > 127
    return result


def project_points(points: np.ndarray, angle_degrees: float) -> np.ndarray:
    radians = math.radians(angle_degrees)
    c, s = math.cos(radians), math.sin(radians)
    horizontal = points[:, 0] * c - points[:, 1] * s
    vertical = points[:, 2]
    projected = np.column_stack((horizontal, vertical))
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    extent = np.maximum(maxs - mins, 1e-8)
    scale = min((INNER - 1) / extent[0], (INNER - 1) / extent[1])
    pixels = (projected - mins) * scale
    pixels[:, 0] += (CANVAS - extent[0] * scale) / 2.0
    pixels[:, 1] += (CANVAS - extent[1] * scale) / 2.0
    x = np.clip(np.rint(pixels[:, 0]).astype(int), 0, CANVAS - 1)
    # World Z increases up; image row increases down.
    y = np.clip(CANVAS - 1 - np.rint(pixels[:, 1]).astype(int), 0, CANVAS - 1)
    mask = np.zeros((CANVAS, CANVAS), dtype=bool)
    mask[y, x] = True
    mask = ndimage.binary_dilation(mask, iterations=2)
    mask = ndimage.binary_closing(mask, iterations=3)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=1)
    return normalize_mask(mask)


def shifted(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    result = np.zeros_like(mask)
    y_src0 = max(0, -dy)
    y_src1 = min(mask.shape[0], mask.shape[0] - dy)
    x_src0 = max(0, -dx)
    x_src1 = min(mask.shape[1], mask.shape[1] - dx)
    y_dst0 = max(0, dy)
    y_dst1 = y_dst0 + (y_src1 - y_src0)
    x_dst0 = max(0, dx)
    x_dst1 = x_dst0 + (x_src1 - x_src0)
    if y_src1 > y_src0 and x_src1 > x_src0:
        result[y_dst0:y_dst1, x_dst0:x_dst1] = mask[y_src0:y_src1, x_src0:x_src1]
    return result


def shape_score(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref_area = max(1, int(reference.sum()))
    best = 0.0
    ref_edge = reference ^ ndimage.binary_erosion(reference)
    ref_distance = ndimage.distance_transform_edt(~ref_edge)
    ref_coords = np.argwhere(reference)
    ref_h = max(1, int(ref_coords[:, 0].ptp() + 1)) if len(ref_coords) else 1
    ref_w = max(1, int(ref_coords[:, 1].ptp() + 1)) if len(ref_coords) else 1
    ref_aspect = ref_w / ref_h

    for dy in range(-6, 7, 2):
        for dx in range(-6, 7, 2):
            moved = shifted(candidate, dy, dx)
            intersection = int(np.logical_and(reference, moved).sum())
            union = int(np.logical_or(reference, moved).sum())
            iou = intersection / max(1, union)
            coverage = intersection / ref_area
            moved_edge = moved ^ ndimage.binary_erosion(moved)
            if moved_edge.any():
                chamfer = float(ref_distance[moved_edge].mean())
            else:
                chamfer = 50.0
            moved_coords = np.argwhere(moved)
            if len(moved_coords):
                h = max(1, int(moved_coords[:, 0].ptp() + 1))
                w = max(1, int(moved_coords[:, 1].ptp() + 1))
                aspect = w / h
                aspect_score = math.exp(-abs(math.log(max(1e-6, aspect / ref_aspect))))
            else:
                aspect_score = 0.0
            chamfer_score = math.exp(-chamfer / 7.0)
            value = 0.48 * iou + 0.24 * coverage + 0.18 * chamfer_score + 0.10 * aspect_score
            best = max(best, value)
    return float(best)


def sample_mesh(path: Path) -> np.ndarray:
    mesh = trimesh.load_mesh(path, force='mesh', process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f'{path} did not load as Trimesh')
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    count = min(500_000, max(180_000, len(mesh.faces) * 2))
    points, _ = trimesh.sample.sample_surface(mesh, count)
    # Suppress tiny far-away helper remnants by clipping to robust bounds.
    lower = np.percentile(points, 0.05, axis=0)
    upper = np.percentile(points, 99.95, axis=0)
    keep = np.all((points >= lower) & (points <= upper), axis=1)
    return points[keep]


def clip_scores(report_path: Path | None, hero: str) -> dict[str, float]:
    if report_path is None or not report_path.exists():
        return {}
    report = json.loads(report_path.read_text(encoding='utf-8'))
    rows = report.get('heroes', {}).get(hero, {}).get('ranked_candidates', [])
    return {row['skin_key']: float(row['combined_score']) for row in rows}


def main() -> int:
    args = parse_args()
    references = [segment_photo(path) for path in args.photos]
    angle_step = args.angle_step
    angles = list(range(0, 360, angle_step))
    quarter = int(round(90 / angle_step))
    clip = clip_scores(args.clip_report, args.hero)

    candidates = []
    render_cache: dict[str, list[np.ndarray]] = {}
    for path in sorted(args.candidates.glob('*.stl')):
        key = path.stem
        points = sample_mesh(path)
        renders = [project_points(points, angle) for angle in angles]
        render_cache[key] = renders
        best_sequence = None
        best_score = -1.0
        for base in range(len(angles)):
            for direction in (1, -1):
                indices = [
                    (base + direction * quarter * index) % len(angles)
                    for index in range(len(references))
                ]
                scores = [shape_score(ref, renders[idx]) for ref, idx in zip(references, indices)]
                mean_score = float(np.mean(scores))
                if mean_score > best_score:
                    best_score = mean_score
                    best_sequence = {
                        'base_angle': angles[base],
                        'direction': direction,
                        'angles': [angles[index] for index in indices],
                        'per_view_scores': scores,
                    }
        candidates.append({
            'skin_key': key,
            'stl_path': str(path),
            'silhouette_score': best_score,
            'best_sequence': best_sequence,
            'clip_score': clip.get(key),
        })

    silhouette_values = np.array([row['silhouette_score'] for row in candidates], dtype=float)
    if len(silhouette_values) and silhouette_values.ptp() > 1e-9:
        silhouette_normalized = (silhouette_values - silhouette_values.min()) / silhouette_values.ptp()
    else:
        silhouette_normalized = np.ones_like(silhouette_values)
    clip_values = np.array([
        row['clip_score'] if row['clip_score'] is not None else 0.0
        for row in candidates
    ], dtype=float)
    if len(clip_values) and clip_values.ptp() > 1e-9:
        clip_normalized = (clip_values - clip_values.min()) / clip_values.ptp()
    else:
        clip_normalized = np.full_like(clip_values, 0.5)

    for row, silhouette_value, clip_value in zip(candidates, silhouette_normalized, clip_normalized):
        row['silhouette_normalized'] = float(silhouette_value)
        row['clip_normalized'] = float(clip_value)
        row['combined_score'] = float(0.68 * silhouette_value + 0.32 * clip_value)
    candidates.sort(key=lambda row: row['combined_score'], reverse=True)

    result = {
        'hero': args.hero,
        'photos': [str(path) for path in args.photos],
        'angle_step': angle_step,
        'selected_skin_key': candidates[0]['skin_key'],
        'selection_method': '68% multi-angle silhouette + 32% CLIP appearance',
        'ranked_candidates': candidates,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding='utf-8')
    with args.output_csv.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'skin_key', 'combined_score', 'silhouette_score', 'silhouette_normalized',
            'clip_score', 'clip_normalized', 'stl_path', 'best_sequence',
        ])
        writer.writeheader()
        for row in candidates:
            serialised = dict(row)
            serialised['best_sequence'] = json.dumps(serialised['best_sequence'])
            writer.writerow(serialised)

    # Compact visual report: reference masks, then each candidate's matched views.
    cell = 270
    rows = 1 + len(candidates)
    sheet = Image.new('RGB', (cell * 4, cell * rows), 'white')
    font = ImageFont.load_default()
    for col, mask in enumerate(references):
        image = Image.fromarray((mask * 255).astype(np.uint8)).convert('RGB')
        sheet.paste(image, (col * cell + 7, 7))
        ImageDraw.Draw(sheet).text((col * cell + 10, 245), f'Reference {col + 1}', fill='red', font=font)
    for row_index, candidate in enumerate(candidates, start=1):
        sequence = candidate['best_sequence']['angles']
        key = candidate['skin_key']
        renders = render_cache[key]
        for col, angle in enumerate(sequence):
            index = angles.index(angle)
            image = Image.fromarray((renders[index] * 255).astype(np.uint8)).convert('RGB')
            sheet.paste(image, (col * cell + 7, row_index * cell + 7))
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (10, row_index * cell + 245),
            f"{key}: combined {candidate['combined_score']:.3f}, silhouette {candidate['silhouette_score']:.3f}",
            fill='blue',
            font=font,
        )
    args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.contact_sheet, quality=92)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
