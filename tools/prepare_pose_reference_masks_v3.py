#!/usr/bin/env python3
"""Prepare normalized four-view foreground masks for HOTS pose matching."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

CANVAS = 128
INNER = 112


def border_pixels(rgb: np.ndarray) -> np.ndarray:
    h, w, _ = rgb.shape
    thickness = max(3, min(h, w) // 30)
    return np.concatenate(
        (
            rgb[:thickness].reshape(-1, 3),
            rgb[-thickness:].reshape(-1, 3),
            rgb[:, :thickness].reshape(-1, 3),
            rgb[:, -thickness:].reshape(-1, 3),
        ),
        axis=0,
    )


def initial_foreground(image: Image.Image) -> tuple[np.ndarray, dict]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    border = border_pixels(rgb)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(rgb - background[None, None, :], axis=2)

    border_distance = np.linalg.norm(border - background[None, :], axis=1)
    border_noise = float(np.percentile(border_distance, 97.0))
    broad_threshold = float(np.percentile(distance, 66.0)) * 0.52
    threshold = max(18.0, border_noise * 1.55, broad_threshold)
    mask = distance > threshold

    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = maximum - minimum
    luminance = rgb.mean(axis=2)
    border_luminance = float(np.median(border.mean(axis=1)))
    mask |= (saturation > 46.0) & (distance > threshold * 0.63)
    if border_luminance > 65.0:
        mask |= (luminance < border_luminance * 0.48) & (distance > threshold * 0.42)

    return mask, {
        "background_rgb": [round(float(value), 3) for value in background],
        "border_noise": border_noise,
        "threshold": threshold,
    }


def clean_mask(mask: np.ndarray) -> np.ndarray:
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_opening(mask, iterations=1)
    labels, count = ndimage.label(mask)
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        largest = max(1, int(sizes.max()))
        h, w = mask.shape
        yy, xx = np.indices(mask.shape)
        kept = np.zeros_like(mask, dtype=bool)
        for label_index in range(1, count + 1):
            size = int(sizes[label_index])
            if size < max(30, int(largest * 0.0015)):
                continue
            component = labels == label_index
            cy = float(yy[component].mean())
            cx = float(xx[component].mean())
            central = 0.05 * w <= cx <= 0.95 * w and 0.03 * h <= cy <= 0.97 * h
            sufficiently_large = size >= largest * 0.012
            accessory_near_hero = central and size >= largest * 0.0025
            if sufficiently_large or accessory_near_hero:
                kept |= component
        if kept.any():
            mask = kept
        else:
            mask = labels == int(np.argmax(sizes))

    mask = ndimage.binary_dilation(mask, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    return mask


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    result = np.zeros((CANVAS, CANVAS), dtype=bool)
    if not len(coords):
        return result
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = mask[y0:y1, x0:x1]
    h, w = cropped.shape
    scale = min(INNER / max(1, w), INNER / max(1, h))
    width = max(1, int(round(w * scale)))
    height = max(1, int(round(h * scale)))
    resized = Image.fromarray((cropped * 255).astype(np.uint8)).resize(
        (width, height), Image.Resampling.NEAREST
    )
    x = (CANVAS - width) // 2
    y = (CANVAS - height) // 2
    result[y : y + height, x : x + width] = np.asarray(resized) > 127
    return result


def edge_distance(mask: np.ndarray) -> np.ndarray:
    edge = mask ^ ndimage.binary_erosion(mask)
    if not edge.any():
        return np.full(mask.shape, 999.0, dtype=np.float32)
    return ndimage.distance_transform_edt(~edge).astype(np.float32)


def aspect_ratio(mask: np.ndarray) -> float:
    coords = np.argwhere(mask)
    if not len(coords):
        return 1.0
    height = max(1, int(coords[:, 0].max() - coords[:, 0].min() + 1))
    width = max(1, int(coords[:, 1].max() - coords[:, 1].min() + 1))
    return float(width / height)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hero", required=True)
    parser.add_argument("--photos", nargs=4, required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--preview", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    masks: list[np.ndarray] = []
    distance_maps: list[np.ndarray] = []
    rows: list[dict] = []
    originals: list[Image.Image] = []
    for index, path in enumerate(args.photos, start=1):
        image = Image.open(path).convert("RGB")
        raw, diagnostics = initial_foreground(image)
        cleaned = clean_mask(raw)
        normalized = normalize_mask(cleaned)
        if int(normalized.sum()) < 150:
            raise RuntimeError(f"Foreground segmentation failed for {path}")
        masks.append(normalized)
        distance_maps.append(edge_distance(normalized))
        originals.append(image)
        rows.append(
            {
                "view": index,
                "photo": str(path),
                "source_size": list(image.size),
                "normalized_area_pixels": int(normalized.sum()),
                "normalized_aspect": aspect_ratio(normalized),
                **diagnostics,
            }
        )

    mask_array = np.stack(masks).astype(np.uint8)
    distance_array = np.stack(distance_maps).astype(np.float32)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        masks=mask_array,
        edge_distances=distance_array,
        areas=mask_array.reshape(4, -1).sum(axis=1).astype(np.int32),
        aspects=np.array([aspect_ratio(mask) for mask in masks], dtype=np.float32),
    )

    cell = 256
    label_height = 22
    sheet = Image.new("RGB", (cell * 4, (cell + label_height) * 2), "white")
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(sheet)
    for column, (original, mask, row) in enumerate(zip(originals, masks, rows)):
        thumb = original.copy()
        thumb.thumbnail((cell - 8, cell - 8), Image.Resampling.LANCZOS)
        x = column * cell + (cell - thumb.width) // 2
        y = (cell - thumb.height) // 2
        sheet.paste(thumb, (x, y))
        draw.text((column * cell + 6, cell), f"Photo {column + 1}", fill="black", font=font)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
        mask_image = mask_image.resize((cell, cell), Image.Resampling.NEAREST)
        y2 = cell + label_height
        sheet.paste(mask_image, (column * cell, y2))
        draw.text(
            (column * cell + 6, y2 + cell),
            f"Mask area {row['normalized_area_pixels']}",
            fill="black",
            font=font,
        )
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.preview, quality=94)

    report = {
        "hero": args.hero,
        "canvas": CANVAS,
        "inner": INNER,
        "views": rows,
        "npz": str(args.output_npz),
        "preview": str(args.preview),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
