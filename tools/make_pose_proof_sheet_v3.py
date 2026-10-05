#!/usr/bin/env python3
"""Create a human-readable four-view proof sheet for a selected HOTS pose."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.convert("RGB")
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(copy, ((width - copy.width) // 2, (height - copy.height) // 2))
    return canvas


def mask_image(mask: np.ndarray, size: int) -> Image.Image:
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").convert("RGB")
    return image.resize((size, size), Image.Resampling.NEAREST)


def overlay(reference: np.ndarray, candidate: np.ndarray, size: int) -> Image.Image:
    if candidate.shape != reference.shape:
        candidate_image = Image.fromarray((candidate.astype(np.uint8) * 255), mode="L")
        candidate_image = candidate_image.resize((reference.shape[1], reference.shape[0]), Image.Resampling.NEAREST)
        candidate = np.asarray(candidate_image) > 127
    rgb = np.full((*reference.shape, 3), 245, dtype=np.uint8)
    only_ref = reference & ~candidate
    only_candidate = candidate & ~reference
    overlap_pixels = reference & candidate
    rgb[only_ref] = (230, 70, 70)
    rgb[only_candidate] = (50, 170, 220)
    rgb[overlap_pixels] = (55, 60, 65)
    return Image.fromarray(rgb, mode="RGB").resize((size, size), Image.Resampling.NEAREST)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hero", required=True)
    parser.add_argument("--photos", nargs=4, required=True, type=Path)
    parser.add_argument("--reference-npz", required=True, type=Path)
    parser.add_argument("--matched-dir", required=True, type=Path)
    parser.add_argument("--pose-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    references = np.load(args.reference_npz)["masks"].astype(bool)
    candidates = []
    for index in range(1, 5):
        path = args.matched_dir / f"matched_pose_view_{index}.pgm"
        candidates.append(np.asarray(Image.open(path).convert("L")) > 127)
    pose_data = json.loads(args.pose_report.read_text(encoding="utf-8"))
    selected = pose_data["selected"]

    cell = 280
    header = 82
    label = 24
    rows = 4
    sheet = Image.new("RGB", (cell * 4, header + rows * (cell + label)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    improvement = pose_data.get("improvement_over_rest_pose")
    improvement_text = "n/a" if improvement is None else f"{float(improvement):+.4f}"
    title = (
        f"{args.hero} | photographed-pose match | animation: {selected['animation_name']} | "
        f"frame {selected['frame']} | score {selected['pose_score']:.4f} | "
        f"improvement over rest/T-pose {improvement_text}"
    )
    draw.text((12, 12), title, fill="black", font=font)
    draw.text(
        (12, 34),
        f"Camera: base {selected['camera']['base_angle']} deg, direction {selected['camera']['direction']}, "
        f"elevation {selected['camera']['elevation']} deg | red=photo only, blue=model only, dark=overlap",
        fill="black",
        font=font,
    )

    row_names = ("Original JPG", "Segmented photo silhouette", "Selected posed mesh silhouette", "Pose overlap")
    for column in range(4):
        original = fit_image(Image.open(args.photos[column]), cell, cell)
        ref_image = mask_image(references[column], cell)
        candidate_image = mask_image(candidates[column], cell)
        overlap_image = overlay(references[column], candidates[column], cell)
        for row, image in enumerate((original, ref_image, candidate_image, overlap_image)):
            y = header + row * (cell + label)
            sheet.paste(image, (column * cell, y))
            text = f"{row_names[row]} - view {column + 1}"
            if row == 3:
                text += f" - score {selected['camera']['per_view_scores'][column]:.4f}"
            draw.text((column * cell + 6, y + cell + 5), text, fill="black", font=font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=95)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
