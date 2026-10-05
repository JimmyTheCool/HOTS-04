#!/usr/bin/env python3
"""Match a rigged HOTS hero to four JPG views and build posed printable STLs.

Run inside Blender 3.6. The selected skin M3 provides exact proportions and surface
geometry. One or more compatible M3A animation banks provide skeletal poses. Every
candidate frame is compared with four normalized screenshot silhouettes; the best
multi-view match is frozen into geometry before the proven watertight STL pipeline
runs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
from typing import Any, Iterable

import bpy
import numpy as np


COARSE_CANVAS = 96
COARSE_INNER = 84
COARSE_ANGLE_STEP = 15
COARSE_ANGLES = tuple(range(0, 360, COARSE_ANGLE_STEP))
COARSE_ELEVATIONS = (-20, -10, 0, 10, 20)
HARD_EXCLUDES = (
    "death", "dead", "ragdoll", "corpse", "dismember", "mount", "vehicle",
    "knock", "stun", "hitreact", "flail", "fall", "despawn", "lowpoly",
)


def blender_argv() -> list[str]:
    argv = sys.argv
    return argv[argv.index("--") + 1 :] if "--" in argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--builder", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--animation", action="append", default=[], type=Path)
    parser.add_argument("--hero", required=True)
    parser.add_argument("--skin-key", required=True)
    parser.add_argument("--reference-npz", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-height-mm", type=float, default=120.0)
    parser.add_argument("--base-height-mm", type=float, default=4.5)
    parser.add_argument("--base-margin-mm", type=float, default=5.0)
    parser.add_argument("--voxel-mm", type=float, default=0.30)
    parser.add_argument("--thin-part-mm", type=float, default=0.70)
    parser.add_argument("--max-triangles", type=int, default=600000)
    parser.add_argument("--max-animations", type=int, default=18)
    parser.add_argument("--max-coarse-frames", type=int, default=112)
    parser.add_argument("--refine-candidates", type=int, default=6)
    parser.add_argument("--keep-blend", action="store_true")
    return parser.parse_args(blender_argv())


def log(message: str) -> None:
    print(f"[POSE] {message}", flush=True)


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_.")
    return value or "hero"


def source_key(model: Path) -> str:
    value = model.stem.casefold()
    value = re.sub(r"^storm_hero_", "", value)
    return re.sub(r"_v[0-9]+$", "", value)


def load_builder(path: Path):
    spec = importlib.util.spec_from_file_location("hots_photo_pose_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load builder module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def downsample_mask(mask: np.ndarray, size: int = COARSE_CANVAS) -> np.ndarray:
    source_h, source_w = mask.shape
    yy = np.minimum(source_h - 1, np.floor(np.arange(size) * source_h / size).astype(int))
    xx = np.minimum(source_w - 1, np.floor(np.arange(size) * source_w / size).astype(int))
    return mask[np.ix_(yy, xx)].astype(bool)


def erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    result = mask.copy()
    h, w = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.zeros_like(mask)
            ys0 = max(0, -dy)
            ys1 = min(h, h - dy)
            xs0 = max(0, -dx)
            xs1 = min(w, w - dx)
            yd0 = max(0, dy)
            xd0 = max(0, dx)
            if ys1 > ys0 and xs1 > xs0:
                shifted[yd0 : yd0 + (ys1 - ys0), xd0 : xd0 + (xs1 - xs0)] = mask[ys0:ys1, xs0:xs1]
            result &= shifted
    return result


def dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    result = mask.copy()
    h, w = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius + 1:
                continue
            ys0 = max(0, -dy)
            ys1 = min(h, h - dy)
            xs0 = max(0, -dx)
            xs1 = min(w, w - dx)
            yd0 = max(0, dy)
            xd0 = max(0, dx)
            if ys1 > ys0 and xs1 > xs0:
                result[yd0 : yd0 + (ys1 - ys0), xd0 : xd0 + (xs1 - xs0)] |= mask[ys0:ys1, xs0:xs1]
    return result


def span_fill(mask: np.ndarray) -> np.ndarray:
    """Cheap silhouette fill that avoids filling every gap between separated limbs."""
    h, w = mask.shape
    row_fill = np.zeros_like(mask)
    col_fill = np.zeros_like(mask)
    for y in range(h):
        xs = np.flatnonzero(mask[y])
        if len(xs) >= 2:
            row_fill[y, xs[0] : xs[-1] + 1] = True
    for x in range(w):
        ys = np.flatnonzero(mask[:, x])
        if len(ys) >= 2:
            col_fill[ys[0] : ys[-1] + 1, x] = True
    return mask | (row_fill & col_fill)


def normalize_binary_mask(mask: np.ndarray, canvas: int, inner: int) -> np.ndarray:
    coords = np.argwhere(mask)
    result = np.zeros((canvas, canvas), dtype=bool)
    if not len(coords):
        return result
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = mask[y0:y1, x0:x1]
    height, width = cropped.shape
    scale = min(inner / max(1, width), inner / max(1, height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    yy = np.minimum(height - 1, np.floor(np.arange(new_h) * height / new_h).astype(int))
    xx = np.minimum(width - 1, np.floor(np.arange(new_w) * width / new_w).astype(int))
    resized = cropped[np.ix_(yy, xx)]
    x = (canvas - new_w) // 2
    y = (canvas - new_h) // 2
    result[y : y + new_h, x : x + new_w] = resized
    return result


def mask_aspect(mask: np.ndarray) -> float:
    coords = np.argwhere(mask)
    if not len(coords):
        return 1.0
    height = max(1, int(coords[:, 0].max() - coords[:, 0].min() + 1))
    width = max(1, int(coords[:, 1].max() - coords[:, 1].min() + 1))
    return float(width / height)


def approximate_distance_to_edge(mask: np.ndarray, max_distance: int = 16) -> np.ndarray:
    edge = mask ^ erode(mask, 1)
    distance = np.full(mask.shape, float(max_distance), dtype=np.float32)
    current = edge.copy()
    distance[current] = 0.0
    reached = current.copy()
    for step in range(1, max_distance + 1):
        current = dilate(current, 1)
        new = current & ~reached
        distance[new] = float(step)
        reached |= new
        if reached.all():
            break
    return distance


def load_references(path: Path) -> dict[str, Any]:
    data = np.load(path)
    masks_full = data["masks"].astype(bool)
    masks = np.stack([downsample_mask(mask) for mask in masks_full])
    return {
        "masks": masks,
        "areas": np.array([max(1, int(mask.sum())) for mask in masks], dtype=np.int32),
        "aspects": np.array([mask_aspect(mask) for mask in masks], dtype=np.float32),
        "edge_distances": np.stack([approximate_distance_to_edge(mask) for mask in masks]),
        "full_masks": masks_full,
    }


def collect_pose_points(max_points: int = 90000) -> np.ndarray:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    chunks: list[np.ndarray] = []
    for original in list(bpy.context.scene.objects):
        if original.type != "MESH" or original.hide_render or original.hide_get():
            continue
        lower_name = original.name.casefold()
        if any(token in lower_name for token in ("collision", "physics", "volume", "shadow", "bounds", "selection")):
            continue
        evaluated = original.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
        try:
            if mesh is None or not mesh.vertices:
                continue
            vertex_data = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", vertex_data)
            coords = vertex_data.reshape((-1, 3))

            if mesh.polygons:
                try:
                    centre_data = np.empty(len(mesh.polygons) * 3, dtype=np.float32)
                    mesh.polygons.foreach_get("center", centre_data)
                    centres = centre_data.reshape((-1, 3))
                    if len(centres) > 30000:
                        indices = np.linspace(0, len(centres) - 1, 30000, dtype=int)
                        centres = centres[indices]
                    coords = np.concatenate((coords, centres), axis=0)
                except Exception:
                    # Vertices alone are sufficient; polygon centres only improve
                    # low-poly silhouette density.
                    pass

            matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
            homogeneous = np.concatenate((coords.astype(np.float64), np.ones((len(coords), 1))), axis=1)
            world = homogeneous @ matrix.T
            chunks.append(world[:, :3].astype(np.float32))
        finally:
            evaluated.to_mesh_clear()

    if not chunks:
        raise RuntimeError("No evaluated mesh points were available for pose scoring")
    points = np.concatenate(chunks, axis=0)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if not len(points):
        raise RuntimeError("All evaluated pose points were non-finite")
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
    return points


def project_mask(points: np.ndarray, angle_degrees: float, elevation_degrees: float) -> np.ndarray:
    theta = math.radians(angle_degrees)
    elevation = math.radians(elevation_degrees)
    c, s = math.cos(theta), math.sin(theta)
    horizontal = points[:, 0] * c - points[:, 1] * s
    depth = points[:, 0] * s + points[:, 1] * c
    vertical = points[:, 2] * math.cos(elevation) - depth * math.sin(elevation)

    projected = np.column_stack((horizontal, vertical))
    lower = np.percentile(projected, 0.02, axis=0)
    upper = np.percentile(projected, 99.98, axis=0)
    keep = np.all((projected >= lower) & (projected <= upper), axis=1)
    if int(keep.sum()) > 100:
        projected = projected[keep]
    minimum = projected.min(axis=0)
    maximum = projected.max(axis=0)
    extent = np.maximum(maximum - minimum, 1e-8)
    scale = min((COARSE_INNER - 1) / extent[0], (COARSE_INNER - 1) / extent[1])
    pixels = (projected - minimum) * scale
    pixels[:, 0] += (COARSE_CANVAS - extent[0] * scale) / 2.0
    pixels[:, 1] += (COARSE_CANVAS - extent[1] * scale) / 2.0
    x = np.clip(np.rint(pixels[:, 0]).astype(np.int32), 0, COARSE_CANVAS - 1)
    y = np.clip(COARSE_CANVAS - 1 - np.rint(pixels[:, 1]).astype(np.int32), 0, COARSE_CANVAS - 1)
    mask = np.zeros((COARSE_CANVAS, COARSE_CANVAS), dtype=bool)
    mask[y, x] = True
    radius = 2 if len(points) < 22000 else 1
    mask = dilate(mask, radius)
    mask = span_fill(mask)
    mask = dilate(erode(dilate(mask, 1), 1), 1)
    return normalize_binary_mask(mask, COARSE_CANVAS, COARSE_INNER)


def shape_score(reference: np.ndarray, reference_area: int, reference_aspect: float,
                reference_edge_distance: np.ndarray, candidate: np.ndarray) -> float:
    candidate_area = max(1, int(candidate.sum()))
    intersection = int(np.logical_and(reference, candidate).sum())
    union = int(np.logical_or(reference, candidate).sum())
    iou = intersection / max(1, union)
    reference_coverage = intersection / max(1, reference_area)
    candidate_coverage = intersection / candidate_area
    candidate_edge = candidate ^ erode(candidate, 1)
    if candidate_edge.any():
        chamfer = float(reference_edge_distance[candidate_edge].mean())
    else:
        chamfer = 20.0
    edge_score = math.exp(-chamfer / 4.8)
    aspect = mask_aspect(candidate)
    aspect_score = math.exp(-abs(math.log(max(1e-6, aspect / max(1e-6, reference_aspect)))))
    return float(
        0.54 * iou
        + 0.19 * reference_coverage
        + 0.10 * candidate_coverage
        + 0.12 * edge_score
        + 0.05 * aspect_score
    )


def score_view(references: dict[str, Any], view: int, mask: np.ndarray) -> float:
    return shape_score(
        references["masks"][view],
        int(references["areas"][view]),
        float(references["aspects"][view]),
        references["edge_distances"][view],
        mask,
    )


def coarse_match(points: np.ndarray, references: dict[str, Any]) -> dict[str, Any]:
    cache: dict[tuple[int, int], np.ndarray] = {}
    first_view: list[tuple[float, int, int]] = []
    for elevation in COARSE_ELEVATIONS:
        for angle in COARSE_ANGLES:
            mask = project_mask(points, angle, elevation)
            cache[(elevation, angle)] = mask
            first_view.append((score_view(references, 0, mask), elevation, angle))
    first_view.sort(reverse=True)

    best: dict[str, Any] | None = None
    # Use the first view to shortlist camera fits, then enforce consistency across
    # the three remaining 90-degree views in either rotation direction.
    for first_score, elevation, base_angle in first_view[:18]:
        for direction in (1, -1):
            angles = [int((base_angle + direction * 90 * index) % 360) for index in range(4)]
            per_view = [first_score]
            for view in range(1, 4):
                per_view.append(score_view(references, view, cache[(elevation, angles[view])]))
            mean = float(np.mean(per_view))
            consistency = float(np.std(per_view))
            score = mean - 0.06 * consistency
            candidate = {
                "score": score,
                "mean_score": mean,
                "consistency_std": consistency,
                "base_angle": base_angle,
                "direction": direction,
                "elevation": elevation,
                "angles": angles,
                "per_view_scores": per_view,
                "masks": [cache[(elevation, angle)] for angle in angles],
            }
            if best is None or score > float(best["score"]):
                best = candidate
    if best is None:
        raise RuntimeError("Could not score projected pose")
    return best


def local_match(points: np.ndarray, references: dict[str, Any], seed: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    base_offsets = (-10, -5, 0, 5, 10)
    elevation_offsets = (-6, 0, 6)
    for elevation_offset in elevation_offsets:
        elevation = int(max(-35, min(35, int(seed["elevation"]) + elevation_offset)))
        for base_offset in base_offsets:
            base = int((int(seed["base_angle"]) + base_offset) % 360)
            angles = [int((base + int(seed["direction"]) * 90 * index) % 360) for index in range(4)]
            masks = [project_mask(points, angle, elevation) for angle in angles]
            per_view = [score_view(references, index, mask) for index, mask in enumerate(masks)]
            mean = float(np.mean(per_view))
            consistency = float(np.std(per_view))
            score = mean - 0.06 * consistency
            candidate = {
                "score": score,
                "mean_score": mean,
                "consistency_std": consistency,
                "base_angle": base,
                "direction": int(seed["direction"]),
                "elevation": elevation,
                "angles": angles,
                "per_view_scores": per_view,
                "masks": masks,
            }
            if best is None or score > float(best["score"]):
                best = candidate
    if best is None:
        raise RuntimeError("Could not refine projected pose")
    return best


def animation_priority(name: str) -> int:
    lower = name.casefold()
    if any(token in lower for token in HARD_EXCLUDES):
        return -100000
    score = 0
    weighted = (
        ("portrait", 1300), ("select", 1200), ("hero", 250), ("stand", 1050),
        ("idle", 1000), ("ready", 900), ("intro", 780), ("victory", 650),
        ("taunt", 420), ("dance", 280), ("walk", 120), ("run", 40),
        ("attack", -160), ("spell", -170), ("ability", -170), ("channel", -100),
    )
    for token, value in weighted:
        if token in lower:
            score += value
    if not lower.strip():
        score -= 40
    return score


def animation_inventory(source_by_index: dict[int, str | None]) -> list[dict[str, Any]]:
    scene = bpy.context.scene
    rows: list[dict[str, Any]] = []
    names_seen: dict[str, int] = {}
    for index, animation in enumerate(list(getattr(scene, "m3_animations", []))):
        name = str(getattr(animation, "name", ""))
        start = int(getattr(animation, "startFrame", scene.frame_start))
        end = int(getattr(animation, "exlusiveEndFrame", scene.frame_end + 1))
        end = max(start + 1, end)
        occurrence = names_seen.get(name, 0)
        names_seen[name] = occurrence + 1
        rows.append(
            {
                "index": index,
                "name": name,
                "name_occurrence": occurrence,
                "start": start,
                "end_exclusive": end,
                "duration": end - start,
                "priority": animation_priority(name),
                "source_animation": source_by_index.get(index),
            }
        )
    return rows


def sample_frames(animation: dict[str, Any]) -> list[int]:
    start = int(animation["start"])
    end = int(animation["end_exclusive"])
    duration = max(1, end - start)
    if duration <= 8:
        return list(range(start, end))
    fractions = (0.04, 0.16, 0.30, 0.46, 0.62, 0.78, 0.94)
    frames = {start + min(duration - 1, max(0, int(round((duration - 1) * fraction)))) for fraction in fractions}
    frames.add(min(end - 1, start + 1))
    return sorted(frames)


def set_animation(index: int, frame: int) -> None:
    scene = bpy.context.scene
    scene.m3_animation_index = index
    scene.frame_set(frame)
    bpy.context.view_layer.update()


def scan_best_pose(builder: Any, model: Path, animation_paths: list[Path], references: dict[str, Any],
                   max_animations: int, max_coarse_frames: int, refine_candidates: int,
                   report_dir: Path) -> dict[str, Any]:
    builder.reset_scene()
    builder.enable_m3_addon()
    log(f"Importing selected skin for pose search: {model}")
    builder.import_m3(model)
    bpy.context.view_layer.update()
    armature = next((obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"), None)
    if armature is None:
        raise RuntimeError("Selected M3 did not create an armature; photographed pose cannot be reconstructed")

    baseline_match = None
    try:
        if hasattr(bpy.context.scene, "m3_animation_index"):
            bpy.context.scene.m3_animation_index = -1
        bpy.context.scene.frame_set(0)
        bpy.context.view_layer.update()
        baseline_match = coarse_match(collect_pose_points(), references)
        log(f"Unposed/rest-pose baseline score: {baseline_match['score']:.4f}")
    except Exception as exc:
        log(f"WARNING: could not score rest-pose baseline: {exc}")

    source_by_index: dict[int, str | None] = {
        index: None for index, _ in enumerate(list(getattr(bpy.context.scene, "m3_animations", [])))
    }
    animation_imports: list[dict[str, Any]] = []
    for animation_path in animation_paths:
        before = len(list(getattr(bpy.context.scene, "m3_animations", [])))
        entry = {"path": str(animation_path), "before_count": before}
        try:
            log(f"Importing pose bank: {animation_path}")
            builder.import_m3(animation_path, armature=armature)
            bpy.context.view_layer.update()
            after = len(list(getattr(bpy.context.scene, "m3_animations", [])))
            entry.update({"after_count": after, "status": "success", "added": max(0, after - before)})
            for index in range(before, after):
                source_by_index[index] = str(animation_path)
        except Exception as exc:
            entry.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
            log(f"WARNING: animation bank failed to import: {animation_path}: {exc}")
        animation_imports.append(entry)

    inventory = animation_inventory(source_by_index)
    if not inventory:
        raise RuntimeError("No skeletal animations were available after importing the selected skin and animation banks")
    eligible = [row for row in inventory if int(row["priority"]) > -100000]
    eligible.sort(key=lambda row: (-int(row["priority"]), -int(row["duration"]), int(row["index"])))
    selected_animations = eligible[:max_animations]
    if not selected_animations:
        raise RuntimeError("All imported animations were excluded as death/ragdoll/non-display poses")

    frame_plan: list[tuple[dict[str, Any], int]] = []
    for animation in selected_animations:
        for frame in sample_frames(animation):
            frame_plan.append((animation, frame))
    frame_plan.sort(key=lambda item: (-int(item[0]["priority"]), int(item[0]["index"]), item[1]))
    frame_plan = frame_plan[:max_coarse_frames]
    log(f"Pose search will test {len(frame_plan)} frames from {len(selected_animations)} animations")

    coarse_results: list[dict[str, Any]] = []
    started = time.monotonic()
    for number, (animation, frame) in enumerate(frame_plan, start=1):
        set_animation(int(animation["index"]), int(frame))
        points = collect_pose_points()
        match = coarse_match(points, references)
        duration = max(1, int(animation["duration"]))
        progress = (int(frame) - int(animation["start"])) / max(1, duration - 1)
        adjusted = float(match["score"]) + max(0.0, min(0.012, int(animation["priority"]) / 100000.0))
        row = {
            "animation_index": int(animation["index"]),
            "animation_name": str(animation["name"]),
            "name_occurrence": int(animation["name_occurrence"]),
            "animation_source": animation.get("source_animation"),
            "animation_start": int(animation["start"]),
            "animation_end_exclusive": int(animation["end_exclusive"]),
            "frame": int(frame),
            "progress": float(progress),
            "animation_priority": int(animation["priority"]),
            "pose_score": float(match["score"]),
            "adjusted_score": adjusted,
            "camera": {key: value for key, value in match.items() if key != "masks"},
        }
        coarse_results.append(row)
        if number == 1 or number % 8 == 0:
            best_now = max(coarse_results, key=lambda item: float(item["adjusted_score"]))
            log(
                f"Coarse {number}/{len(frame_plan)}; best {best_now['animation_name']} "
                f"frame {best_now['frame']} score {best_now['pose_score']:.4f}"
            )

    coarse_results.sort(key=lambda row: float(row["adjusted_score"]), reverse=True)
    seeds: list[dict[str, Any]] = []
    seen_seed: set[tuple[int, int]] = set()
    for row in coarse_results:
        key = (int(row["animation_index"]), int(row["frame"]))
        if key in seen_seed:
            continue
        seeds.append(row)
        seen_seed.add(key)
        if len(seeds) >= refine_candidates:
            break

    refined_results: list[dict[str, Any]] = []
    evaluated: set[tuple[int, int]] = set()
    for seed in seeds:
        start = int(seed["animation_start"])
        end = int(seed["animation_end_exclusive"])
        seed_frame = int(seed["frame"])
        for frame in range(max(start, seed_frame - 4), min(end, seed_frame + 5)):
            key = (int(seed["animation_index"]), frame)
            if key in evaluated:
                continue
            evaluated.add(key)
            set_animation(int(seed["animation_index"]), frame)
            points = collect_pose_points()
            match = local_match(points, references, seed["camera"])
            duration = max(1, end - start)
            progress = (frame - start) / max(1, duration - 1)
            adjusted = float(match["score"]) + max(0.0, min(0.012, int(seed["animation_priority"]) / 100000.0))
            refined_results.append(
                {
                    **{key_name: seed[key_name] for key_name in (
                        "animation_index", "animation_name", "name_occurrence", "animation_source",
                        "animation_start", "animation_end_exclusive", "animation_priority",
                    )},
                    "frame": frame,
                    "progress": float(progress),
                    "pose_score": float(match["score"]),
                    "adjusted_score": adjusted,
                    "camera": {key_name: value for key_name, value in match.items() if key_name != "masks"},
                    "masks": match["masks"],
                }
            )

    if not refined_results:
        raise RuntimeError("Pose refinement produced no candidates")
    refined_results.sort(key=lambda row: float(row["adjusted_score"]), reverse=True)
    winner = refined_results[0]
    runner_up = refined_results[1] if len(refined_results) > 1 else None
    confidence_margin = float(winner["pose_score"]) - (float(runner_up["pose_score"]) if runner_up else 0.0)

    report_dir.mkdir(parents=True, exist_ok=True)
    for index, mask in enumerate(winner.pop("masks"), start=1):
        path = report_dir / f"matched_pose_view_{index}.pgm"
        with path.open("wb") as handle:
            handle.write(f"P5\n{mask.shape[1]} {mask.shape[0]}\n255\n".encode("ascii"))
            handle.write((mask.astype(np.uint8) * 255).tobytes())

    result = {
        "selected": winner,
        "confidence_margin": confidence_margin,
        "coarse_frames_tested": len(coarse_results),
        "refined_frames_tested": len(refined_results),
        "elapsed_pose_search_seconds": round(time.monotonic() - started, 2),
        "rest_pose_baseline": ({key: value for key, value in baseline_match.items() if key != "masks"} if baseline_match else None),
        "improvement_over_rest_pose": (float(winner["pose_score"]) - float(baseline_match["score"]) if baseline_match else None),
        "animation_imports": animation_imports,
        "animation_inventory": inventory,
        "top_coarse_candidates": coarse_results[:20],
        "top_refined_candidates": [
            {key: value for key, value in row.items() if key != "masks"}
            for row in refined_results[:20]
        ],
    }
    (report_dir / "pose_match.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(
        f"Selected pose: {winner['animation_name']} frame {winner['frame']} "
        f"score {winner['pose_score']:.4f}, source={winner.get('animation_source')}"
    )
    return result


def forced_pose_selector(selected: dict[str, Any]):
    def choose_pose() -> dict[str, Any]:
        scene = bpy.context.scene
        animations = list(getattr(scene, "m3_animations", []))
        if not animations:
            raise RuntimeError("Forced photographed pose requested, but rebuilt scene contains no animations")
        target_name = str(selected["animation_name"])
        target_occurrence = int(selected.get("name_occurrence", 0))
        matches = [index for index, anim in enumerate(animations) if str(getattr(anim, "name", "")) == target_name]
        if matches:
            index = matches[min(target_occurrence, len(matches) - 1)]
        else:
            folded = target_name.casefold()
            fuzzy = [
                index for index, anim in enumerate(animations)
                if folded and folded in str(getattr(anim, "name", "")).casefold()
            ]
            index = fuzzy[0] if fuzzy else min(int(selected["animation_index"]), len(animations) - 1)
        scene.m3_animation_index = index
        animation = animations[index]
        start = int(getattr(animation, "startFrame", scene.frame_start))
        end = int(getattr(animation, "exlusiveEndFrame", scene.frame_end + 1))
        end = max(start + 1, end)
        progress = float(selected.get("progress", 0.0))
        frame = start + int(round(progress * max(0, end - start - 1)))
        frame = min(end - 1, max(start, frame))
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        return {
            "animation": str(getattr(animation, "name", "")),
            "index": index,
            "frame": frame,
            "reason": "four-JPG multi-view photographed-pose match",
            "target_animation": target_name,
            "target_progress": progress,
            "pose_match_score": float(selected["pose_score"]),
            "camera_match": selected["camera"],
            "animation_count": len(animations),
        }
    return choose_pose


def main() -> int:
    args = parse_args()
    args.builder = args.builder.resolve()
    args.model = args.model.resolve()
    args.reference_npz = args.reference_npz.resolve()
    args.output = args.output.resolve()
    animation_paths = [path.resolve() for path in args.animation if path.is_file()]
    for required in (args.builder, args.model, args.reference_npz):
        if not required.is_file():
            raise FileNotFoundError(required)

    hero_safe = safe_name(args.hero)
    raw_dir = args.output / "Raw"
    initial_dir = args.output / "Initial"
    reports_dir = args.output / "Reports"
    for directory in (raw_dir, initial_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    builder = load_builder(args.builder)
    references = load_references(args.reference_npz)
    report: dict[str, Any] = {
        "hero": args.hero,
        "hero_safe": hero_safe,
        "skin_key": args.skin_key,
        "model_path": str(args.model),
        "candidate_animation_paths": [str(path) for path in animation_paths],
        "source_key": source_key(args.model),
    }
    try:
        pose_result = scan_best_pose(
            builder,
            args.model,
            animation_paths,
            references,
            args.max_animations,
            args.max_coarse_frames,
            args.refine_candidates,
            reports_dir,
        )
        selected = pose_result["selected"]
        selected_animation = selected.get("animation_source")
        selected_animation_path = Path(selected_animation) if selected_animation else None
        if selected_animation_path and not selected_animation_path.is_file():
            raise FileNotFoundError(f"Selected animation source disappeared: {selected_animation_path}")

        builder.find_required_animation = (
            lambda model_path, source_key_value: selected_animation_path
        )
        builder.choose_pose = forced_pose_selector(selected)
        builder_args = argparse.Namespace(
            target_height_mm=args.target_height_mm,
            base_height_mm=args.base_height_mm,
            base_margin_mm=args.base_margin_mm,
            voxel_mm=args.voxel_mm,
            thin_part_mm=args.thin_part_mm,
            max_triangles=args.max_triangles,
            skip_animation=False,
            keep_blend=args.keep_blend,
        )
        result = builder.process_source_group(
            report["source_key"],
            [args.model],
            args.output,
            builder_args,
        )
        detail_source = Path(result["detail"]["path"])
        initial_source = Path(result["print_ready"]["path"])
        detail_target = raw_dir / f"{hero_safe}.stl"
        initial_target = initial_dir / f"{hero_safe}.stl"
        shutil.copy2(detail_source, detail_target)
        shutil.copy2(initial_source, initial_target)
        report.update(
            {
                "status": "success",
                "pose_match": pose_result,
                "selected_animation_path": str(selected_animation_path) if selected_animation_path else None,
                "settings": vars(builder_args),
                "detail_stl": str(detail_target),
                "initial_print_stl": str(initial_target),
                "builder_result": result,
            }
        )
        log(f"Completed posed STL build for {args.hero}")
    except Exception as exc:
        report.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        print(report["traceback"], flush=True)
        (reports_dir / "photo_posed_build.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 2

    (reports_dir / "photo_posed_build.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
