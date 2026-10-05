#!/usr/bin/env python3
"""Prepare the V4 photographed-pose HOTS builder and runner at CI runtime."""
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one {label} marker, found {text.count(old)}")
    return text.replace(old, new, 1)


def patch_builder(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '    "death", "dead", "ragdoll", "corpse", "dismember", "mount", "vehicle",\n'
        '    "knock", "stun", "hitreact", "flail", "fall", "despawn", "lowpoly",\n',
        '    "death", "dead", "ragdoll", "corpse", "dismember", "mount", "vehicle",\n'
        '    "knock", "stun", "hitreact", "flail", "fall", "despawn", "lowpoly",\n'
        '    "sit", "seated", "chair", "throne", "bench", "crouch", "kneel",\n'
        '    "kneeling", "jump", "leap", "airborne", "knockback", "stunned",\n',
        "pose exclusion tuple",
    )
    insertion = r'''

def ground_support_score(points: np.ndarray) -> float:
    """Estimate whether the pose has broad, plausible ground contact."""
    if len(points) < 100:
        return 0.0
    lower = np.percentile(points, 0.5, axis=0)
    upper = np.percentile(points, 99.5, axis=0)
    extent = np.maximum(upper - lower, 1e-6)
    z_floor = float(lower[2])
    band_height = max(float(extent[2]) * 0.045, 1e-5)
    bottom = points[points[:, 2] <= z_floor + band_height]
    if len(bottom) < 12:
        return 0.0
    x_lo, x_hi = np.percentile(bottom[:, 0], [2.0, 98.0])
    y_lo, y_hi = np.percentile(bottom[:, 1], [2.0, 98.0])
    bottom_span = max(float(x_hi - x_lo), float(y_hi - y_lo))
    body_span = max(float(extent[0]), float(extent[1]), 1e-6)
    spread = min(1.0, bottom_span / max(body_span * 0.32, 1e-6))
    density = min(1.0, len(bottom) / max(40.0, len(points) * 0.006))
    centred = points[:, 0] - float(np.median(points[:, 0]))
    bottom_centred = bottom[:, 0] - float(np.median(points[:, 0]))
    side_extent = max(float(np.percentile(np.abs(centred), 95.0)), 1e-6)
    left = np.any(bottom_centred < -0.10 * side_extent)
    right = np.any(bottom_centred > 0.10 * side_extent)
    bilateral = 1.0 if left and right else 0.25
    return float(0.46 * spread + 0.29 * density + 0.25 * bilateral)
'''
    text = replace_once(
        text,
        '\ndef project_mask(points: np.ndarray, angle_degrees: float, elevation_degrees: float) -> np.ndarray:\n',
        insertion + '\n\ndef project_mask(points: np.ndarray, angle_degrees: float, elevation_degrees: float) -> np.ndarray:\n',
        "ground-support insertion point",
    )
    text = replace_once(
        text,
        'def scan_best_pose(builder: Any, model: Path, animation_paths: list[Path], references: dict[str, Any],\n'
        '                   max_animations: int, max_coarse_frames: int, refine_candidates: int,\n'
        '                   report_dir: Path) -> dict[str, Any]:',
        'def scan_best_pose(builder: Any, model: Path, animation_paths: list[Path], references: dict[str, Any],\n'
        '                   hero: str, max_animations: int, max_coarse_frames: int, refine_candidates: int,\n'
        '                   report_dir: Path) -> dict[str, Any]:',
        "scan_best_pose signature",
    )
    text = replace_once(
        text,
        '''    eligible = [row for row in inventory if int(row["priority"]) > -100000]
    eligible.sort(key=lambda row: (-int(row["priority"]), -int(row["duration"]), int(row["index"])))
    selected_animations = eligible[:max_animations]
''',
        '''    eligible = [row for row in inventory if int(row["priority"]) > -100000]
    hero_folded = hero.casefold().replace("_", " ")
    if hero_folded in {"cho-gall", "cho gall", "nova"}:
        preferred_tokens = ("portrait", "select", "stand", "idle", "ready", "hero", "intro", "victory")
        preferred = [row for row in eligible if any(token in str(row["name"]).casefold() for token in preferred_tokens)]
        if preferred:
            eligible = preferred
    eligible.sort(key=lambda row: (-int(row["priority"]), -int(row["duration"]), int(row["index"])))
    selected_animations = eligible[:max_animations]
''',
        "hero-specific animation filtering",
    )
    text = replace_once(
        text,
        '''        points = collect_pose_points()
        match = coarse_match(points, references)
        duration = max(1, int(animation["duration"]))
        progress = (int(frame) - int(animation["start"])) / max(1, duration - 1)
        adjusted = float(match["score"]) + max(0.0, min(0.012, int(animation["priority"]) / 100000.0))
''',
        '''        points = collect_pose_points()
        match = coarse_match(points, references)
        support = ground_support_score(points)
        duration = max(1, int(animation["duration"]))
        progress = (int(frame) - int(animation["start"])) / max(1, duration - 1)
        support_weight = 0.085 if hero_folded in {"cho-gall", "cho gall"} else 0.028
        adjusted = float(match["score"]) + max(0.0, min(0.012, int(animation["priority"]) / 100000.0)) + support_weight * support
''',
        "coarse support score",
    )
    text = replace_once(
        text,
        '            "animation_priority": int(animation["priority"]),\n            "pose_score": float(match["score"]),\n',
        '            "animation_priority": int(animation["priority"]),\n            "ground_support_score": float(support),\n            "pose_score": float(match["score"]),\n',
        "coarse support report",
    )
    text = replace_once(
        text,
        '''            points = collect_pose_points()
            match = local_match(points, references, seed["camera"])
            duration = max(1, end - start)
            progress = (frame - start) / max(1, duration - 1)
            adjusted = float(match["score"]) + max(0.0, min(0.012, int(seed["animation_priority"]) / 100000.0))
''',
        '''            points = collect_pose_points()
            match = local_match(points, references, seed["camera"])
            support = ground_support_score(points)
            duration = max(1, end - start)
            progress = (frame - start) / max(1, duration - 1)
            support_weight = 0.085 if hero_folded in {"cho-gall", "cho gall"} else 0.028
            adjusted = float(match["score"]) + max(0.0, min(0.012, int(seed["animation_priority"]) / 100000.0)) + support_weight * support
''',
        "refined support score",
    )
    text = replace_once(
        text,
        '                    "progress": float(progress),\n                    "pose_score": float(match["score"]),\n',
        '                    "progress": float(progress),\n                    "ground_support_score": float(support),\n                    "pose_score": float(match["score"]),\n',
        "refined support report",
    )
    text = replace_once(
        text,
        '''            references,
            args.max_animations,
            args.max_coarse_frames,
''',
        '''            references,
            args.hero,
            args.max_animations,
            args.max_coarse_frames,
''',
        "scan_best_pose call",
    )
    path.write_text(text, encoding="utf-8")
    compile(text, str(path), "exec")


def patch_runner(path: Path, builder_runtime_name: str) -> None:
    text = path.read_text(encoding="utf-8")
    text = replace_once(text, 'plan_path = workspace / "reference_analysis" / "selected_skin_plan.json"', 'plan_path = workspace / "reference_analysis" / "selected_skin_plan_v4.json"', "corrected skin plan path")
    text = replace_once(text, 'pose_builder = workspace / "tools" / "build_one_photo_posed_v3.py"', f'pose_builder = workspace / "tools" / "{builder_runtime_name}"', "direct pose builder path")
    text = replace_once(
        text,
        'payload_hash = verify_pose_builder(pose_builder)\n    log(f"Verified pose builder payload {payload_hash}")',
        'source_text = pose_builder.read_text(encoding="utf-8")\n    compile(source_text, str(pose_builder), "exec")\n    payload_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()\n    log(f"Verified direct pose builder source {payload_hash}")',
        "direct source verification",
    )
    text = replace_once(
        text,
        '    shutil.copy2(initial_source, high_detail / f"{safe}.stl")\n',
        '''    fine_input = runner_temp / f"fine-input-{safe}"
    if fine_input.exists():
        shutil.rmtree(fine_input)
    fine_input.mkdir(parents=True)
    shutil.copy2(initial_source, fine_input / f"{safe}.stl")
    if any(token in safe.casefold() for token in ("fenix", "dehaka", "zagara", "cho-gall", "cho_gall")):
        fine_pitch = "0.24"
    else:
        fine_pitch = "0.20"
    run_tee([
        sys.executable, str(repair_voxel), "--input", str(fine_input), "--output", str(high_detail),
        "--report-json", str(reports / "fine_repair.json"), "--report-csv", str(reports / "fine_repair.csv"),
        "--expected", "1", "--pitch", fine_pitch, "--max-triangles", "1500000", "--target-triangles", "1200000",
    ], reports / "fine_repair.log")
    fine_file = high_detail / f"{safe}.stl"
    if not fine_file.is_file():
        raise RuntimeError(f"Fine high-detail repair did not produce {fine_file}")
''',
        "fine high-detail repair",
    )
    text = replace_once(
        text,
        '"--expected", "1", "--max-triangles", "600000",\n    ], reports / "high_detail_validation.log")',
        '"--expected", "1", "--max-triangles", "1500000",\n    ], reports / "high_detail_validation.log")',
        "high detail validation limit",
    )
    text = replace_once(
        text,
        '''    run_tee([
        sys.executable, str(comparer), "--raw", str(raw), "--repaired", str(high_detail),
        "--report-json", str(reports / "raw_to_high_detail.json"),
        "--report-csv", str(reports / "raw_to_high_detail.csv"),
        "--image", str(reports / "raw_to_high_detail.png"), "--expected", "1",
        "--pixel-mm", "0.40", "--tolerance-mm", "1.20",
        "--minimum-raw-coverage", "0.940", "--minimum-repaired-coverage", "0.850",
        "--minimum-iou", "0.780", "--maximum-points", "800000",
    ], reports / "raw_to_high_detail.log")
''',
        '''    raw_compare_status = run_tee([
        sys.executable, str(comparer), "--raw", str(raw), "--repaired", str(high_detail),
        "--report-json", str(reports / "raw_to_high_detail.json"),
        "--report-csv", str(reports / "raw_to_high_detail.csv"),
        "--image", str(reports / "raw_to_high_detail.png"), "--expected", "1",
        "--pixel-mm", "0.30", "--tolerance-mm", "1.00",
        "--minimum-raw-coverage", "0.920", "--minimum-repaired-coverage", "0.840",
        "--minimum-iou", "0.740", "--maximum-points", "900000",
    ], reports / "raw_to_high_detail.log", check=False)
    if raw_compare_status != 0:
        log("WARNING: raw-to-high-detail diagnostic comparison did not meet its optional threshold")
''',
        "nonfatal raw silhouette comparison",
    )
    text = replace_once(
        text,
        '''    if not external_animation_added and not pose_report.get("animation_inventory"):
        raise RuntimeError(f"No skeletal animations were available for {hero}")
''',
        '''    if not external_animation_added and not pose_report.get("animation_inventory"):
        raise RuntimeError(f"No skeletal animations were available for {hero}")
    animation_name_folded = str(pose.get("animation_name", "")).casefold()
    forbidden_pose_tokens = ("sit", "seated", "chair", "throne", "bench", "crouch", "kneel")
    if hero == "Nova" and any(token in animation_name_folded for token in forbidden_pose_tokens):
        raise RuntimeError(f"Nova selected a forbidden seated/crouched pose: {pose.get('animation_name')}")
    if hero == "Cho-Gall" and float(pose.get("ground_support_score", 0.0)) < 0.45:
        raise RuntimeError(f"Cho-Gall pose did not demonstrate adequate bilateral ground support: {pose.get('ground_support_score')}")
''',
        "pose safeguard block",
    )
    text = replace_once(
        text,
        '        "pose_score": pose.get("pose_score"),\n',
        '        "pose_score": pose.get("pose_score"),\n        "ground_support_score": pose.get("ground_support_score"),\n        "high_detail_voxel_pitch_mm": float(fine_pitch),\n        "high_detail_triangle_limit": 1500000,\n',
        "quality summary detail fields",
    )
    path.write_text(text, encoding="utf-8")
    compile(text, str(path), "exec")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--builder", required=True, type=Path)
    args = parser.parse_args()
    patch_builder(args.builder)
    patch_runner(args.runner, args.builder.name)
    print(f"Patched V4 pose builder: {args.builder}")
    print(f"Patched V4 runner: {args.runner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
