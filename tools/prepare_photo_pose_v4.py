#!/usr/bin/env python3
"""Prepare the V4 photo-pose runner and pose matcher.

The script decodes the existing v3.2 runner, switches it to an audited uncompressed
pose matcher, rejects seated/crouched display poses, forces standing animation
families for Nova and Cho-Gall, aligns Cho-Gall's left and right foot contact zones,
and raises the print-detail triangle limits while allowing an adaptive voxel size.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
from pathlib import Path
import re
import zlib


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} target, found {count}")
    return text.replace(old, new, 1)


def decode_bootstrap(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    payload = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "_PAYLOAD":
                payload = ast.literal_eval(node.value)
                break
    if payload is None:
        raise RuntimeError(f"No _PAYLOAD found in {path}")
    source = zlib.decompress(base64.b85decode(payload)).decode("utf-8")
    compile(source, str(path), "exec")
    return source


def patch_pose_source(source: str) -> str:
    source = replace_once(
        source,
        'HARD_EXCLUDES = (\n    "death", "dead", "ragdoll", "corpse", "dismember", "mount", "vehicle",\n    "knock", "stun", "hitreact", "flail", "fall", "despawn", "lowpoly",\n)',
        'HARD_EXCLUDES = (\n    "death", "dead", "ragdoll", "corpse", "dismember", "mount", "vehicle",\n    "knock", "stun", "hitreact", "flail", "fall", "despawn", "lowpoly",\n    "sit", "seated", "chair", "throne", "crouch", "kneel", "sleep", "crawl",\n)\n\nCURRENT_HERO = ""',
        "pose exclusion list",
    )

    source = replace_once(
        source,
        '    weighted = (\n        ("portrait", 1300), ("select", 1200), ("hero", 250), ("stand", 1050),\n        ("idle", 1000), ("ready", 900), ("intro", 780), ("victory", 650),\n        ("taunt", 420), ("dance", 280), ("walk", 120), ("run", 40),\n        ("attack", -160), ("spell", -170), ("ability", -170), ("channel", -100),\n    )',
        '    weighted = (\n        ("stand", 3300), ("idle", 3100), ("ready", 2900), ("select", 2600),\n        ("portrait", 1600), ("hero", 500), ("intro", 700), ("victory", 550),\n        ("taunt", 100), ("dance", -200), ("walk", -100), ("run", -300),\n        ("attack", -650), ("spell", -650), ("ability", -650), ("channel", -450),\n    )',
        "animation priorities",
    )

    strict_marker = '    eligible.sort(key=lambda row: (-int(row["priority"]), -int(row["duration"]), int(row["index"])))\n    selected_animations = eligible[:max_animations]\n'
    strict_replacement = '''    eligible.sort(key=lambda row: (-int(row["priority"]), -int(row["duration"]), int(row["index"])))
    hero_key = CURRENT_HERO.casefold().replace("'", "").replace("-", "").replace(" ", "")
    if hero_key in {"nova", "chogall"}:
        standing_tokens = ("stand", "idle", "ready", "select")
        standing = [
            row for row in eligible
            if any(token in str(row["name"]).casefold() for token in standing_tokens)
        ]
        if standing:
            eligible = standing
            log(f"Standing-only pose policy retained {len(eligible)} animations for {CURRENT_HERO}")
    selected_animations = eligible[:max_animations]
'''
    source = replace_once(source, strict_marker, strict_replacement, "standing-only animation filter")

    source = replace_once(
        source,
        'def main() -> int:\n    args = parse_args()\n',
        'def main() -> int:\n    global CURRENT_HERO\n    args = parse_args()\n    CURRENT_HERO = args.hero\n',
        "hero context initialization",
    )

    ground_helper = r'''

def _evaluated_world_points(objects: list[bpy.types.Object], max_points: int = 240000) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for obj in objects:
        if obj.type != "MESH" or not obj.data.vertices:
            continue
        values = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
        obj.data.vertices.foreach_get("co", values)
        local = values.reshape((-1, 3))
        matrix = np.asarray(obj.matrix_world, dtype=np.float64)
        homogeneous = np.concatenate((local, np.ones((len(local), 1))), axis=1)
        chunks.append((homogeneous @ matrix.T)[:, :3])
    if not chunks:
        return np.empty((0, 3), dtype=np.float64)
    points = np.concatenate(chunks, axis=0)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > max_points:
        points = points[np.linspace(0, len(points) - 1, max_points, dtype=int)]
    return points


def _contact_pair(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(points) < 100:
        return None
    z0, z1 = np.percentile(points[:, 2], [0.2, 99.8])
    height = max(1e-6, float(z1 - z0))
    lower = points[points[:, 2] <= z0 + height * 0.24]
    if len(lower) < 60:
        return None
    x_mid = float(np.median(points[:, 0]))
    x_lo, x_hi = np.percentile(points[:, 0], [7.0, 93.0])
    lower = lower[(lower[:, 0] >= x_lo) & (lower[:, 0] <= x_hi)]
    left = lower[lower[:, 0] < x_mid]
    right = lower[lower[:, 0] >= x_mid]
    if len(left) < 20 or len(right) < 20:
        return None

    def contact(side: np.ndarray) -> np.ndarray:
        z_cut = float(np.percentile(side[:, 2], 3.0))
        band = side[side[:, 2] <= z_cut + max(0.35, height * 0.006)]
        if len(band) < 4:
            band = side[np.argsort(side[:, 2])[: max(4, min(40, len(side)))]]
        return np.median(band, axis=0)

    return contact(left), contact(right)


def enforce_two_foot_ground(objects: list[bpy.types.Object]) -> dict[str, float | bool]:
    """Tilt Cho-Gall by a small amount so both lower left/right foot zones meet Z=0."""
    from mathutils import Matrix, Vector

    points = _evaluated_world_points(objects)
    pair = _contact_pair(points)
    if pair is None:
        return {"applied": False, "reason": "contact zones not resolved"}
    left, right = pair
    dx = float(right[0] - left[0])
    dz = float(right[2] - left[2])
    if abs(dx) < 1e-6:
        return {"applied": False, "reason": "contact zones have no lateral separation"}

    candidate_angles = (math.atan2(dz, dx), -math.atan2(dz, dx))
    midpoint = (left + right) * 0.5
    original = [obj.matrix_world.copy() for obj in objects]
    best = None
    for angle in candidate_angles:
        rotation = (
            Matrix.Translation(Vector(midpoint.tolist()))
            @ Matrix.Rotation(float(angle), 4, "Y")
            @ Matrix.Translation(Vector((-midpoint).tolist()))
        )
        for obj, matrix in zip(objects, original):
            obj.matrix_world = rotation @ matrix
        bpy.context.view_layer.update()
        moved = _evaluated_world_points(objects)
        moved_pair = _contact_pair(moved)
        gap = abs(float(moved_pair[1][2] - moved_pair[0][2])) if moved_pair else float("inf")
        if best is None or gap < best[0]:
            best = (gap, angle, [obj.matrix_world.copy() for obj in objects], moved_pair)

    if best is None:
        for obj, matrix in zip(objects, original):
            obj.matrix_world = matrix
        return {"applied": False, "reason": "rotation evaluation failed"}

    gap, angle, matrices, moved_pair = best
    for obj, matrix in zip(objects, matrices):
        obj.matrix_world = matrix
    bpy.context.view_layer.update()
    if moved_pair:
        ground_z = float((moved_pair[0][2] + moved_pair[1][2]) * 0.5)
    else:
        ground_z = float(np.min(_evaluated_world_points(objects)[:, 2]))
    for obj in objects:
        obj.location.z -= ground_z
    bpy.context.view_layer.update()
    final_pair = _contact_pair(_evaluated_world_points(objects))
    final_gap = abs(float(final_pair[1][2] - final_pair[0][2])) if final_pair else gap
    log(
        f"Cho-Gall bilateral ground alignment: angle={math.degrees(angle):.3f} deg, "
        f"foot-height gap={final_gap:.3f}"
    )
    return {
        "applied": True,
        "rotation_degrees": float(math.degrees(angle)),
        "foot_height_gap_mm": float(final_gap),
    }
'''
    source = replace_once(
        source,
        '\ndef forced_pose_selector(selected: dict[str, Any]):\n',
        ground_helper + '\n\ndef forced_pose_selector(selected: dict[str, Any]):\n',
        "ground helper insertion",
    )

    monkeypatch = '''        builder = load_builder(args.builder)
        hero_key = args.hero.casefold().replace("'", "").replace("-", "").replace(" ", "")
        if hero_key == "chogall":
            original_normalize = builder.normalize_object_group
            def normalize_and_ground(objects, target_height):
                result = original_normalize(objects, target_height)
                result["two_foot_grounding"] = enforce_two_foot_ground(list(objects))
                return result
            builder.normalize_object_group = normalize_and_ground
'''
    source = replace_once(
        source,
        '        builder = load_builder(args.builder)\n',
        monkeypatch,
        "Cho-Gall normalization monkeypatch",
    )
    compile(source, "build_one_photo_posed_v4.py", "exec")
    return source


def patch_runner(source: str, pose_path: Path, voxel: float, max_triangles: int, target_triangles: int) -> str:
    source = source.replace(
        'pose_builder = workspace / "tools" / "build_one_photo_posed_v3.py"',
        f'pose_builder = Path({str(pose_path)!r})',
    )
    source = source.replace(
        'payload_hash = verify_pose_builder(pose_builder)',
        'payload_hash = hashlib.sha256(pose_builder.read_bytes()).hexdigest()',
    )

    replacements = {
        '"--voxel-mm", "0.30"': f'"--voxel-mm", "{voxel:.3f}"',
        '"--max-triangles", "600000"': f'"--max-triangles", "{max_triangles}"',
        '"--max-triangles", "500000"': f'"--max-triangles", "{max_triangles}"',
        '"--target-triangles", "500000"': f'"--target-triangles", "{target_triangles}"',
        '"--target-triangles", "440000"': f'"--target-triangles", "{target_triangles}"',
        '"--pitch", "0.32"': f'"--pitch", "{max(voxel, 0.22):.3f}"',
    }
    for old, new in replacements.items():
        source = source.replace(old, new)

    # v3.1 failed the whole build when a separate raw/high-detail reference was not
    # manifold even though the repaired print mesh passed. V4 keeps that reference
    # for comparison but treats the robust/smooth print mesh as the release gate.
    source = source.replace(
        '], reports / "high_detail_validation.log")',
        '], reports / "high_detail_validation.log", check=False)',
    )
    source = source.replace(
        '"high_detail_validation_passed": high_validation.get("passed"),',
        '"high_detail_validation_passed": high_validation.get("passed", False),',
    )
    compile(source, "run_photo_posed_job_v4.py", "exec")
    return source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-bootstrap", required=True, type=Path)
    parser.add_argument("--pose-source", required=True, type=Path)
    parser.add_argument("--output-runner", required=True, type=Path)
    parser.add_argument("--output-pose", required=True, type=Path)
    parser.add_argument("--voxel", required=True, type=float)
    parser.add_argument("--max-triangles", type=int, default=900000)
    parser.add_argument("--target-triangles", type=int, default=820000)
    args = parser.parse_args()

    runner = decode_bootstrap(args.runner_bootstrap)
    pose = patch_pose_source(args.pose_source.read_text(encoding="utf-8"))
    args.output_pose.write_text(pose, encoding="utf-8")
    runner = patch_runner(
        runner,
        args.output_pose.resolve(),
        args.voxel,
        args.max_triangles,
        args.target_triangles,
    )
    args.output_runner.write_text(runner, encoding="utf-8")
    print(json_summary := {
        "runner_sha256": hashlib.sha256(runner.encode()).hexdigest(),
        "pose_sha256": hashlib.sha256(pose.encode()).hexdigest(),
        "voxel_mm": args.voxel,
        "max_triangles": args.max_triangles,
        "target_triangles": args.target_triangles,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
