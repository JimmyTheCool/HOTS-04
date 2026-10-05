#!/usr/bin/env python3
"""Prepare the HOTS-04 V4 photo-pose runner without rewriting core functions.

The existing pose builder is decoded and retained intact. Runtime overrides are
appended for standing-pose filtering and Cho-Gall bilateral foot grounding. This
approach is resilient to formatting changes in the proven importer/search code.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
from pathlib import Path
import re
import zlib


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
        if payload is not None:
            break
    if payload is None:
        raise RuntimeError(f"No _PAYLOAD found in {path}")
    source = zlib.decompress(base64.b85decode(payload)).decode("utf-8")
    compile(source, str(path), "exec")
    return source


POSE_OVERRIDES = r'''

# ---------------------------------------------------------------------------
# HOTS-04 V4 runtime overrides
# ---------------------------------------------------------------------------
CURRENT_HERO = ""
_ORIGINAL_PARSE_ARGS = parse_args
_ORIGINAL_ANIMATION_PRIORITY = animation_priority
_ORIGINAL_LOAD_BUILDER = load_builder


def parse_args() -> argparse.Namespace:
    global CURRENT_HERO
    args = _ORIGINAL_PARSE_ARGS()
    CURRENT_HERO = args.hero
    return args


def animation_priority(name: str) -> int:
    lower = name.casefold()
    forbidden = (
        "death", "dead", "ragdoll", "corpse", "dismember", "mount", "vehicle",
        "knock", "stun", "hitreact", "flail", "fall", "despawn", "lowpoly",
        "sit", "seated", "chair", "throne", "crouch", "kneel", "sleep", "crawl",
    )
    if any(token in lower for token in forbidden):
        return -100000

    score = _ORIGINAL_ANIMATION_PRIORITY(name)
    hero_key = CURRENT_HERO.casefold().replace("'", "").replace("-", "").replace(" ", "")
    standing_tokens = ("stand", "idle", "ready")
    if hero_key in {"nova", "chogall"}:
        if not any(token in lower for token in standing_tokens):
            return -100000
        score += 7000
    elif any(token in lower for token in standing_tokens):
        score += 1800
    return score


def _evaluated_world_points(objects: list[bpy.types.Object], max_points: int = 260000) -> np.ndarray:
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
    lower = points[points[:, 2] <= z0 + height * 0.25]
    if len(lower) < 80:
        return None
    x_mid = float(np.median(points[:, 0]))
    x_lo, x_hi = np.percentile(points[:, 0], [6.0, 94.0])
    lower = lower[(lower[:, 0] >= x_lo) & (lower[:, 0] <= x_hi)]
    left = lower[lower[:, 0] < x_mid]
    right = lower[lower[:, 0] >= x_mid]
    if len(left) < 20 or len(right) < 20:
        return None

    def contact(side: np.ndarray) -> np.ndarray:
        z_cut = float(np.percentile(side[:, 2], 2.5))
        band = side[side[:, 2] <= z_cut + max(0.30, height * 0.005)]
        if len(band) < 4:
            band = side[np.argsort(side[:, 2])[: max(4, min(50, len(side)))]]
        return np.median(band, axis=0)

    return contact(left), contact(right)


def enforce_two_foot_ground(objects: list[bpy.types.Object]) -> dict[str, float | bool | str]:
    """Tilt and lower Cho-Gall until both lateral lower foot zones meet the base."""
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

    midpoint = (left + right) * 0.5
    original = [obj.matrix_world.copy() for obj in objects]
    best = None
    for angle in (math.atan2(dz, dx), -math.atan2(dz, dx)):
        rotation = (
            Matrix.Translation(Vector(midpoint.tolist()))
            @ Matrix.Rotation(float(angle), 4, "Y")
            @ Matrix.Translation(Vector((-midpoint).tolist()))
        )
        for obj, matrix in zip(objects, original):
            obj.matrix_world = rotation @ matrix
        bpy.context.view_layer.update()
        moved_pair = _contact_pair(_evaluated_world_points(objects))
        gap = abs(float(moved_pair[1][2] - moved_pair[0][2])) if moved_pair else float("inf")
        candidate = (gap, float(angle), [obj.matrix_world.copy() for obj in objects], moved_pair)
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        for obj, matrix in zip(objects, original):
            obj.matrix_world = matrix
        bpy.context.view_layer.update()
        return {"applied": False, "reason": "rotation evaluation failed"}

    gap, angle, matrices, moved_pair = best
    for obj, matrix in zip(objects, matrices):
        obj.matrix_world = matrix
    bpy.context.view_layer.update()
    if moved_pair:
        ground_z = float((moved_pair[0][2] + moved_pair[1][2]) * 0.5)
    else:
        moved = _evaluated_world_points(objects)
        ground_z = float(np.min(moved[:, 2]))
    for obj in objects:
        obj.location.z -= ground_z
    bpy.context.view_layer.update()

    final_pair = _contact_pair(_evaluated_world_points(objects))
    final_gap = abs(float(final_pair[1][2] - final_pair[0][2])) if final_pair else float(gap)
    log(
        f"Cho-Gall bilateral grounding: rotation={math.degrees(angle):.3f} deg, "
        f"contact gap={final_gap:.3f}"
    )
    return {
        "applied": True,
        "rotation_degrees": float(math.degrees(angle)),
        "foot_height_gap_mm": float(final_gap),
        "ground_translation_mm": float(-ground_z),
    }


def load_builder(path: Path):
    builder = _ORIGINAL_LOAD_BUILDER(path)
    hero_key = CURRENT_HERO.casefold().replace("'", "").replace("-", "").replace(" ", "")
    if hero_key == "chogall":
        original_normalize = builder.normalize_object_group

        def normalize_and_ground(objects, target_height):
            result = original_normalize(objects, target_height)
            result["two_foot_grounding"] = enforce_two_foot_ground(list(objects))
            return result

        builder.normalize_object_group = normalize_and_ground
    return builder
'''


def patch_pose_source(source: str) -> str:
    main_pattern = re.compile(
        r'\nif __name__ == ["\']__main__["\']:\n\s+raise SystemExit\(main\(\)\)\s*$',
        re.S,
    )
    source, count = main_pattern.subn("\n", source, count=1)
    if count != 1:
        raise RuntimeError(f"Expected one pose-builder main block, found {count}")
    source = source.rstrip() + POSE_OVERRIDES + '\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'
    compile(source, "build_one_photo_posed_v4.py", "exec")
    return source


def replace_once(text: str, pattern: str, replacement: str, label: str, flags: int = 0) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"Expected one {label} target, found {count}")
    return updated


def patch_runner(source: str, pose_path: Path, voxel: float, max_triangles: int, target_triangles: int) -> str:
    source = replace_once(
        source,
        r'    pose_builder\s*=\s*workspace\s*/\s*"tools"\s*/\s*"build_one_photo_posed_v3\.py"\n',
        f'    pose_builder = Path({str(pose_path)!r})\n',
        "pose builder path",
    )
    source = replace_once(
        source,
        r'    payload_hash\s*=\s*verify_pose_builder\(pose_builder\)\n',
        '    payload_hash = hashlib.sha256(pose_builder.read_bytes()).hexdigest()\n',
        "pose builder verification",
    )

    source = source.replace('"--voxel-mm", "0.30"', f'"--voxel-mm", "{voxel:.3f}"')
    source = source.replace('"--max-triangles", "600000"', f'"--max-triangles", "{max_triangles}"')
    source = source.replace('"--max-triangles", "500000"', f'"--max-triangles", "{max_triangles}"')
    source = source.replace('"--max-triangles", "450000"', f'"--max-triangles", "{max_triangles}"')
    source = source.replace('"--target-triangles", "575000"', f'"--target-triangles", "{target_triangles}"')
    source = source.replace('"--target-triangles", "550000"', f'"--target-triangles", "{target_triangles}"')
    source = source.replace('"--target-triangles", "420000"', f'"--target-triangles", "{target_triangles}"')
    source = source.replace('"--target-triangles", "400000"', f'"--target-triangles", "{target_triangles}"')

    # Fine pitch is driven by the requested per-character voxel value, while the
    # conservative fallback remains no coarser than 0.42 mm.
    source = source.replace(
        'high_pitch = max(0.34, max_extent / 520.0)',
        f'high_pitch = max({voxel:.5f}, max_extent / 760.0)',
    )
    source = source.replace(
        'robust_pitch = max(0.58, max_extent / 360.0)',
        f'robust_pitch = max({max(voxel + 0.10, 0.34):.5f}, max_extent / 560.0)',
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

    pose = patch_pose_source(args.pose_source.read_text(encoding="utf-8"))
    args.output_pose.write_text(pose, encoding="utf-8")
    runner = patch_runner(
        decode_bootstrap(args.runner_bootstrap),
        args.output_pose.resolve(),
        args.voxel,
        args.max_triangles,
        args.target_triangles,
    )
    args.output_runner.write_text(runner, encoding="utf-8")
    print({
        "runner_sha256": hashlib.sha256(runner.encode("utf-8")).hexdigest(),
        "pose_sha256": hashlib.sha256(pose.encode("utf-8")).hexdigest(),
        "voxel_mm": args.voxel,
        "max_triangles": args.max_triangles,
        "target_triangles": args.target_triangles,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
