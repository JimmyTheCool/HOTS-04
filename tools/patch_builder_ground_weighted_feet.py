#!/usr/bin/env python3
"""Patch the HOTS Blender builder so weighted foot soles meet the display base.

The normal builder aligns the base to the lowest geometry in the complete model.
For characters with hanging armour, cloth, weapons or effects below the boots this
can leave both feet visibly floating. This patch records the evaluated vertices
weighted to foot/toe/ankle bones before the armature is baked, then translates the
baked character so the higher sole intersects the base by a small controlled amount.
"""
from __future__ import annotations

from pathlib import Path
import sys


HELPERS = r'''

def _ground_side_from_group(name: str) -> str:
    value = name.casefold().replace("-", "_").replace(".", "_")
    tokens = [token for token in value.split("_") if token]
    left_tokens = {"l", "lf", "left", "lft"}
    right_tokens = {"r", "rt", "right", "rgt"}
    if any(token in left_tokens for token in tokens) or value.endswith(("left", "_l", " l")):
        return "left"
    if any(token in right_tokens for token in tokens) or value.endswith(("right", "_r", " r")):
        return "right"
    return "unknown"


def estimate_weighted_foot_soles() -> dict:
    """Measure posed sole heights from semantic foot-related vertex groups."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    foot_tokens = ("foot", "toe", "ankle", "heel", "hoof")
    ignored_tokens = ("footstep", "footprint", "sound", "fx", "attach")
    points = {"left": [], "right": [], "unknown": []}
    group_names = []
    examined_vertices = 0

    for original in list(bpy.context.scene.objects):
        if original.type != "MESH" or original.hide_render or original.hide_get():
            continue
        matching = {}
        for group in original.vertex_groups:
            lower = group.name.casefold()
            if any(token in lower for token in foot_tokens) and not any(token in lower for token in ignored_tokens):
                matching[group.index] = (group.name, _ground_side_from_group(group.name))
                group_names.append(group.name)
        if not matching:
            continue

        evaluated = original.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
        try:
            if mesh is None or len(mesh.vertices) != len(original.data.vertices):
                continue
            matrix = evaluated.matrix_world
            for index, vertex in enumerate(original.data.vertices):
                strongest_weight = 0.0
                strongest_side = "unknown"
                for membership in vertex.groups:
                    info = matching.get(membership.group)
                    if info is None:
                        continue
                    weight = float(membership.weight)
                    if weight > strongest_weight:
                        strongest_weight = weight
                        strongest_side = info[1]
                if strongest_weight < 0.12:
                    continue
                examined_vertices += 1
                point = matrix @ mesh.vertices[index].co
                points[strongest_side].append((float(point.x), float(point.y), float(point.z), strongest_weight))
        finally:
            evaluated.to_mesh_clear()

    def sole(values):
        if not values:
            return None
        ordered = sorted(float(item[2]) for item in values)
        # A low percentile is more stable than the absolute minimum and follows
        # the actual sole rather than one isolated spike or malformed vertex.
        position = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * 0.025))))
        return ordered[position]

    left = sole(points["left"])
    right = sole(points["right"])
    unknown = sole(points["unknown"])
    combined_values = points["left"] + points["right"] + points["unknown"]
    combined = sole(combined_values)
    return {
        "method": "evaluated vertex groups",
        "left_raw_z": left,
        "right_raw_z": right,
        "unknown_raw_z": unknown,
        "combined_raw_z": combined,
        "left_vertices": len(points["left"]),
        "right_vertices": len(points["right"]),
        "unknown_vertices": len(points["unknown"]),
        "examined_vertices": examined_vertices,
        "matching_groups": sorted(set(group_names)),
        "both_feet_detected": left is not None and right is not None,
    }


def estimate_geometric_foot_plane(objects: Sequence[bpy.types.Object]) -> dict:
    """Conservative fallback when the imported rig has no semantic group names."""
    vertices = []
    minimum, maximum = group_bounds(objects)
    height = max(1e-6, maximum.z - minimum.z)
    cutoff = minimum.z + min(height * 0.24, max(18.0, height * 0.12))
    for obj in objects:
        matrix = obj.matrix_world
        for vertex in obj.data.vertices:
            point = matrix @ vertex.co
            if point.z <= cutoff:
                vertices.append((float(point.x), float(point.y), float(point.z)))
    if not vertices:
        return {"method": "geometric fallback", "combined_raw_z": float(minimum.z), "both_feet_detected": False}
    centre_x = (minimum.x + maximum.x) / 2.0
    left = sorted(point[2] for point in vertices if point[0] <= centre_x)
    right = sorted(point[2] for point in vertices if point[0] > centre_x)
    all_z = sorted(point[2] for point in vertices)
    def percentile(values, fraction=0.08):
        if not values:
            return None
        return float(values[min(len(values) - 1, max(0, int(round((len(values) - 1) * fraction))))])
    return {
        "method": "lower-body geometric fallback",
        "left_raw_z": percentile(left),
        "right_raw_z": percentile(right),
        "combined_raw_z": percentile(all_z),
        "left_vertices": len(left),
        "right_vertices": len(right),
        "both_feet_detected": bool(left and right),
    }


def apply_weighted_foot_grounding(objects: Sequence[bpy.types.Object], normalization: dict, estimate: dict) -> dict:
    """Translate the baked model so both soles intersect the base surface."""
    if not estimate or estimate.get("combined_raw_z") is None:
        estimate = estimate_geometric_foot_plane(objects)

    factor = float(normalization["scale_factor"])
    before_min_z = float(normalization["bounds_before"][0][2])

    def normalized(value):
        return None if value is None else (float(value) - before_min_z) * factor

    left = normalized(estimate.get("left_raw_z"))
    right = normalized(estimate.get("right_raw_z"))
    combined = normalized(estimate.get("combined_raw_z"))

    soles = [value for value in (left, right) if value is not None]
    if len(soles) >= 2:
        higher_sole = max(soles)
        lower_sole = min(soles)
        sole_gap = higher_sole - lower_sole
    elif combined is not None:
        higher_sole = lower_sole = combined
        sole_gap = 0.0
    else:
        return {**estimate, "applied": False, "reason": "no usable sole measurement"}

    # The standard builder later lowers the character by 0.55 mm while the base
    # top is at +0.70 mm. Leaving the higher sole at +0.95 mm before that step
    # gives a final +0.40 mm sole height: 0.30 mm embedded into the base top.
    desired_pre_base_sole_z = 0.95
    shift_down = max(0.0, higher_sole - desired_pre_base_sole_z)
    for obj in objects:
        obj.location.z -= shift_down
    bpy.context.view_layer.update()

    base_top_after_final_shift = 0.70
    later_character_shift = 0.55
    final_left = None if left is None else left - shift_down - later_character_shift
    final_right = None if right is None else right - shift_down - later_character_shift
    final_combined = None if combined is None else combined - shift_down - later_character_shift
    contact = {
        "left_penetration_mm": None if final_left is None else base_top_after_final_shift - final_left,
        "right_penetration_mm": None if final_right is None else base_top_after_final_shift - final_right,
        "combined_penetration_mm": None if final_combined is None else base_top_after_final_shift - final_combined,
    }
    contacts = [value for value in (contact["left_penetration_mm"], contact["right_penetration_mm"]) if value is not None]
    both_contact = len(contacts) == 2 and min(contacts) >= -0.05

    return {
        **estimate,
        "applied": True,
        "normalization_scale": factor,
        "normalized_left_sole_z": left,
        "normalized_right_sole_z": right,
        "normalized_combined_sole_z": combined,
        "sole_height_difference_mm": sole_gap,
        "desired_pre_base_sole_z": desired_pre_base_sole_z,
        "translation_down_mm": shift_down,
        "predicted_final_left_sole_z": final_left,
        "predicted_final_right_sole_z": final_right,
        "predicted_final_combined_sole_z": final_combined,
        **contact,
        "both_feet_predicted_in_contact": both_contact,
    }
'''


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: patch_builder_ground_weighted_feet.py BUILDER.py")
    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")

    marker = "\ndef group_bounds(objects: Sequence[bpy.types.Object]) -> Tuple[Vector, Vector]:\n"
    if text.count(marker) != 1:
        raise RuntimeError(f"Expected one group_bounds marker; found {text.count(marker)}")
    text = text.replace(marker, HELPERS + marker, 1)

    old = """    import_info = import_source_group(paths, source_key, args.skip_animation)\n    mesh_objects = evaluated_mesh_objects()\n"""
    new = """    import_info = import_source_group(paths, source_key, args.skip_animation)\n    weighted_foot_estimate = estimate_weighted_foot_soles()\n    mesh_objects = evaluated_mesh_objects()\n"""
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one pre-bake insertion point; found {text.count(old)}")
    text = text.replace(old, new, 1)

    old = """    normalization = normalize_object_group(mesh_objects, args.target_height_mm)\n\n    detail_parts ="""
    new = """    normalization = normalize_object_group(mesh_objects, args.target_height_mm)\n    grounding = apply_weighted_foot_grounding(mesh_objects, normalization, weighted_foot_estimate)\n\n    detail_parts ="""
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one post-normalization insertion point; found {text.count(old)}")
    text = text.replace(old, new, 1)

    old = '        "normalization": normalization,\n        "components": component_reports,'
    new = '        "normalization": normalization,\n        "grounding": grounding,\n        "components": component_reports,'
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one result insertion point; found {text.count(old)}")
    text = text.replace(old, new, 1)

    path.write_text(text, encoding="utf-8")
    compile(text, str(path), "exec")
    print(f"Patched rig-aware foot grounding into {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
