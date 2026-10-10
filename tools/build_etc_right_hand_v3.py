#!/usr/bin/env python3
"""Rebuild E.T.C.'s raised right hand with three clear, printable fingers.

The photographed-pose ETC mesh already has the correct body, guitar and left hand.
This script removes only the old raised hand, constructs a fresh broad Tauren hand,
and then delegates to the existing high-quality printable-mesh pipeline.

Right-hand anatomy:
* one full-size finger points upward;
* two full-size curled fingers are deliberately spread apart;
* all three digits have tapered phalanges and broad knuckles;
* the palm and wrist overlap the original forearm for a strong printable union.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Iterable

import bmesh
import bpy
from mathutils import Matrix, Vector

BASE_PATH = Path(__file__).resolve().with_name("build_etc_super_smooth_blender.py")
spec = importlib.util.spec_from_file_location("etc_right_hand_v3_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base builder: {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

_ORIGINAL_IMPORT_SOURCE = base.import_source
_ORIGINAL_RENDER_VIEWS = base.render_views


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def average(points: Iterable[Vector]) -> Vector:
    points = list(points)
    if not points:
        return Vector((0.0, 0.0, 0.0))
    total = Vector((0.0, 0.0, 0.0))
    for point in points:
        total += point
    return total / len(points)


def make_frame(side: Vector, front: Vector, up: Vector) -> Matrix:
    """Return a 4x4 rotation matrix whose local X/Y/Z axes match side/front/up."""
    frame = Matrix.Identity(4)
    frame[0][0], frame[1][0], frame[2][0] = side.x, side.y, side.z
    frame[0][1], frame[1][1], frame[2][1] = front.x, front.y, front.z
    frame[0][2], frame[1][2], frame[2][2] = up.x, up.y, up.z
    return frame


def apply_object_transform(obj: bpy.types.Object) -> None:
    base.select_only(obj)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)


def add_ellipsoid(name: str, center: Vector, half_size: tuple[float, float, float],
                  frame: Matrix, segments: int = 64, rings: int = 32) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments,
        ring_count=rings,
        radius=1.0,
        location=(0.0, 0.0, 0.0),
    )
    obj = bpy.context.active_object
    obj.name = name
    scale = Matrix.Diagonal((half_size[0], half_size[1], half_size[2], 1.0))
    obj.matrix_world = Matrix.Translation(center) @ frame @ scale
    apply_object_transform(obj)
    return obj


def add_sphere(name: str, center: Vector, radius: float, segments: int = 56,
               rings: int = 28) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments,
        ring_count=rings,
        radius=radius,
        location=center,
    )
    obj = bpy.context.active_object
    obj.name = name
    apply_object_transform(obj)
    return obj


def add_tapered_segment(name: str, start: Vector, end: Vector,
                        radius_start: float, radius_end: float,
                        vertices: int = 56) -> bpy.types.Object:
    direction = end - start
    length = direction.length
    if length < 0.05:
        return add_sphere(name, start, max(radius_start, radius_end), vertices, vertices // 2)
    midpoint = (start + end) * 0.5
    bpy.ops.mesh.primitive_cone_add(
        vertices=vertices,
        radius1=radius_start,
        radius2=radius_end,
        depth=length,
        end_fill_type="NGON",
        location=midpoint,
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = direction.to_track_quat("Z", "Y")
    apply_object_transform(obj)
    return obj


def add_finger(name: str, points: list[Vector], radii: list[float]) -> list[bpy.types.Object]:
    if len(points) != len(radii) or len(points) < 2:
        raise ValueError("Finger points and radii must have equal length >= 2")
    objects: list[bpy.types.Object] = []
    for index, (point, radius) in enumerate(zip(points, radii)):
        # A slightly broader sphere at each joint gives distinct, organic phalanges.
        joint_radius = radius * (1.055 if 0 < index < len(points) - 1 else 1.0)
        objects.append(add_sphere(f"{name}_joint_{index:02d}", point, joint_radius))
    for index in range(len(points) - 1):
        objects.append(add_tapered_segment(
            f"{name}_segment_{index:02d}",
            points[index], points[index + 1], radii[index], radii[index + 1],
        ))
    return objects


def delete_old_raised_hand(obj: bpy.types.Object) -> dict[str, object]:
    """Delete the old hand using the raised fingertip as an automatic landmark."""
    minimum, maximum = base.object_bounds(obj)
    center = (minimum + maximum) * 0.5
    width = maximum.x - minimum.x
    height = maximum.z - minimum.z

    points = [vertex.co.copy() for vertex in obj.data.vertices]
    upper_z = minimum.z + height * 0.70

    # In the photographed pose the raised right arm is on the model's left side.
    candidates = [
        point for point in points
        if point.z >= upper_z and point.x <= center.x - width * 0.08
    ]
    if not candidates:
        candidates = [point for point in points if point.z >= minimum.z + height * 0.80]
    if not candidates:
        raise RuntimeError("Could not locate the raised right hand")

    highest_z = max(point.z for point in candidates)
    tip_points = [point for point in candidates if point.z >= highest_z - max(0.8, height * 0.009)]
    tip = average(tip_points)

    cut_depth = clamp(height * 0.225, 21.5, 25.5)
    cut_z = tip.z - cut_depth
    delete_radius = clamp(height * 0.165, 15.5, 18.5)

    wrist_band = [
        point for point in points
        if cut_z - 3.2 <= point.z <= cut_z + 1.4
        and math.hypot(point.x - tip.x, point.y - tip.y) <= delete_radius
    ]
    if wrist_band:
        wrist = average(wrist_band)
    else:
        wrist = Vector((tip.x, tip.y, cut_z - 0.8))

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    doomed = [
        vertex for vertex in bm.verts
        if vertex.co.z >= cut_z
        and math.hypot(vertex.co.x - tip.x, vertex.co.y - tip.y) <= delete_radius
    ]
    if len(doomed) < 20:
        bm.free()
        raise RuntimeError(f"Raised-hand deletion selected too little geometry: {len(doomed)} vertices")
    bmesh.ops.delete(bm, geom=doomed, context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

    base.log(
        "Removed old raised hand: "
        f"{len(doomed):,} vertices; tip={tuple(round(v, 3) for v in tip)}; "
        f"wrist={tuple(round(v, 3) for v in wrist)}"
    )
    return {
        "minimum": minimum,
        "maximum": maximum,
        "center": center,
        "height": height,
        "width": width,
        "old_tip": tip,
        "wrist": wrist,
        "cut_z": cut_z,
        "delete_radius": delete_radius,
    }


def construct_fresh_three_finger_hand(obj: bpy.types.Object,
                                      landmarks: dict[str, object]) -> dict[str, object]:
    center: Vector = landmarks["center"]  # type: ignore[assignment]
    old_tip: Vector = landmarks["old_tip"]  # type: ignore[assignment]
    wrist: Vector = landmarks["wrist"]  # type: ignore[assignment]
    height = float(landmarks["height"])

    up = old_tip - wrist
    if up.length < 8.0:
        up = Vector((0.0, 0.0, 1.0))
    else:
        up.normalize()
    if up.z < 0.72:
        up = (up * 0.45 + Vector((0.0, 0.0, 1.0)) * 0.55).normalized()

    # Front in the established ETC model points toward negative Y.
    front = Vector((0.0, -1.0, 0.0))
    front -= up * front.dot(up)
    if front.length < 0.1:
        front = Vector((0.0, -1.0, 0.0))
    front.normalize()
    side = up.cross(front).normalized()

    # The pointing finger sits on the outside edge of the raised hand.
    outer_sign = -1.0 if wrist.x < center.x else 1.0
    outer = side * outer_sign
    inner = -outer
    frame = make_frame(outer, front, up)

    scale = clamp(height / 107.0, 0.88, 1.16)
    palm_half_width = 9.6 * scale
    palm_half_depth = 6.0 * scale
    palm_half_height = 7.1 * scale
    finger_radius = 3.65 * scale

    components: list[bpy.types.Object] = []

    # A strong wrist bridge overlaps the cut forearm so the final voxel union is robust.
    bridge_start = wrist - up * (2.8 * scale)
    bridge_end = wrist + up * (5.8 * scale)
    components.append(add_tapered_segment(
        "ETC_RH_wrist_bridge", bridge_start, bridge_end,
        5.3 * scale, 6.7 * scale, vertices=64,
    ))
    components.append(add_sphere(
        "ETC_RH_wrist_knuckle", wrist + up * (3.3 * scale), 6.25 * scale,
        segments=64, rings=32,
    ))

    palm_center = wrist + up * (7.0 * scale) + front * (0.25 * scale)
    components.append(add_ellipsoid(
        "ETC_RH_broad_palm", palm_center,
        (palm_half_width, palm_half_depth, palm_half_height), frame,
        segments=72, rings=36,
    ))

    # 1) Upward-pointing finger: preserve the successful large scale and silhouette.
    hand_length = clamp((old_tip - wrist).length, 22.0 * scale, 26.0 * scale)
    pointing_base = wrist + up * (8.5 * scale) + outer * (5.4 * scale) - front * (0.2 * scale)
    pointing_points = [
        pointing_base,
        wrist + up * (13.5 * scale) + outer * (5.65 * scale) - front * (0.35 * scale),
        wrist + up * (19.0 * scale) + outer * (5.75 * scale) - front * (0.20 * scale),
        wrist + up * hand_length + outer * (5.25 * scale),
    ]
    pointing_radii = [
        4.05 * scale,
        3.90 * scale,
        3.55 * scale,
        2.85 * scale,
    ]
    components.extend(add_finger("ETC_RH_pointing_finger", pointing_points, pointing_radii))

    # 2) First curled finger: full-size, broad, and clearly offset from the third digit.
    curled_a_lateral = outer * (0.25 * scale)
    curled_a = [
        wrist + up * (8.0 * scale) + curled_a_lateral - front * (0.35 * scale),
        wrist + up * (13.2 * scale) + curled_a_lateral - front * (1.25 * scale),
        wrist + up * (15.0 * scale) + outer * (0.55 * scale) + front * (4.7 * scale),
        wrist + up * (11.2 * scale) + outer * (0.75 * scale) + front * (7.1 * scale),
        wrist + up * (7.6 * scale) + outer * (0.85 * scale) + front * (4.8 * scale),
    ]
    curled_a_radii = [
        3.90 * scale,
        3.78 * scale,
        3.58 * scale,
        3.30 * scale,
        2.92 * scale,
    ]
    components.extend(add_finger("ETC_RH_curled_finger_A", curled_a, curled_a_radii))

    # 3) Second curled finger: equally substantial and separated by roughly two radii.
    # Its slightly different curl prevents the two digits reading as one fused lump.
    curled_b_lateral = inner * (7.25 * scale)
    curled_b = [
        wrist + up * (7.7 * scale) + curled_b_lateral + front * (0.15 * scale),
        wrist + up * (12.7 * scale) + inner * (7.45 * scale) - front * (0.35 * scale),
        wrist + up * (14.3 * scale) + inner * (7.75 * scale) + front * (4.25 * scale),
        wrist + up * (10.5 * scale) + inner * (7.95 * scale) + front * (6.55 * scale),
        wrist + up * (7.1 * scale) + inner * (7.55 * scale) + front * (4.25 * scale),
    ]
    curled_b_radii = [
        3.92 * scale,
        3.80 * scale,
        3.60 * scale,
        3.32 * scale,
        2.94 * scale,
    ]
    components.extend(add_finger("ETC_RH_curled_finger_B", curled_b, curled_b_radii))

    # Add broad metacarpal pads below the two curled digits. These support the curl
    # without bridging the visible gap between their shafts and tips.
    components.append(add_ellipsoid(
        "ETC_RH_metacarpal_A",
        wrist + up * (9.0 * scale) + outer * (0.15 * scale),
        (3.9 * scale, 4.6 * scale, 4.7 * scale), frame,
        segments=56, rings=28,
    ))
    components.append(add_ellipsoid(
        "ETC_RH_metacarpal_B",
        wrist + up * (8.7 * scale) + inner * (7.0 * scale),
        (3.95 * scale, 4.55 * scale, 4.65 * scale), frame,
        segments=56, rings=28,
    ))

    # Join source and all new hand pieces. The downstream fine voxel remesh creates
    # one watertight connected shell while retaining the deliberate finger gaps.
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    for component in components:
        component.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.join()
    obj.name = "ETC_WITH_FRESH_RIGHT_HAND"
    base.apply_transforms(obj)

    separation = (curled_a[3] - curled_b[3]).length
    base.log(
        "Constructed fresh three-finger right hand: "
        f"pointing length={hand_length:.2f} mm; "
        f"curled-finger bend separation={separation:.2f} mm; "
        f"finger radius={finger_radius:.2f} mm"
    )
    return {
        "wrist": wrist.copy(),
        "pointing_tip": pointing_points[-1].copy(),
        "palm_center": palm_center.copy(),
        "curled_a_tip": curled_a[-1].copy(),
        "curled_b_tip": curled_b[-1].copy(),
        "curled_bend_separation_mm": separation,
    }


def import_source_with_new_hand(path: Path) -> bpy.types.Object:
    obj = _ORIGINAL_IMPORT_SOURCE(path)
    landmarks = delete_old_raised_hand(obj)
    construct_fresh_three_finger_hand(obj, landmarks)
    return obj


def high_quality_fast_smooth(obj: bpy.types.Object, iterations: int, strength: float) -> None:
    """Preserve more detail than R5 while remaining practical on a CI runner."""
    before = len(obj.data.polygons)
    cap = 1_550_000
    if before > cap:
        modifier = obj.modifiers.new("ETC_RH_DETAIL_CAP", "DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        modifier.ratio = cap / float(before)
        modifier.use_collapse_triangulate = True
        base.apply_modifier(obj, modifier.name)
        base.log(f"Dense detail cap: {before:,} -> {len(obj.data.polygons):,} polygons")

    lap = obj.modifiers.new("ETC_RH_VOLUME_SMOOTH", "LAPLACIANSMOOTH")
    lap.iterations = 2
    lap.lambda_factor = 0.060
    lap.lambda_border = 0.0
    lap.use_volume_preserve = True
    if hasattr(lap, "use_normalized"):
        lap.use_normalized = True
    base.apply_modifier(obj, lap.name)

    micro = obj.modifiers.new("ETC_RH_MICRO_SMOOTH", "SMOOTH")
    micro.factor = 0.065
    micro.iterations = 1
    micro.use_x = True
    micro.use_y = True
    micro.use_z = True
    base.apply_modifier(obj, micro.name)


def render_views_with_hand_closeups(obj: bpy.types.Object, output_dir: Path,
                                     label: str) -> list[str]:
    paths = _ORIGINAL_RENDER_VIEWS(obj, output_dir, label)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 1100
    scene.render.resolution_y = 1100
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "SINGLE"
    scene.display.shading.single_color = (0.39, 0.55, 0.72)
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "BOTH"
    scene.display.shading.curvature_ridge_factor = 2.0
    scene.display.shading.curvature_valley_factor = 1.4

    minimum, maximum = base.object_bounds(obj)
    center = (minimum + maximum) * 0.5
    width = maximum.x - minimum.x
    height = maximum.z - minimum.z
    points = [vertex.co.copy() for vertex in obj.data.vertices]
    candidates = [
        point for point in points
        if point.z >= minimum.z + height * 0.72
        and point.x <= center.x - width * 0.06
    ]
    if not candidates:
        candidates = [point for point in points if point.z >= maximum.z - height * 0.25]
    top_z = max(point.z for point in candidates)
    local = [point for point in candidates if point.z >= top_z - 25.0]
    target = average(local) if local else max(candidates, key=lambda point: point.z)

    camera_data = bpy.data.cameras.new(f"{label}_HAND_CAMERA")
    camera = bpy.data.objects.new(f"{label}_HAND_CAMERA", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = clamp(height * 0.42, 38.0, 48.0)

    views = [
        ("front", Vector((target.x, target.y - 95.0, target.z + 1.0))),
        ("three_quarter", Vector((target.x - 68.0, target.y - 68.0, target.z + 4.0))),
        ("side", Vector((target.x - 95.0, target.y, target.z + 1.0))),
    ]
    for suffix, location in views:
        camera.location = location
        base.point_camera(camera, target)
        path = output_dir / f"{label}_right_hand_{suffix}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        paths.append(str(path))
    bpy.data.objects.remove(camera, do_unlink=True)
    return paths


base.import_source = import_source_with_new_hand
base.volume_preserving_smooth = high_quality_fast_smooth
base.render_views = render_views_with_hand_closeups
raise SystemExit(base.main())
