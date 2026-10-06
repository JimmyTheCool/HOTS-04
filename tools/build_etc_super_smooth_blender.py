#!/usr/bin/env python3
"""Build high-resolution, smoothed and printable E.T.C. STL variants.

Run inside Blender 3.6 in background mode. The input is an already photo-posed
E.T.C. STL. The script performs a fine OpenVDB voxel union to remove open shells
and disconnected overlaps, applies volume-preserving geometric smoothing, keeps
the base underside flat, limits triangle count without changing the pose, exports
binary STL files, and renders four inspection views.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import bpy
from mathutils import Vector


def blender_args() -> list[str]:
    argv = sys.argv
    return argv[argv.index("--") + 1 :] if "--" in argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--high-pitch-mm", type=float, default=0.17)
    parser.add_argument("--print-pitch-mm", type=float, default=0.24)
    parser.add_argument("--high-target-triangles", type=int, default=1_350_000)
    parser.add_argument("--print-target-triangles", type=int, default=700_000)
    parser.add_argument("--base-height-mm", type=float, default=4.5)
    parser.add_argument("--base-margin-mm", type=float, default=5.0)
    return parser.parse_args(blender_args())


def log(message: str) -> None:
    print(f"[ETC-SMOOTH] {message}", flush=True)


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials,
                       bpy.data.cameras, bpy.data.lights):
        for datablock in list(datablocks):
            try:
                datablocks.remove(datablock)
            except Exception:
                pass


def enable_stl_addon() -> None:
    try:
        bpy.ops.preferences.addon_enable(module="io_mesh_stl")
    except Exception as exc:
        log(f"STL add-on enable warning: {exc}")


def select_only(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def apply_modifier(obj: bpy.types.Object, name: str) -> None:
    select_only(obj)
    bpy.ops.object.modifier_apply(modifier=name)


def apply_transforms(obj: bpy.types.Object) -> None:
    select_only(obj)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)


def import_source(path: Path) -> bpy.types.Object:
    reset_scene()
    enable_stl_addon()
    bpy.ops.import_mesh.stl(filepath=str(path), global_scale=1.0)
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh imported from {path}")
    if len(meshes) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in meshes:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active
    else:
        obj = meshes[0]
    obj.name = "ETC_POSED_SOURCE"
    apply_transforms(obj)
    return obj


def object_bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    minimum = Vector((
        min(corner.x for corner in corners),
        min(corner.y for corner in corners),
        min(corner.z for corner in corners),
    ))
    maximum = Vector((
        max(corner.x for corner in corners),
        max(corner.y for corner in corners),
        max(corner.z for corner in corners),
    ))
    return minimum, maximum


def has_broad_flat_base(obj: bpy.types.Object) -> bool:
    mesh = obj.data
    if not mesh.vertices:
        return False
    minimum, maximum = object_bounds(obj)
    width = max(1e-6, maximum.x - minimum.x)
    depth = max(1e-6, maximum.y - minimum.y)
    band = max(0.55, 0.006 * max(width, depth))
    points = [obj.matrix_world @ vertex.co for vertex in mesh.vertices]
    bottom = [point for point in points if point.z <= minimum.z + band]
    if len(bottom) < max(24, int(len(points) * 0.0015)):
        return False
    spread_x = max(point.x for point in bottom) - min(point.x for point in bottom)
    spread_y = max(point.y for point in bottom) - min(point.y for point in bottom)
    return spread_x >= width * 0.45 and spread_y >= depth * 0.45


def add_base_if_needed(obj: bpy.types.Object, height: float, margin: float) -> dict[str, Any]:
    minimum, maximum = object_bounds(obj)
    width = maximum.x - minimum.x
    depth = maximum.y - minimum.y
    if has_broad_flat_base(obj):
        log("Existing broad base detected; preserving it")
        return {"added": False, "height_mm": None, "radius_mm": None}

    center_x = (minimum.x + maximum.x) / 2.0
    center_y = (minimum.y + maximum.y) / 2.0
    radius = max(width, depth) / 2.0 + margin
    # Raise the character so its lowest geometry overlaps the base by 0.65 mm.
    overlap = 0.65
    obj.location.z += height - overlap - minimum.z
    bpy.context.view_layer.update()
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=192,
        radius=radius,
        depth=height,
        location=(center_x, center_y, height / 2.0),
    )
    base = bpy.context.active_object
    base.name = "ETC_PRINT_BASE"
    apply_transforms(base)

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    base.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.join()
    log(f"Added {height:.2f} mm circular base, radius {radius:.2f} mm")
    return {"added": True, "height_mm": height, "radius_mm": radius}


def mesh_stats(obj: bpy.types.Object) -> dict[str, Any]:
    minimum, maximum = object_bounds(obj)
    return {
        "vertices": len(obj.data.vertices),
        "edges": len(obj.data.edges),
        "polygons": len(obj.data.polygons),
        "dimensions_mm": [
            float(maximum.x - minimum.x),
            float(maximum.y - minimum.y),
            float(maximum.z - minimum.z),
        ],
        "minimum": [float(minimum.x), float(minimum.y), float(minimum.z)],
        "maximum": [float(maximum.x), float(maximum.y), float(maximum.z)],
    }


def voxel_union(obj: bpy.types.Object, pitch: float) -> None:
    select_only(obj)
    mesh = obj.data
    mesh.remesh_voxel_size = pitch
    mesh.remesh_voxel_adaptivity = 0.0
    if hasattr(mesh, "use_remesh_fix_poles"):
        mesh.use_remesh_fix_poles = True
    if hasattr(mesh, "use_remesh_preserve_volume"):
        mesh.use_remesh_preserve_volume = True
    if hasattr(mesh, "use_remesh_preserve_attributes"):
        mesh.use_remesh_preserve_attributes = False
    log(f"Running fine voxel union at {pitch:.4f} mm")
    bpy.ops.object.voxel_remesh()


def volume_preserving_smooth(obj: bpy.types.Object, iterations: int, strength: float) -> None:
    lap = obj.modifiers.new("ETC_VOLUME_SMOOTH", "LAPLACIANSMOOTH")
    lap.iterations = iterations
    lap.lambda_factor = strength
    lap.lambda_border = 0.0
    lap.use_volume_preserve = True
    if hasattr(lap, "use_normalized"):
        lap.use_normalized = True
    apply_modifier(obj, lap.name)

    corrective = obj.modifiers.new("ETC_CORRECTIVE_SMOOTH", "CORRECTIVE_SMOOTH")
    corrective.factor = min(0.22, strength * 1.35)
    corrective.iterations = 2
    if hasattr(corrective, "smooth_type"):
        corrective.smooth_type = "LENGTH_WEIGHTED"
    if hasattr(corrective, "use_only_smooth"):
        corrective.use_only_smooth = True
    apply_modifier(obj, corrective.name)


def flatten_base_and_ground(obj: bpy.types.Object, pitch: float) -> dict[str, float]:
    mesh = obj.data
    if not mesh.vertices:
        raise RuntimeError("Cannot ground empty mesh")
    minimum = min(vertex.co.z for vertex in mesh.vertices)
    flatten_band = max(0.36, pitch * 2.35)
    changed = 0
    for vertex in mesh.vertices:
        if vertex.co.z <= minimum + flatten_band:
            vertex.co.z = minimum
            changed += 1
    for vertex in mesh.vertices:
        vertex.co.z -= minimum
    mesh.update()
    return {
        "original_min_z": float(minimum),
        "flatten_band_mm": float(flatten_band),
        "flattened_vertices": int(changed),
    }


def decimate_if_needed(obj: bpy.types.Object, target_triangles: int) -> dict[str, Any]:
    before = len(obj.data.polygons)
    if before <= target_triangles:
        return {"before": before, "after": before, "applied": False, "ratio": 1.0}
    ratio = max(0.05, min(1.0, target_triangles / float(before)))
    modifier = obj.modifiers.new("ETC_DETAIL_DECIMATE", "DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = ratio
    modifier.use_collapse_triangulate = True
    if hasattr(modifier, "use_symmetry"):
        modifier.use_symmetry = False
    apply_modifier(obj, modifier.name)
    return {
        "before": before,
        "after": len(obj.data.polygons),
        "applied": True,
        "ratio": ratio,
    }


def finalize_mesh(obj: bpy.types.Object) -> None:
    select_only(obj)
    triangulate = obj.modifiers.new("ETC_TRIANGULATE", "TRIANGULATE")
    triangulate.keep_custom_normals = True
    apply_modifier(obj, triangulate.name)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    try:
        bpy.ops.mesh.remove_doubles(threshold=0.0005)
    except Exception:
        pass
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    obj.data.validate(clean_customdata=True)
    obj.data.update()


def export_stl(obj: bpy.types.Object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    select_only(obj)
    bpy.ops.export_mesh.stl(
        filepath=str(output),
        use_selection=True,
        ascii=False,
        use_mesh_modifiers=True,
        global_scale=1.0,
    )
    if not output.is_file() or output.stat().st_size < 1024:
        raise RuntimeError(f"STL export failed: {output}")


def point_camera(camera: bpy.types.Object, target: Vector) -> None:
    direction = target - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def render_views(obj: bpy.types.Object, output_dir: Path, label: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 1000
    scene.render.resolution_y = 1000
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "SINGLE"
    scene.display.shading.single_color = (0.36, 0.52, 0.68)
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "BOTH"
    scene.display.shading.curvature_ridge_factor = 1.8
    scene.display.shading.curvature_valley_factor = 1.2

    minimum, maximum = object_bounds(obj)
    center = (minimum + maximum) * 0.5
    width = maximum.x - minimum.x
    depth = maximum.y - minimum.y
    height = maximum.z - minimum.z
    radius = max(width, depth, height)

    camera_data = bpy.data.cameras.new("ETC_PREVIEW_CAMERA")
    camera = bpy.data.objects.new("ETC_PREVIEW_CAMERA", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type = "ORTHO"
    camera_data.lens = 55

    paths: list[str] = []
    distance = radius * 2.8
    for index, degrees in enumerate((0, 90, 180, 270), start=1):
        radians = math.radians(degrees)
        camera.location = Vector((
            center.x + math.sin(radians) * distance,
            center.y - math.cos(radians) * distance,
            center.z + height * 0.04,
        ))
        point_camera(camera, center)
        projected_width = width if degrees in (0, 180) else depth
        camera_data.ortho_scale = max(height * 1.15, projected_width * 1.18)
        path = output_dir / f"{label}_view_{index}_{degrees:03d}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        paths.append(str(path))
    bpy.data.objects.remove(camera, do_unlink=True)
    return paths


def build_variant(source: Path, output: Path, preview_dir: Path | None,
                  pitch: float, target_triangles: int,
                  smooth_iterations: int, smooth_strength: float,
                  base_height: float, base_margin: float,
                  label: str) -> dict[str, Any]:
    started = time.monotonic()
    obj = import_source(source)
    source_stats = mesh_stats(obj)
    base = add_base_if_needed(obj, base_height, base_margin)
    apply_transforms(obj)
    pre_remesh = mesh_stats(obj)
    voxel_union(obj, pitch)
    post_remesh = mesh_stats(obj)
    volume_preserving_smooth(obj, smooth_iterations, smooth_strength)
    grounding = flatten_base_and_ground(obj, pitch)
    decimation = decimate_if_needed(obj, target_triangles)
    finalize_mesh(obj)
    final_stats = mesh_stats(obj)
    export_stl(obj, output)
    previews = render_views(obj, preview_dir, label) if preview_dir else []
    return {
        "label": label,
        "output": str(output),
        "bytes": output.stat().st_size,
        "voxel_pitch_mm": pitch,
        "smooth_iterations": smooth_iterations,
        "smooth_strength": smooth_strength,
        "target_triangles": target_triangles,
        "source_stats": source_stats,
        "base": base,
        "pre_remesh_stats": pre_remesh,
        "post_remesh_stats": post_remesh,
        "grounding": grounding,
        "decimation": decimation,
        "final_stats": final_stats,
        "preview_files": previews,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    previews = output_dir / "Previews"

    high_path = output_dir / "ETC_High_Detail_Super_Smooth.stl"
    print_path = output_dir / "ETC_Print_Ready_Smooth.stl"
    report: dict[str, Any] = {
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "blender_version": bpy.app.version_string,
        "variants": [],
    }

    report["variants"].append(build_variant(
        source=source,
        output=high_path,
        preview_dir=previews,
        pitch=args.high_pitch_mm,
        target_triangles=args.high_target_triangles,
        smooth_iterations=3,
        smooth_strength=0.075,
        base_height=args.base_height_mm,
        base_margin=args.base_margin_mm,
        label="ETC_High_Detail",
    ))
    report["variants"].append(build_variant(
        source=source,
        output=print_path,
        preview_dir=None,
        pitch=args.print_pitch_mm,
        target_triangles=args.print_target_triangles,
        smooth_iterations=4,
        smooth_strength=0.09,
        base_height=args.base_height_mm,
        base_margin=args.base_margin_mm,
        label="ETC_Print_Ready",
    ))

    report_path = output_dir / "blender_build_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
