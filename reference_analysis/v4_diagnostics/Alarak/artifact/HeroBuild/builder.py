#!/usr/bin/env python3
"""
Build printable STL files from Heroes of the Storm .m3 source meshes.

This script is intended to be run by Blender 3.6 in background mode. It:
  * discovers canonical base hero meshes in one or more checked-out repositories;
  * imports the M3 geometry with SC2Mapster/m3addon;
  * optionally imports a matching required-animation M3A and chooses a neutral pose;
  * exports a detail-preserving STL;
  * creates a watertight, voxel-remeshed, based STL for practical printing;
  * records a machine-readable and human-readable validation manifest.

The source meshes remain Blizzard Entertainment intellectual property. The generated
files are intended for personal, non-commercial fan use only.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import time
import traceback
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import bpy
import bmesh
from mathutils import Matrix, Vector


BUILDER_VERSION = "2.1.0"
M3ADDON_COMMIT = "58935005c4465a219a740f6b740253533eda773f"
BLENDER_VERSION = "3.6.23"

# 90 playable hero selections as at 2026-10-04. Cho and Gall deliberately share
# one in-game body/source mesh and are emitted as separate named STL aliases.
HERO_SOURCE_CANDIDATES: Dict[str, Sequence[str]] = {
    "Abathur": ("abathur",),
    "Alarak": ("alarak",),
    "Alexstrasza": ("alexstrasza",),
    "Ana": ("ana",),
    "Anduin": ("anduin",),
    "Anub'arak": ("anubarak",),
    "Artanis": ("artanis",),
    "Arthas": ("arthas",),
    "Auriel": ("auriel",),
    "Azmodan": ("azmodan",),
    "Blaze": ("firebat",),
    "Brightwing": ("brightwing",),
    "The Butcher": ("butcher",),
    "Cassia": ("d2amazonf",),
    "Chen": ("chen",),
    "Cho": ("chogall",),
    "Chromie": ("chromie",),
    "D.Va": ("dva", "dvamech"),
    "Deathwing": ("deathwing",),
    "Deckard": ("deckard",),
    "Dehaka": ("dehaka",),
    "Diablo": ("diablo",),
    "E.T.C.": ("etc",),
    "Falstad": ("falstad",),
    "Fenix": ("fenix",),
    "Gall": ("chogall",),
    "Garrosh": ("garrosh",),
    "Gazlowe": ("gazlowe",),
    "Genji": ("genji",),
    "Greymane": ("greymane",),
    "Gul'dan": ("guldan",),
    "Hanzo": ("hanzo",),
    "Hogger": ("hogger",),
    "Illidan": ("illidan",),
    "Imperius": ("imperius",),
    "Jaina": ("jaina",),
    "Johanna": ("d3crusaderf",),
    "Junkrat": ("junkrat",),
    "Kael'thas": ("kaelthas",),
    "Kel'Thuzad": ("kelthuzad",),
    "Kerrigan": ("kerrigan",),
    "Kharazim": ("d3monkm",),
    "Leoric": ("kingleoric",),
    "Li Li": ("lili",),
    "Li-Ming": ("d3wizardf",),
    "Lt. Morales": ("medic",),
    "Lucio": ("lucio",),
    "Lunara": ("dryad",),
    "Maiev": ("maiev",),
    "Mal'Ganis": ("malganis",),
    "Malfurion": ("malfurion",),
    "Malthael": ("malthael",),
    "Medivh": ("medivh",),
    "Mei": ("meiow",),
    "Mephisto": ("mephisto",),
    "Muradin": ("muradin",),
    "Murky": ("murky",),
    "Nazeebo": ("d3witchdoctorm",),
    "Nova": ("nova",),
    "Orphea": ("orphea",),
    "Probius": ("probius",),
    "Qhira": ("nexushunter",),
    "Ragnaros": ("ragnaros",),
    "Raynor": ("raynor",),
    "Rehgar": ("rehgar",),
    "Rexxar": ("rexxar",),
    "Samuro": ("samuro",),
    "Sgt. Hammer": ("sgthammer",),
    "Sonya": ("d3barbarianf",),
    "Stitches": ("stitches",),
    "Stukov": ("stukov",),
    "Sylvanas": ("sylvanas",),
    "Tassadar": ("tassadar",),
    "The Lost Vikings": ("lostbaleog", "losterik", "lostolaf"),
    "Thrall": ("thrall",),
    "Tracer": ("tracer",),
    "Tychus": ("tychus",),
    "Tyrael": ("tyrael",),
    "Tyrande": ("tyrande",),
    "Uther": ("uther",),
    "Valeera": ("valeera",),
    "Valla": ("d3demonhunterf",),
    "Varian": ("varian",),
    "Whitemane": ("whitemane",),
    "Xul": ("d2necrom",),
    "Yrel": ("yrel",),
    "Zagara": ("zagara",),
    "Zarya": ("zarya",),
    "Zeratul": ("zeratul",),
    "Zul'jin": ("zuljin",),
}

POSE_PREFERENCES = (
    "stand",
    "idle",
    "ready",
    "portrait",
    "walk",
)
POSE_EXCLUDES = (
    "death",
    "dead",
    "ragdoll",
    "stun",
    "knock",
    "victory",
    "dance",
    "taunt",
    "mount",
    "attack",
    "spell",
)
EXCLUDED_SOURCE_TOKENS = (
    "death",
    "ragdoll",
    "facial",
    "portrait",
    "skin",
    "mount",
    "minion",
    "summon",
    "effect",
    "fx",
    "ability",
    "weapon",
)


def log(message: str) -> None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def safe_filename(name: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    result = re.sub(r"_+", "_", result).strip("_.")
    return result or "unnamed"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, help="Checked-out repository root. Repeatable.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-height-mm", type=float, default=110.0)
    parser.add_argument("--base-height-mm", type=float, default=4.0)
    parser.add_argument("--base-margin-mm", type=float, default=4.0)
    parser.add_argument("--voxel-mm", type=float, default=0.38)
    parser.add_argument("--thin-part-mm", type=float, default=0.70)
    parser.add_argument("--max-triangles", type=int, default=650_000)
    parser.add_argument("--only", action="append", default=[], help="Hero display name or source key for smoke builds.")
    parser.add_argument("--skip-animation", action="store_true")
    parser.add_argument("--keep-blend", action="store_true")
    return parser.parse_args(argv)


def reset_scene() -> None:
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.armatures,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(datablocks):
            try:
                datablocks.remove(block, do_unlink=True)
            except TypeError:
                try:
                    datablocks.remove(block)
                except Exception:
                    pass
            except Exception:
                pass
    for action in list(bpy.data.actions):
        try:
            bpy.data.actions.remove(action)
        except Exception:
            pass
    scene = bpy.context.scene
    scene.frame_set(0)

    # The M3 add-on stores imported animations, materials and effects in custom
    # scene collections. Clear those collections between heroes so a long batch
    # does not accidentally reuse an earlier hero's pose or leak memory.
    for attribute_name in dir(scene):
        if not attribute_name.startswith("m3_") or attribute_name.endswith("_options"):
            continue
        try:
            value = getattr(scene, attribute_name)
        except Exception:
            continue
        clear_method = getattr(value, "clear", None)
        if callable(clear_method):
            try:
                clear_method()
            except Exception:
                pass

    if hasattr(scene, "m3_animation_index"):
        try:
            scene.m3_animation_index = -1
        except Exception:
            pass


def enable_m3_addon() -> None:
    import addon_utils

    loaded, enabled = addon_utils.check("m3addon")
    if not enabled:
        log("Enabling m3addon")
        addon_utils.enable("m3addon", default_set=False, persistent=False)
    loaded, enabled = addon_utils.check("m3addon")
    if not enabled:
        raise RuntimeError("m3addon could not be enabled")


def discover_base_files(roots: Sequence[Path]) -> Dict[str, List[Path]]:
    """Return source-key -> candidate .m3 files."""
    result: Dict[str, List[Path]] = {}
    pattern = re.compile(r"^storm_hero_(?P<key>[a-z0-9_]+)_base(?:_v[0-9]+)?\.m3$", re.I)
    for root in roots:
        if not root.exists():
            log(f"WARNING: source root does not exist: {root}")
            continue
        for path in root.rglob("*.m3"):
            match = pattern.match(path.name)
            if not match:
                continue
            # Canonical source directories normally have the same name as the file.
            # Keep mismatches only as lower-priority fallbacks.
            key = match.group("key").lower()
            result.setdefault(key, []).append(path)
    for paths in result.values():
        paths.sort(key=lambda p: (p.parent.name.lower() != p.stem.lower(), len(p.as_posix()), p.as_posix()))
    return result


def discover_all_m3(roots: Sequence[Path]) -> List[Path]:
    paths: List[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(root.rglob("*.m3"))
    return sorted(set(paths))


def choose_source_for_hero(
    hero: str,
    candidates: Sequence[str],
    base_index: Dict[str, List[Path]],
    all_m3: Sequence[Path],
) -> Tuple[Optional[str], List[Path], str]:
    # Composite roster entries. D.Va includes pilot and mech; The Lost Vikings
    # includes all three playable characters. Each component is preserved and laid out
    # together on one print base by process_source_group.
    if hero in {"D.Va", "The Lost Vikings"}:
        selected = []
        for key in candidates:
            if key in base_index:
                selected.append(base_index[key][0])
        if len(selected) == len(candidates):
            return ("dva_composite" if hero == "D.Va" else "lostvikings_composite"), selected, "explicit multi-model composite"

    # First, exact canonical key.
    for key in candidates:
        if key in base_index:
            return key, [base_index[key][0]], "exact canonical base"

    # Second, conservative substring match among base files. Prefer the shortest key,
    # then a parent/file stem match and paths without excluded tokens.
    ranked: List[Tuple[int, int, str, Path]] = []
    for key, paths in base_index.items():
        if not any(candidate in key for candidate in candidates):
            continue
        penalty = sum(20 for token in EXCLUDED_SOURCE_TOKENS if token in key)
        penalty += key.count("_") * 3
        for path in paths:
            ranked.append((penalty, len(key), key, path))
    if ranked:
        ranked.sort(key=lambda row: (row[0], row[1], row[2], row[3].as_posix()))
        penalty, _, key, path = ranked[0]
        return key, [path], f"fallback base match (penalty {penalty})"

    # The Lost Vikings are sometimes stored as three separate character meshes rather
    # than one combined canonical base model. Build a multi-source group when found.
    if hero == "The Lost Vikings":
        viking_paths: List[Path] = []
        for path in all_m3:
            stem = path.stem.lower()
            joined = f"{path.parent.name.lower()}/{stem}"
            if "lostviking" not in joined and "lost_viking" not in joined:
                continue
            if any(token in joined for token in EXCLUDED_SOURCE_TOKENS):
                continue
            if any(name in joined for name in ("baleog", "erik", "olaf")) or stem.endswith("_base"):
                viking_paths.append(path)
        # De-duplicate by likely character token.
        selected: List[Path] = []
        for token in ("olaf", "baleog", "erik"):
            matches = [p for p in viking_paths if token in p.as_posix().lower()]
            if matches:
                selected.append(sorted(matches, key=lambda p: (len(p.as_posix()), p.as_posix()))[0])
        if not selected and viking_paths:
            selected = sorted(viking_paths, key=lambda p: (len(p.as_posix()), p.as_posix()))[:3]
        if selected:
            return "lostvikings_multi", selected, "three-character composite fallback"

    return None, [], "not found"


def resolve_build_plan(
    roots: Sequence[Path], only: Sequence[str]
) -> Tuple[List[dict], Dict[str, List[Path]]]:
    base_index = discover_base_files(roots)
    all_m3 = discover_all_m3(roots)
    filters = {item.casefold() for item in only}
    plan: List[dict] = []
    source_cache: Dict[str, List[Path]] = {}

    for hero, candidates in HERO_SOURCE_CANDIDATES.items():
        if filters and hero.casefold() not in filters and not any(c.casefold() in filters for c in candidates):
            continue
        key, paths, reason = choose_source_for_hero(hero, candidates, base_index, all_m3)
        entry = {
            "hero": hero,
            "source_key": key,
            "source_paths": [str(p) for p in paths],
            "resolution": reason,
            "status": "pending" if paths else "missing-source",
        }
        plan.append(entry)
        if key and paths:
            source_cache.setdefault(key, paths)

    return plan, source_cache


def find_required_animation(model_path: Path, source_key: str) -> Optional[Path]:
    root = model_path.parents[1] if len(model_path.parents) > 1 else model_path.parent
    patterns = (
        f"storm_hero_{source_key}_requiredanims/**/*.m3a",
        f"storm_hero_{source_key}_requiredanims/*.m3a",
        f"storm_hero_{source_key}*requiredanims*/*.m3a",
    )
    found: List[Path] = []
    for pattern in patterns:
        found.extend(root.glob(pattern))
    unique = sorted(set(found))
    if not unique:
        return None

    def rank(path: Path) -> Tuple[int, int, int, str]:
        lower = path.as_posix().lower()
        facial_penalty = 100 if "facial" in lower else 0
        exact_bonus = -20 if path.stem == f"storm_hero_{source_key}_requiredanims" else 0
        return (facial_penalty + exact_bonus, -path.stat().st_size, len(lower), lower)

    unique.sort(key=rank)
    return unique[0]


def configure_import_options(model_path: Path, armature: Optional[bpy.types.Object] = None) -> None:
    scene = bpy.context.scene
    options = scene.m3_import_options
    options.path = str(model_path)
    options.rootDirectory = str(model_path.parents[1] if len(model_path.parents) > 1 else model_path.parent)
    options.generateBlenderMaterials = False
    options.applySmoothShading = True
    options.markSharpEdges = True
    options.contentPreset = "EVERYTHING" if model_path.suffix.lower() == ".m3a" else "MESH_MATERIALS_RIG"
    options.armatureObject = armature


def import_m3(model_path: Path, armature: Optional[bpy.types.Object] = None) -> List[bpy.types.Object]:
    before = set(bpy.data.objects)
    configure_import_options(model_path, armature)
    result = getattr(bpy.ops.m3, "import")(filepath=str(model_path))
    if "FINISHED" not in result:
        raise RuntimeError(f"M3 import returned {result} for {model_path}")
    return [obj for obj in bpy.data.objects if obj not in before]


def choose_pose() -> Optional[dict]:
    scene = bpy.context.scene
    animations = list(getattr(scene, "m3_animations", []))
    if not animations:
        return None

    names = [getattr(anim, "name", "") for anim in animations]
    selected_index: Optional[int] = None
    selected_reason = "first available"
    for preferred in POSE_PREFERENCES:
        for index, name in enumerate(names):
            lower = name.casefold()
            if preferred not in lower:
                continue
            if any(excluded in lower for excluded in POSE_EXCLUDES):
                continue
            selected_index = index
            selected_reason = f"matched '{preferred}'"
            break
        if selected_index is not None:
            break
    if selected_index is None:
        for index, name in enumerate(names):
            lower = name.casefold()
            if not any(excluded in lower for excluded in POSE_EXCLUDES):
                selected_index = index
                break
    if selected_index is None:
        selected_index = 0

    scene.m3_animation_index = selected_index
    animation = animations[selected_index]
    start = int(getattr(animation, "startFrame", scene.frame_start))
    end_exclusive = int(getattr(animation, "exlusiveEndFrame", scene.frame_end + 1))
    duration = max(1, end_exclusive - start)
    # Avoid frame zero transition artifacts; take a calm early-middle sample.
    frame = start + min(max(1, duration // 4), 12)
    frame = min(frame, max(start, end_exclusive - 1))
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    return {
        "animation": names[selected_index],
        "index": selected_index,
        "frame": frame,
        "reason": selected_reason,
        "animation_count": len(names),
    }


def import_source_group(paths: Sequence[Path], source_key: str, skip_animation: bool) -> dict:
    imported_paths: List[str] = []
    animation_paths: List[str] = []
    pose_info: Optional[dict] = None
    armatures_before = set(obj for obj in bpy.data.objects if obj.type == "ARMATURE")

    for index, path in enumerate(paths):
        log(f"Importing M3: {path}")
        created_objects = import_m3(path)
        for created_object in created_objects:
            created_object["hots_source_index"] = index
        imported_paths.append(str(path))
        bpy.context.view_layer.update()

        # For multi-source composites (Lost Vikings fallback), distribute components
        # later after evaluating their meshes; each source gets its own armature.
        if index == 0 and not skip_animation:
            armatures_after = [obj for obj in bpy.data.objects if obj.type == "ARMATURE" and obj not in armatures_before]
            armature = armatures_after[0] if armatures_after else next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
            animation = find_required_animation(path, source_key)
            if animation and armature:
                try:
                    log(f"Importing required animation: {animation}")
                    import_m3(animation, armature=armature)
                    animation_paths.append(str(animation))
                    pose_info = choose_pose()
                except Exception as exc:
                    log(f"WARNING: required animation import/selection failed: {exc}")
                    log(traceback.format_exc())
            else:
                pose_info = choose_pose()

    return {
        "imported_paths": imported_paths,
        "animation_paths": animation_paths,
        "pose": pose_info,
    }


def evaluated_mesh_objects() -> List[bpy.types.Object]:
    """Bake every visible, printable evaluated mesh while excluding rendering helpers."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    objects: List[bpy.types.Object] = []
    excluded = (
        "collision", "physics", "ragdoll", "shadow", "holo", "decal",
        "volume", "ribbon", "particle", "emitter", "hitbox", "selection",
        "placement", "bounds", "proxy", "helper", "reference", "ref_",
        "camera", "light", "weapontrail", "trail_",
    )
    for original in list(bpy.context.scene.objects):
        if original.type != "MESH" or original.hide_render or original.hide_get():
            continue
        lower_name = original.name.casefold()
        if any(token in lower_name for token in excluded):
            continue
        evaluated = original.evaluated_get(depsgraph)
        mesh = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph, preserve_all_data_layers=False)
        if not mesh or len(mesh.vertices) < 3 or len(mesh.polygons) < 1:
            if mesh:
                bpy.data.meshes.remove(mesh)
            continue
        copy = bpy.data.objects.new(f"PRINT_{safe_filename(original.name)}", mesh)
        bpy.context.scene.collection.objects.link(copy)
        copy.matrix_world = original.matrix_world.copy()
        copy["hots_source_index"] = int(original.get("hots_source_index", 0))
        copy["hots_original_name"] = original.name
        objects.append(copy)
    return objects


def select_only(objects: Iterable[bpy.types.Object], active: Optional[bpy.types.Object] = None) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.hide_set(False)
        obj.select_set(True)
    if active is not None:
        bpy.context.view_layer.objects.active = active


def apply_world_transforms(objects: Sequence[bpy.types.Object]) -> None:
    for obj in objects:
        select_only([obj], obj)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def join_meshes(objects: Sequence[bpy.types.Object], name: str) -> bpy.types.Object:
    if not objects:
        raise RuntimeError("No mesh geometry was imported")
    select_only(objects, objects[0])
    if len(objects) > 1:
        bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    if joined is None:
        raise RuntimeError("Mesh join did not produce an active object")
    joined.name = name
    return joined


def clean_mesh_object(obj: bpy.types.Object, merge_distance: float = 0.00001) -> None:
    select_only([obj], obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    try:
        bpy.ops.mesh.remove_doubles(threshold=merge_distance)
    except Exception:
        try:
            bpy.ops.mesh.merge_by_distance(distance=merge_distance)
        except Exception:
            pass
    try:
        bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)
    except Exception:
        pass
    try:
        bpy.ops.mesh.normals_make_consistent(inside=False)
    except Exception:
        pass
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.data.validate(clean_customdata=True)
    obj.data.update()


def object_bounds(obj: bpy.types.Object) -> Tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    minimum = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    maximum = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return minimum, maximum


def normalize_to_height(obj: bpy.types.Object, target_height: float) -> dict:
    minimum, maximum = object_bounds(obj)
    dimensions = maximum - minimum
    current_height = max(dimensions.z, dimensions.y, dimensions.x)
    # Prefer Z as vertical if it is meaningful; M3 imports normally use Z-up.
    if dimensions.z > max(dimensions.x, dimensions.y) * 0.20:
        current_height = dimensions.z
    if current_height <= 1e-8:
        raise RuntimeError("Imported mesh has zero-size bounds")
    factor = target_height / current_height
    obj.scale = (factor, factor, factor)
    select_only([obj], obj)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    minimum, maximum = object_bounds(obj)
    obj.location -= Vector(((minimum.x + maximum.x) / 2.0, (minimum.y + maximum.y) / 2.0, minimum.z))
    bpy.context.view_layer.update()
    minimum2, maximum2 = object_bounds(obj)
    return {
        "scale_factor": factor,
        "bounds_before": [list(minimum), list(maximum)],
        "bounds_after": [list(minimum2), list(maximum2)],
        "dimensions_mm": list(maximum2 - minimum2),
    }



def group_bounds(objects: Sequence[bpy.types.Object]) -> Tuple[Vector, Vector]:
    mins, maxs = zip(*(object_bounds(obj) for obj in objects))
    minimum = Vector((min(v.x for v in mins), min(v.y for v in mins), min(v.z for v in mins)))
    maximum = Vector((max(v.x for v in maxs), max(v.y for v in maxs), max(v.z for v in maxs)))
    return minimum, maximum


def normalize_object_group(objects: Sequence[bpy.types.Object], target_height: float) -> dict:
    minimum, maximum = group_bounds(objects)
    dimensions = maximum - minimum
    current_height = dimensions.z if dimensions.z > max(dimensions.x, dimensions.y) * 0.20 else max(dimensions)
    if current_height <= 1e-8:
        raise RuntimeError("Imported mesh group has zero-size bounds")
    factor = target_height / current_height
    for obj in objects:
        obj.scale = (factor, factor, factor)
    apply_world_transforms(objects)
    minimum, maximum = group_bounds(objects)
    shift = Vector((-(minimum.x + maximum.x) / 2.0, -(minimum.y + maximum.y) / 2.0, -minimum.z))
    for obj in objects:
        obj.location += shift
    bpy.context.view_layer.update()
    minimum2, maximum2 = group_bounds(objects)
    return {
        "scale_factor": factor,
        "bounds_before": [list(minimum), list(maximum)],
        "bounds_after": [list(minimum2), list(maximum2)],
        "dimensions_mm": list(maximum2 - minimum2),
    }


def mesh_topology(obj: bpy.types.Object) -> dict:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    boundary = sum(1 for edge in bm.edges if len(edge.link_faces) == 1)
    nonmanifold = sum(1 for edge in bm.edges if len(edge.link_faces) not in (1, 2))
    edges = len(bm.edges)
    bm.free()
    minimum, maximum = object_bounds(obj)
    dimensions = maximum - minimum
    return {
        "boundary_edges": boundary,
        "nonmanifold_edges": nonmanifold,
        "edges": edges,
        "boundary_ratio": boundary / max(1, edges),
        "dimensions": list(dimensions),
        "min_dimension": min(dimensions),
        "max_dimension": max(dimensions),
    }


def prepare_print_component(obj: bpy.types.Object, thickness: float) -> dict:
    clean_mesh_object(obj, merge_distance=0.005)
    topo = mesh_topology(obj)
    name = str(obj.get("hots_original_name", obj.name)).casefold()
    sheet_tokens = (
        "cape", "cloak", "cloth", "hair", "wing", "feather", "banner",
        "skirt", "robe", "coat", "scarf", "veil", "tabard", "loin",
        "leaf", "fin", "membrane", "flag", "dress", "tassel",
    )
    sheet_like = (
        topo["boundary_edges"] > 0
        and (
            topo["min_dimension"] < max(0.9, topo["max_dimension"] * 0.035)
            or topo["boundary_ratio"] > 0.22
            or any(token in name for token in sheet_tokens)
        )
    )
    if sheet_like:
        solidify_open_surfaces(obj, thickness)
        clean_mesh_object(obj, merge_distance=0.005)

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.normal_update()
    for vertex in bm.verts:
        if vertex.normal.length_squared > 1e-16:
            vertex.co += vertex.normal.normalized() * 0.08
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return {"name": name, "sheet_solidified": sheet_like, "topology_before": topo}


def final_mesh_cleanup(obj: bpy.types.Object, weld_distance: float) -> None:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    if bm.verts:
        bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=weld_distance)
    if bm.edges:
        try:
            bmesh.ops.dissolve_degenerate(bm, edges=list(bm.edges), dist=max(1e-6, weld_distance * 0.25))
        except Exception:
            pass
    if bm.faces:
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.validate(clean_customdata=True)
    obj.data.update()


def duplicate_object(obj: bpy.types.Object, name: str) -> bpy.types.Object:
    clone = obj.copy()
    clone.data = obj.data.copy()
    clone.name = name
    bpy.context.scene.collection.objects.link(clone)
    return clone


def triangulate_object(obj: bpy.types.Object) -> None:
    modifier = obj.modifiers.new(name="Triangulate", type="TRIANGULATE")
    modifier.quad_method = "BEAUTY"
    modifier.ngon_method = "BEAUTY"
    select_only([obj], obj)
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def solidify_open_surfaces(obj: bpy.types.Object, thickness: float) -> None:
    modifier = obj.modifiers.new(name="Minimum wall thickness", type="SOLIDIFY")
    modifier.thickness = thickness
    modifier.offset = 0.0
    modifier.use_even_offset = True
    modifier.use_quality_normals = True
    select_only([obj], obj)
    try:
        bpy.ops.object.modifier_apply(modifier=modifier.name)
    except Exception as exc:
        log(f"WARNING: Solidify failed; continuing to voxel remesh: {exc}")
        try:
            obj.modifiers.remove(modifier)
        except Exception:
            pass


def add_base_for_object(obj: bpy.types.Object, height: float, margin: float) -> bpy.types.Object:
    minimum, maximum = object_bounds(obj)
    width = max(10.0, maximum.x - minimum.x)
    depth = max(10.0, maximum.y - minimum.y)
    radius = max(width, depth) / 2.0 + margin
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=radius, depth=height, location=(0.0, 0.0, -height / 2.0 + 0.35))
    base = bpy.context.active_object
    base.name = "PRINT_BASE"
    # Slightly push the character into the base to guarantee a connected voxel union.
    obj.location.z -= 0.25
    bpy.context.view_layer.update()
    return base


def voxel_remesh_object(obj: bpy.types.Object, voxel_size: float) -> None:
    select_only([obj], obj)
    # Blender 3.6 object voxel remesh operator uses mesh voxel properties.
    obj.data.remesh_voxel_size = voxel_size
    obj.data.remesh_voxel_adaptivity = 0.0
    try:
        bpy.ops.object.voxel_remesh()
        return
    except Exception as first_exc:
        log(f"Object voxel_remesh operator failed ({first_exc}); trying Remesh modifier")

    modifier = obj.modifiers.new(name="Voxel Remesh", type="REMESH")
    if hasattr(modifier, "mode"):
        modifier.mode = "VOXEL"
    if hasattr(modifier, "voxel_size"):
        modifier.voxel_size = voxel_size
    if hasattr(modifier, "use_smooth_shade"):
        modifier.use_smooth_shade = True
    select_only([obj], obj)
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def decimate_if_needed(obj: bpy.types.Object, max_triangles: int) -> dict:
    triangulate_object(obj)
    triangles_before = len(obj.data.polygons)
    info = {"triangles_before_decimate": triangles_before, "decimated": False}
    if triangles_before > max_triangles:
        ratio = max(0.05, min(1.0, max_triangles / triangles_before))
        modifier = obj.modifiers.new(name="Triangle budget", type="DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        modifier.ratio = ratio
        modifier.use_collapse_triangulate = True
        select_only([obj], obj)
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        triangulate_object(obj)
        info.update({"decimated": True, "decimate_ratio": ratio})
    info["triangles_after_decimate"] = len(obj.data.polygons)
    return info


def write_binary_stl(obj: bpy.types.Object, path: Path, solid_name: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
    try:
        mesh.calc_loop_triangles()
        matrix = evaluated.matrix_world
        triangles = mesh.loop_triangles
        header_text = f"HOTS STL | {solid_name} | builder {BUILDER_VERSION}".encode("ascii", errors="replace")[:80]
        header = header_text.ljust(80, b"\0")
        with path.open("wb") as handle:
            handle.write(header)
            handle.write(struct.pack("<I", len(triangles)))
            for tri in triangles:
                coords = [matrix @ mesh.vertices[index].co for index in tri.vertices]
                normal = (coords[1] - coords[0]).cross(coords[2] - coords[0])
                if normal.length_squared > 1e-20:
                    normal.normalize()
                else:
                    normal = Vector((0.0, 0.0, 0.0))
                handle.write(struct.pack("<12fH", normal.x, normal.y, normal.z,
                                         coords[0].x, coords[0].y, coords[0].z,
                                         coords[1].x, coords[1].y, coords[1].z,
                                         coords[2].x, coords[2].y, coords[2].z, 0))
        minimum, maximum = object_bounds(obj)
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "triangles": len(triangles),
            "bounds_mm": [list(minimum), list(maximum)],
            "dimensions_mm": list(maximum - minimum),
        }
    finally:
        evaluated.to_mesh_clear()


def validate_binary_stl(path: Path, edge_quantum: float = 0.01) -> dict:
    """Validate binary STL and estimate manifoldness by quantized edge incidence."""
    result = {
        "valid_binary_stl": False,
        "watertight_edge_test": False,
        "boundary_edges": None,
        "nonmanifold_edges": None,
        "degenerate_triangles": 0,
    }
    data = path.read_bytes()
    if len(data) < 84:
        result["error"] = "file shorter than binary STL header"
        return result
    tri_count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + tri_count * 50
    if expected != len(data):
        result["error"] = f"size mismatch: header says {tri_count} triangles ({expected} bytes), file has {len(data)} bytes"
        return result
    result["valid_binary_stl"] = True
    result["triangles"] = tri_count

    # Keep validation memory bounded by counting compact integer edges.
    edge_counts: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], int] = {}
    offset = 84
    for _ in range(tri_count):
        values = struct.unpack_from("<12fH", data, offset)
        offset += 50
        verts = [values[3:6], values[6:9], values[9:12]]
        qverts = [tuple(int(round(c / edge_quantum)) for c in v) for v in verts]
        if len(set(qverts)) < 3:
            result["degenerate_triangles"] += 1
        for a, b in ((qverts[0], qverts[1]), (qverts[1], qverts[2]), (qverts[2], qverts[0])):
            edge = (a, b) if a <= b else (b, a)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    boundary = sum(1 for count in edge_counts.values() if count == 1)
    nonmanifold = sum(1 for count in edge_counts.values() if count > 2)
    result["boundary_edges"] = boundary
    result["nonmanifold_edges"] = nonmanifold
    result["watertight_edge_test"] = boundary == 0 and nonmanifold == 0 and result["degenerate_triangles"] == 0
    return result


def process_source_group(
    source_key: str,
    paths: Sequence[Path],
    output_root: Path,
    args: argparse.Namespace,
) -> dict:
    started = time.monotonic()
    reset_scene()
    import_info = import_source_group(paths, source_key, args.skip_animation)
    mesh_objects = evaluated_mesh_objects()
    if not mesh_objects:
        raise RuntimeError("M3 import created no printable evaluated mesh objects")

    baked_set = set(mesh_objects)
    for obj in list(bpy.context.scene.objects):
        if obj not in baked_set:
            bpy.data.objects.remove(obj, do_unlink=True)
    apply_world_transforms(mesh_objects)

    if len(paths) > 1:
        grouped: Dict[int, List[bpy.types.Object]] = {}
        for obj in mesh_objects:
            grouped.setdefault(int(obj.get("hots_source_index", 0)), []).append(obj)
        ordered_groups = [grouped[key] for key in sorted(grouped)]
        bounds = [group_bounds(group) for group in ordered_groups]
        widths = [max(0.1, maximum.x - minimum.x) for minimum, maximum in bounds]
        gap = max(0.25, max(widths) * 0.14)
        cursor = 0.0
        centres = []
        for width in widths:
            centres.append(cursor + width / 2.0)
            cursor += width + gap
        overall_centre = (centres[0] + centres[-1]) / 2.0
        for group, (minimum, maximum), centre in zip(ordered_groups, bounds, centres):
            current_centre = (minimum.x + maximum.x) / 2.0
            shift = centre - overall_centre - current_centre
            for obj in group:
                obj.location.x += shift
        bpy.context.view_layer.update()

    normalization = normalize_object_group(mesh_objects, args.target_height_mm)

    detail_parts = [duplicate_object(obj, f"DETAIL_{index}_{obj.name}") for index, obj in enumerate(mesh_objects)]
    detail_obj = join_meshes(detail_parts, f"DETAIL_{source_key}")
    clean_mesh_object(detail_obj, merge_distance=0.002)
    triangulate_object(detail_obj)
    detail_path = output_root / ".build_cache" / "detail" / f"{safe_filename(source_key)}.stl"
    detail_export = write_binary_stl(detail_obj, detail_path, source_key)
    detail_export["validation"] = validate_binary_stl(detail_path)

    component_reports = []
    for obj in mesh_objects:
        component_reports.append(prepare_print_component(obj, args.thin_part_mm))

    base_minimum, base_maximum = group_bounds(mesh_objects)
    width = max(10.0, base_maximum.x - base_minimum.x)
    depth = max(10.0, base_maximum.y - base_minimum.y)
    radius = max(width, depth) / 2.0 + args.base_margin_mm
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=96,
        radius=radius,
        depth=args.base_height_mm,
        location=(0.0, 0.0, -args.base_height_mm / 2.0 + 0.70),
    )
    base = bpy.context.active_object
    base.name = "PRINT_BASE"
    for obj in mesh_objects:
        obj.location.z -= 0.55
    bpy.context.view_layer.update()
    apply_world_transforms(mesh_objects + [base])

    print_obj = join_meshes(mesh_objects + [base], f"PRINT_READY_{source_key}")
    clean_mesh_object(print_obj, merge_distance=max(0.005, args.voxel_mm * 0.03))
    voxel_remesh_object(print_obj, args.voxel_mm)
    final_mesh_cleanup(print_obj, max(0.005, args.voxel_mm * 0.06))
    decimation = decimate_if_needed(print_obj, args.max_triangles)
    final_mesh_cleanup(print_obj, max(0.005, args.voxel_mm * 0.05))

    minimum, maximum = object_bounds(print_obj)
    print_obj.location.z -= minimum.z
    bpy.context.view_layer.update()
    print_path = output_root / ".build_cache" / "print_ready" / f"{safe_filename(source_key)}.stl"
    print_export = write_binary_stl(print_obj, print_path, source_key)
    print_export["validation"] = validate_binary_stl(print_path)

    if not print_export["validation"]["watertight_edge_test"]:
        retry_voxel = max(args.voxel_mm, 0.62)
        log(f"Strict edge test failed; retrying clean voxel union at {retry_voxel:.3f} mm")
        voxel_remesh_object(print_obj, retry_voxel)
        final_mesh_cleanup(print_obj, max(0.008, retry_voxel * 0.05))
        decimation["retry_voxel_mm"] = retry_voxel
        decimation["retry"] = decimate_if_needed(print_obj, args.max_triangles)
        minimum, maximum = object_bounds(print_obj)
        print_obj.location.z -= minimum.z
        bpy.context.view_layer.update()
        print_export = write_binary_stl(print_obj, print_path, source_key)
        print_export["validation"] = validate_binary_stl(print_path)

    blend_path = None
    if args.keep_blend:
        blend_path = output_root / "03_Blender_Source" / f"{safe_filename(source_key)}.blend"
        blend_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), compress=True)

    return {
        "source_key": source_key,
        "source_paths": [str(p) for p in paths],
        "import": import_info,
        "normalization": normalization,
        "components": component_reports,
        "detail": detail_export,
        "print_ready": print_export,
        "decimation": decimation,
        "blend_path": str(blend_path) if blend_path else None,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


def copy_source_result_for_hero(source_result: dict, hero: str, output_root: Path) -> dict:
    hero_stem = safe_filename(hero)
    source_key = source_result["source_key"]
    hero_outputs: Dict[str, dict] = {}
    for label, folder in (
        ("detail", "01_Detail_Preserving"),
        ("print_ready", "02_Print_Ready_Watertight"),
    ):
        original = Path(source_result[label]["path"])
        target = output_root / folder / f"{hero_stem}.stl"
        target.parent.mkdir(parents=True, exist_ok=True)
        if original.resolve() != target.resolve():
            shutil.copy2(original, target)
        info = dict(source_result[label])
        info.update({
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "source_artifact": str(original),
        })
        hero_outputs[label] = info
    return {
        "hero": hero,
        "source_key": source_key,
        "source_paths": source_result["source_paths"],
        "import": source_result["import"],
        "normalization": source_result["normalization"],
        "detail": hero_outputs["detail"],
        "print_ready": hero_outputs["print_ready"],
        "decimation": source_result["decimation"],
        "elapsed_source_seconds": source_result["elapsed_seconds"],
        "status": "success" if hero_outputs["print_ready"]["validation"]["valid_binary_stl"] else "failed-validation",
    }


def write_reports(output_root: Path, build: dict) -> None:
    reports = output_root / "00_Reports"
    reports.mkdir(parents=True, exist_ok=True)
    manifest_path = reports / "manifest.json"
    manifest_path.write_text(json.dumps(build, indent=2, sort_keys=False), encoding="utf-8")

    csv_path = reports / "manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "hero", "status", "source_key", "source_paths", "pose", "detail_stl",
            "detail_triangles", "detail_watertight", "print_ready_stl",
            "print_ready_triangles", "print_ready_watertight", "dimensions_mm", "error",
        ])
        writer.writeheader()
        for item in build["heroes"]:
            writer.writerow({
                "hero": item.get("hero"),
                "status": item.get("status"),
                "source_key": item.get("source_key"),
                "source_paths": " | ".join(item.get("source_paths", [])),
                "pose": json.dumps(item.get("import", {}).get("pose"), ensure_ascii=False),
                "detail_stl": item.get("detail", {}).get("path"),
                "detail_triangles": item.get("detail", {}).get("triangles"),
                "detail_watertight": item.get("detail", {}).get("validation", {}).get("watertight_edge_test"),
                "print_ready_stl": item.get("print_ready", {}).get("path"),
                "print_ready_triangles": item.get("print_ready", {}).get("triangles"),
                "print_ready_watertight": item.get("print_ready", {}).get("validation", {}).get("watertight_edge_test"),
                "dimensions_mm": json.dumps(item.get("print_ready", {}).get("dimensions_mm")),
                "error": item.get("error"),
            })

    success = [item for item in build["heroes"] if item.get("status") == "success"]
    missing = [item for item in build["heroes"] if item.get("status") == "missing-source"]
    failed = [item for item in build["heroes"] if item.get("status") not in ("success", "missing-source")]
    nonwatertight = [item for item in success if not item.get("print_ready", {}).get("validation", {}).get("watertight_edge_test")]

    readme = f"""# Heroes of the Storm — Printable STL Package

Generated: {build['generated_utc']}  
Builder: {BUILDER_VERSION}  
Blender: {BLENDER_VERSION}  
M3 importer commit: `{M3ADDON_COMMIT}`

## What is included

- `01_Detail_Preserving/` — posed source geometry with the least destructive processing. These files retain the original game-mesh silhouette and fine polygon detail, but some may contain open or disconnected shells.
- `02_Print_Ready_Watertight/` — minimum wall thickness, circular base, voxel union/remesh, manifold cleanup and triangle-budgeting for practical slicing.
- `00_Reports/manifest.csv` and `manifest.json` — exact source paths, pose choice, dimensions, triangle counts, hashes and manifold validation.

## Build result

- Playable roster entries requested: {len(build['heroes'])}
- Successful: {len(success)}
- Missing source: {len(missing)}
- Failed: {len(failed)}
- Successful but failed strict watertight edge test: {len(nonwatertight)}

Cho and Gall share the same canonical in-game body mesh. Separate filenames are included for roster completeness. The Lost Vikings may be assembled from multiple character meshes when no combined canonical source is present.

## Printing guidance

The models are normalized to approximately {build['settings']['target_height_mm']} mm character height plus a {build['settings']['base_height_mm']} mm base. Start with the `02_Print_Ready_Watertight` version. Inspect the preview in your slicer, orient supports for weapons/capes/wings, and avoid scaling below roughly 70–80% unless your printer can reliably reproduce the resulting thin features.

## Important limitations

STL stores geometry only; it does not preserve game textures, materials, transparency, emissive effects or normal-map detail. Game assets also use thin cards, floating effects and animation rigs that were designed for real-time rendering, not manufacturing. The print-ready files therefore use geometric thickening and remeshing. Automated repair can preserve the overall character closely, but it cannot equal a human artist's character-by-character sculpting and support design.

## Rights

Heroes of the Storm and its characters/assets are owned by Blizzard Entertainment. These derivative files are intended for personal, non-commercial fan use. Do not sell or redistribute them without the relevant rights-holder's permission.
"""
    (output_root / "README_FIRST.txt").write_text(readme, encoding="utf-8")


def main() -> int:
    args = parse_args()
    roots = [Path(path).resolve() for path in args.source]
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "00_Reports").mkdir(parents=True, exist_ok=True)

    log(f"HOTS STL Builder {BUILDER_VERSION}")
    log(f"Blender runtime: {bpy.app.version_string}")
    log(f"Source roots: {roots}")
    enable_m3_addon()

    plan, source_cache = resolve_build_plan(roots, args.only)
    log(f"Resolved {len(plan)} roster entries to {len(source_cache)} unique source groups")
    for entry in plan:
        log(f"PLAN {entry['hero']}: {entry['source_key']} — {entry['resolution']}")

    build = {
        "builder_version": BUILDER_VERSION,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "blender_runtime": bpy.app.version_string,
        "expected_blender": BLENDER_VERSION,
        "m3addon_commit": M3ADDON_COMMIT,
        "source_roots": [str(root) for root in roots],
        "settings": {
            "target_height_mm": args.target_height_mm,
            "base_height_mm": args.base_height_mm,
            "base_margin_mm": args.base_margin_mm,
            "voxel_mm": args.voxel_mm,
            "thin_part_mm": args.thin_part_mm,
            "max_triangles": args.max_triangles,
            "skip_animation": args.skip_animation,
        },
        "heroes": [],
        "source_results": {},
    }

    source_results: Dict[str, dict] = {}
    for source_key, paths in source_cache.items():
        try:
            source_result = process_source_group(source_key, paths, output_root, args)
            source_results[source_key] = source_result
            build["source_results"][source_key] = source_result
            log(f"SUCCESS source {source_key}: {source_result['print_ready']['triangles']} print-ready triangles")
        except Exception as exc:
            error = {
                "source_key": source_key,
                "source_paths": [str(p) for p in paths],
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            source_results[source_key] = error
            build["source_results"][source_key] = error
            log(f"FAILED source {source_key}: {exc}")
            log(error["traceback"])

    for entry in plan:
        hero = entry["hero"]
        source_key = entry.get("source_key")
        if not source_key:
            build["heroes"].append({**entry, "error": "No canonical or fallback base mesh found"})
            continue
        source_result = source_results.get(source_key, {})
        if "error" in source_result:
            build["heroes"].append({
                **entry,
                "status": "failed-conversion",
                "error": source_result["error"],
                "traceback": source_result.get("traceback"),
            })
            continue
        try:
            hero_result = copy_source_result_for_hero(source_result, hero, output_root)
            hero_result["resolution"] = entry["resolution"]
            build["heroes"].append(hero_result)
        except Exception as exc:
            build["heroes"].append({
                **entry,
                "status": "failed-packaging",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })

    write_reports(output_root, build)
    shutil.rmtree(output_root / ".build_cache", ignore_errors=True)
    successful = sum(1 for item in build["heroes"] if item.get("status") == "success")
    failed = len(build["heroes"]) - successful
    log(f"Completed: {successful} successful roster entries, {failed} missing/failed")
    # Smoke and full builds should complete and publish diagnostics even if individual
    # heroes fail; workflow enforces a minimum success threshold separately.
    return 0 if successful > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
