#!/usr/bin/env python3
"""Memory-safe super-smooth E.T.C. builder.

This wrapper uses the proven connected voxel-union builder at a fine but practical
resolution, then physically subdivides the resulting surface before applying
volume-preserving smoothing. Subdivision removes visible voxel facets without the
extreme memory footprint of a 0.16 mm whole-model voxel grid.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


BASE_PATH = Path(__file__).resolve().with_name("build_etc_super_smooth_blender.py")
spec = importlib.util.spec_from_file_location("etc_smooth_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base builder: {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
_original_smooth = base.volume_preserving_smooth


def subdivided_volume_smooth(obj, iterations: int, strength: float) -> None:
    """Add real geometric density, then smooth while preserving overall volume."""
    before = len(obj.data.polygons)
    # A single Catmull-Clark level gives four times the geometric sampling while
    # avoiding the multi-million-voxel OpenVDB memory spike seen at 0.16 mm.
    subdivision = obj.modifiers.new("ETC_PHYSICAL_SUBDIVISION", "SUBSURF")
    subdivision.subdivision_type = "CATMULL_CLARK"
    subdivision.levels = 1
    subdivision.render_levels = 1
    subdivision.quality = 3
    if hasattr(subdivision, "use_creases"):
        subdivision.use_creases = True
    if hasattr(subdivision, "boundary_smooth"):
        subdivision.boundary_smooth = "PRESERVE_CORNERS"
    base.apply_modifier(obj, subdivision.name)
    after = len(obj.data.polygons)
    base.log(f"Physical Catmull-Clark subdivision: {before:,} -> {after:,} polygons")

    # Use slightly gentler smoothing after subdivision so sculpted edges and the
    # guitar silhouette remain identifiable while small voxel terraces disappear.
    _original_smooth(
        obj,
        iterations=max(3, iterations),
        strength=min(0.065, strength),
    )


base.volume_preserving_smooth = subdivided_volume_smooth
raise SystemExit(base.main())
