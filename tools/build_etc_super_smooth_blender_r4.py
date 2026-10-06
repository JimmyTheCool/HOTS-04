#!/usr/bin/env python3
"""High-detail E.T.C. smoothing without memory-heavy subdivision.

The 0.26 mm connected remesh already contains roughly 1.4 million polygons.
This wrapper lightly caps that mesh before applying multiple physical,
volume-preserving smoothing passes directly to it. It removes voxel terraces
without expanding the model to more than five million polygons.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


BASE_PATH = Path(__file__).resolve().with_name("build_etc_super_smooth_blender.py")
spec = importlib.util.spec_from_file_location("etc_smooth_base_r4", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base builder: {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def direct_million_polygon_smooth(obj, iterations: int, strength: float) -> None:
    before = len(obj.data.polygons)
    # Keep enough geometry for resin-scale detail while leaving headroom for the
    # Laplacian solver. The previous 5.7-million-polygon subdivision was the only
    # operation that exceeded memory.
    cap = 1_100_000
    if before > cap:
        ratio = cap / float(before)
        modifier = obj.modifiers.new("ETC_PRE_SMOOTH_DETAIL_CAP", "DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        modifier.ratio = ratio
        modifier.use_collapse_triangulate = True
        base.apply_modifier(obj, modifier.name)
        base.log(f"Pre-smoothing detail cap: {before:,} -> {len(obj.data.polygons):,} polygons")

    lap = obj.modifiers.new("ETC_MULTI_PASS_VOLUME_SMOOTH", "LAPLACIANSMOOTH")
    lap.iterations = 7 if len(obj.data.polygons) > 750_000 else 6
    lap.lambda_factor = min(0.095, max(0.075, strength))
    lap.lambda_border = 0.0
    lap.use_volume_preserve = True
    if hasattr(lap, "use_normalized"):
        lap.use_normalized = True
    base.apply_modifier(obj, lap.name)

    corrective = obj.modifiers.new("ETC_SURFACE_CORRECTIVE_SMOOTH", "CORRECTIVE_SMOOTH")
    corrective.factor = 0.12
    corrective.iterations = 3
    if hasattr(corrective, "smooth_type"):
        corrective.smooth_type = "LENGTH_WEIGHTED"
    if hasattr(corrective, "use_only_smooth"):
        corrective.use_only_smooth = True
    base.apply_modifier(obj, corrective.name)

    # A final gentle standard pass blends any remaining tiny voxel terraces.
    final_smooth = obj.modifiers.new("ETC_FINAL_MICRO_SMOOTH", "SMOOTH")
    final_smooth.factor = 0.11
    final_smooth.iterations = 2
    final_smooth.use_x = True
    final_smooth.use_y = True
    final_smooth.use_z = True
    apply = base.apply_modifier
    apply(obj, final_smooth.name)
    base.log(f"Completed direct geometric smoothing on {len(obj.data.polygons):,} polygons")


base.volume_preserving_smooth = direct_million_polygon_smooth
raise SystemExit(base.main())
