#!/usr/bin/env python3
"""Fast high-detail E.T.C. surface smoother for reliable CI production.

Uses a fine connected voxel surface, caps it to a dense 650k-class mesh, and
applies three volume-preserving Laplacian passes plus two micro-smoothing passes.
The result remains far denser and smoother than the earlier blocky output while
avoiding the very slow million-polygon corrective solver.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

BASE_PATH = Path(__file__).resolve().with_name("build_etc_super_smooth_blender.py")
spec = importlib.util.spec_from_file_location("etc_smooth_base_r5", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base builder: {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def fast_volume_smooth(obj, iterations: int, strength: float) -> None:
    before = len(obj.data.polygons)
    cap = 680_000 if before > 800_000 else before
    if before > cap:
        modifier = obj.modifiers.new("ETC_HIGH_DETAIL_CAP", "DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        modifier.ratio = cap / float(before)
        modifier.use_collapse_triangulate = True
        base.apply_modifier(obj, modifier.name)
        base.log(f"Dense detail cap: {before:,} -> {len(obj.data.polygons):,} polygons")

    lap = obj.modifiers.new("ETC_FAST_VOLUME_SMOOTH", "LAPLACIANSMOOTH")
    lap.iterations = 3
    lap.lambda_factor = 0.085
    lap.lambda_border = 0.0
    lap.use_volume_preserve = True
    if hasattr(lap, "use_normalized"):
        lap.use_normalized = True
    base.apply_modifier(obj, lap.name)

    micro = obj.modifiers.new("ETC_FAST_MICRO_SMOOTH", "SMOOTH")
    micro.factor = 0.10
    micro.iterations = 2
    micro.use_x = True
    micro.use_y = True
    micro.use_z = True
    base.apply_modifier(obj, micro.name)
    base.log(f"Completed fast physical smoothing on {len(obj.data.polygons):,} polygons")

base.volume_preserving_smooth = fast_volume_smooth
raise SystemExit(base.main())
