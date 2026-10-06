#!/usr/bin/env python3
"""Create a smooth, detail-preserving E.T.C. printable STL.

The input is the existing photograph-pose STL. Unlike voxel remeshing, this keeps
its topology and dimensions, pins the flat base, and moves only existing vertices
with shrinkage-compensated Taubin smoothing. This specifically targets little
voxel terraces and faceted patches without creating a new block grid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import trimesh
from trimesh import smoothing


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=True)
    if isinstance(loaded, trimesh.Scene):
        geometry = list(loaded.geometry.values())
        if not geometry:
            raise RuntimeError(f"No geometry in {path}")
        loaded = trimesh.util.concatenate(geometry)
    if not isinstance(loaded, trimesh.Trimesh):
        raise RuntimeError(f"Unsupported geometry: {type(loaded)!r}")
    loaded.remove_unreferenced_vertices()
    return loaded


def cleanup(mesh: trimesh.Trimesh) -> None:
    try:
        mesh.update_faces(mesh.nondegenerate_faces(height=1e-8))
    except TypeError:
        mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    try:
        trimesh.repair.fix_normals(mesh, multibody=True)
    except TypeError:
        trimesh.repair.fix_normals(mesh)


def analyse(mesh: trimesh.Trimesh) -> dict:
    angles = np.asarray(mesh.face_adjacency_angles, dtype=float)
    angles = angles[np.isfinite(angles)]
    return {
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "connected_bodies": int(mesh.body_count),
        "volume_mm3": float(abs(mesh.volume)),
        "dimensions_mm": [float(value) for value in mesh.extents],
        "bounds_mm": [[float(v) for v in row] for row in mesh.bounds],
        "median_dihedral_degrees": float(np.degrees(np.percentile(angles, 50))) if len(angles) else None,
        "p90_dihedral_degrees": float(np.degrees(np.percentile(angles, 90))) if len(angles) else None,
        "p95_dihedral_degrees": float(np.degrees(np.percentile(angles, 95))) if len(angles) else None,
    }


def smooth_variant(source: trimesh.Trimesh, iterations: int, lamb: float,
                   nu: float, output: Path) -> tuple[trimesh.Trimesh, dict]:
    mesh = source.copy()
    minimum_z = float(mesh.bounds[0, 2])
    base_band = max(0.18, float(mesh.extents.max()) * 0.0015)
    pinned = np.flatnonzero(mesh.vertices[:, 2] <= minimum_z + base_band)
    pinned_positions = mesh.vertices[pinned].copy()
    initial_centroid = mesh.centroid.copy()
    initial_volume = abs(float(mesh.volume))

    operator = smoothing.laplacian_calculation(
        mesh,
        equal_weight=True,
        pinned_vertices=pinned,
    )
    smoothing.filter_taubin(
        mesh,
        lamb=lamb,
        nu=nu,
        iterations=iterations,
        laplacian_operator=operator,
    )

    # Restore the entire base band exactly, not merely its Z coordinate. This
    # prevents rounded or drifting base edges and guarantees a flat print plane.
    mesh.vertices[pinned] = pinned_positions
    mesh.vertices[:, 2] -= float(mesh.vertices[:, 2].min())
    mesh.vertices[pinned, 2] = 0.0
    cleanup(mesh)

    output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output, file_type="stl")
    if not output.is_file() or output.stat().st_size < 1024:
        raise RuntimeError(f"Failed to export {output}")

    result = {
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "iterations": iterations,
        "lambda": lamb,
        "nu": nu,
        "pinned_base_vertices": int(len(pinned)),
        "base_band_mm": base_band,
        "centroid_shift_mm": float(np.linalg.norm(mesh.centroid - initial_centroid)),
        "volume_retention": (abs(float(mesh.volume)) / initial_volume if initial_volume > 0 else None),
        "analysis": analyse(mesh),
    }
    return mesh, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    source_path = args.source.resolve()
    output_dir = args.output_dir.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = load_mesh(source_path)
    source_analysis = analyse(source)
    if not source_analysis["watertight"]:
        raise RuntimeError("Source ETC STL is not watertight; direct topology-preserving smoothing is unsafe")
    if source_analysis["connected_bodies"] != 1:
        raise RuntimeError(f"Source ETC STL has {source_analysis['connected_bodies']} bodies; expected one")

    high_path = output_dir / "ETC_High_Detail_Detail_Preserving_Smooth.stl"
    print_path = output_dir / "ETC_Print_Ready_Detail_Preserving_Smooth.stl"
    _, high = smooth_variant(source, iterations=14, lamb=0.46, nu=0.53, output=high_path)
    _, printable = smooth_variant(source, iterations=20, lamb=0.47, nu=0.54, output=print_path)

    report = {
        "method": "topology-preserving Taubin smoothing with pinned flat base",
        "source": str(source_path),
        "source_bytes": source_path.stat().st_size,
        "source_sha256": sha256(source_path),
        "source_analysis": source_analysis,
        "high_detail": high,
        "print_ready": printable,
    }
    report["passed"] = all(
        variant["analysis"]["watertight"]
        and variant["analysis"]["winding_consistent"]
        and variant["analysis"]["connected_bodies"] == 1
        and variant["analysis"]["volume_mm3"] > 1.0
        and abs(variant["analysis"]["bounds_mm"][0][2]) <= 0.02
        for variant in (high, printable)
    )
    (output_dir / "taubin_smoothing_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
