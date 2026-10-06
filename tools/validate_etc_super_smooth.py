#!/usr/bin/env python3
"""Validate the high-detail and print-ready smooth E.T.C. STL variants."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [geometry for geometry in loaded.geometry.values()]
        if not geometries:
            raise RuntimeError(f"No geometry in {path}")
        return trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise RuntimeError(f"Unsupported geometry type for {path}: {type(loaded)!r}")
    return loaded


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(path: Path, minimum_faces: int, maximum_faces: int) -> dict[str, Any]:
    mesh = load_mesh(path)
    finite = bool(np.isfinite(mesh.vertices).all())
    extents = np.asarray(mesh.extents, dtype=float)
    adjacency_angles = np.asarray(mesh.face_adjacency_angles, dtype=float)
    adjacency_angles = adjacency_angles[np.isfinite(adjacency_angles)]
    if len(adjacency_angles):
        smoothness = {
            "median_dihedral_degrees": float(np.degrees(np.percentile(adjacency_angles, 50))),
            "p90_dihedral_degrees": float(np.degrees(np.percentile(adjacency_angles, 90))),
            "p95_dihedral_degrees": float(np.degrees(np.percentile(adjacency_angles, 95))),
        }
    else:
        smoothness = {
            "median_dihedral_degrees": None,
            "p90_dihedral_degrees": None,
            "p95_dihedral_degrees": None,
        }
    minimum_z = float(mesh.bounds[0][2])
    bottom_band = max(0.08, float(extents.max()) * 0.0008)
    bottom_vertices = mesh.vertices[mesh.vertices[:, 2] <= minimum_z + bottom_band]
    bottom_z_span = (
        float(np.ptp(bottom_vertices[:, 2])) if len(bottom_vertices) else None
    )
    row = {
        "file": path.name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "finite_coordinates": finite,
        "positive_volume": bool(np.isfinite(mesh.volume) and abs(float(mesh.volume)) > 1.0),
        "volume_mm3": float(abs(mesh.volume)),
        "connected_bodies": int(mesh.body_count),
        "dimensions_mm": [float(value) for value in extents],
        "minimum_z_mm": minimum_z,
        "bottom_vertex_count": int(len(bottom_vertices)),
        "bottom_z_span_mm": bottom_z_span,
        "smoothness": smoothness,
    }
    row["checks"] = {
        "watertight": row["watertight"],
        "winding_consistent": row["winding_consistent"],
        "finite_coordinates": row["finite_coordinates"],
        "positive_volume": row["positive_volume"],
        "one_connected_body": row["connected_bodies"] == 1,
        "triangle_range": minimum_faces <= row["triangles"] <= maximum_faces,
        "dimension_range": bool(np.all(extents >= 2.0) and np.all(extents <= 350.0)),
        "grounded_at_zero": abs(minimum_z) <= 0.02,
        "flat_underside": bottom_z_span is not None and bottom_z_span <= 0.03,
    }
    row["passed"] = all(row["checks"].values())
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--high", required=True, type=Path)
    parser.add_argument("--print-ready", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    high = args.high.resolve()
    print_ready = args.print_ready.resolve()
    for path in (high, print_ready):
        if not path.is_file():
            raise FileNotFoundError(path)

    report = {
        "expected_files": 2,
        "files": [
            inspect(high, minimum_faces=150_000, maximum_faces=1_500_000),
            inspect(print_ready, minimum_faces=100_000, maximum_faces=850_000),
        ],
    }
    report["passed"] = all(item["passed"] for item in report["files"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
