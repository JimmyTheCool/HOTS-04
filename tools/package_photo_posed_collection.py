#!/usr/bin/env python3
"""Assemble, validate, and document the HOTS-04 photo-posed STL collection.

The input folder is expected to contain one downloaded GitHub Actions artifact per
hero. Each artifact is produced by run_photo_posed_job_v3_2.py and contains
HighDetail, Robust, and Reports folders. The script refuses to package incomplete,
unposed, disconnected, non-watertight, or duplicated models.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import trimesh


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def one_per_parent(paths: list[Path], label: str) -> dict[Path, Path]:
    result: dict[Path, Path] = {}
    for path in paths:
        artifact_root = path
        while artifact_root.parent.name and artifact_root.parent.name != "":
            if artifact_root.parent.parent == artifact_root.parent:
                break
            artifact_root = artifact_root.parent
            if artifact_root.parent.name == "":
                break
        key = path.parents[1] if len(path.parents) > 1 else path.parent
        if key in result:
            raise RuntimeError(f"Duplicate {label} entry below {key}: {result[key]} and {path}")
        result[key] = path
    return result


def validate_stl(path: Path, max_triangles: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "error": "",
    }
    try:
        mesh = trimesh.load_mesh(path, force="mesh", process=False)
        extents = [float(value) for value in mesh.extents]
        row.update(
            {
                "vertices": int(len(mesh.vertices)),
                "triangles": int(len(mesh.faces)),
                "watertight": bool(mesh.is_watertight),
                "winding_consistent": bool(mesh.is_winding_consistent),
                "finite_coordinates": bool(np.isfinite(mesh.vertices).all()),
                "volume_mm3": float(abs(mesh.volume)),
                "x_mm": extents[0],
                "y_mm": extents[1],
                "z_mm": extents[2],
                "connected_bodies": int(mesh.body_count),
            }
        )
        row["passed"] = bool(
            row["watertight"]
            and row["winding_consistent"]
            and row["finite_coordinates"]
            and row["volume_mm3"] > 1.0
            and 100 <= row["triangles"] <= max_triangles
            and min(extents) >= 2.0
            and max(extents) <= 350.0
            and row["connected_bodies"] == 1
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        row.update(
            {
                "vertices": 0,
                "triangles": 0,
                "watertight": False,
                "winding_consistent": False,
                "finite_coordinates": False,
                "volume_mm3": 0.0,
                "x_mm": 0.0,
                "y_mm": 0.0,
                "z_mm": 0.0,
                "connected_bodies": 0,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return row


def find_single(root: Path, pattern: str, label: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {label} below {root}, found {len(matches)}: {matches}")
    return matches[0]


def safe_folder_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value).strip("_")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected", type=int, default=18)
    args = parser.parse_args()

    parts = args.parts.resolve()
    output = args.output.resolve()
    plan = load_json(args.plan.resolve())
    expected_heroes = list(plan["heroes"].keys())
    if len(expected_heroes) != args.expected:
        raise RuntimeError(
            f"Selected-skin plan contains {len(expected_heroes)} heroes, expected {args.expected}"
        )

    if output.exists():
        shutil.rmtree(output)
    print_ready = output / "01_Print_Ready_Photo_Posed"
    high_detail = output / "02_High_Detail_Photo_Posed"
    proofs = output / "03_Pose_Proofs"
    reports_out = output / "Reports"
    for directory in (print_ready, high_detail, proofs, reports_out):
        directory.mkdir(parents=True, exist_ok=True)

    artifact_roots = sorted(path for path in parts.iterdir() if path.is_dir())
    if len(artifact_roots) != args.expected:
        raise RuntimeError(f"Found {len(artifact_roots)} hero artifacts, expected {args.expected}")

    manifest: list[dict[str, Any]] = []
    robust_validation: list[dict[str, Any]] = []
    high_validation: list[dict[str, Any]] = []
    heroes_found: list[str] = []

    for artifact_root in artifact_roots:
        quality_path = find_single(
            artifact_root,
            "Reports/*/photo_pose_quality_summary.json",
            "photo-pose quality summary",
        )
        quality = load_json(quality_path)
        hero = str(quality.get("hero", "")).strip()
        safe = str(quality.get("safe", "")).strip()
        if not hero or not safe:
            raise RuntimeError(f"Missing hero/safe name in {quality_path}")
        if hero in heroes_found:
            raise RuntimeError(f"Duplicate hero artifact: {hero}")
        heroes_found.append(hero)

        required_quality = {
            "selected_animation": bool(str(quality.get("selected_animation", "")).strip()),
            "selected_frame": quality.get("selected_frame") is not None,
            "coarse_frames": int(quality.get("coarse_frames_tested", 0)) >= 2,
            "refined_frames": int(quality.get("refined_frames_tested", 0)) >= 1,
            "high_detail_validation": quality.get("high_detail_validation_passed") is True,
            "robust_validation": quality.get("robust_validation_passed") is True,
        }
        if not all(required_quality.values()):
            raise RuntimeError(f"Pose/quality gate failed for {hero}: {required_quality}")

        robust_source = find_single(artifact_root, "Robust/*.stl", "robust STL")
        high_source = find_single(artifact_root, "HighDetail/*.stl", "high-detail STL")
        proof_source = find_single(artifact_root, "Reports/*/*_pose_proof.jpg", "pose proof")
        report_source = quality_path.parent

        robust_target = print_ready / f"{safe}.stl"
        high_target = high_detail / f"{safe}.stl"
        proof_target = proofs / f"{safe}_pose_proof.jpg"
        report_target = reports_out / safe
        shutil.copy2(robust_source, robust_target)
        shutil.copy2(high_source, high_target)
        shutil.copy2(proof_source, proof_target)
        shutil.copytree(report_source, report_target, dirs_exist_ok=True)

        robust_row = validate_stl(robust_target, max_triangles=500000)
        high_row = validate_stl(high_target, max_triangles=600000)
        robust_row["hero"] = hero
        robust_row["variant"] = "print-ready"
        high_row["hero"] = hero
        high_row["variant"] = "high-detail"
        robust_validation.append(robust_row)
        high_validation.append(high_row)
        if not robust_row["passed"] or not high_row["passed"]:
            raise RuntimeError(
                f"Independent STL validation failed for {hero}: "
                f"print-ready={robust_row['passed']}, high-detail={high_row['passed']}"
            )

        manifest.append(
            {
                "hero": hero,
                "safe_name": safe,
                "skin_key": quality.get("skin_key"),
                "model_path": quality.get("model_path"),
                "selected_animation": quality.get("selected_animation"),
                "selected_frame": quality.get("selected_frame"),
                "pose_score": quality.get("pose_score"),
                "rest_pose_baseline_score": quality.get("rest_pose_baseline_score"),
                "improvement_over_rest_pose": quality.get("improvement_over_rest_pose"),
                "coarse_frames_tested": quality.get("coarse_frames_tested"),
                "refined_frames_tested": quality.get("refined_frames_tested"),
                "high_detail_method": quality.get("high_detail_method"),
                "robust_method": quality.get("robust_method"),
                "print_ready_file": robust_target.name,
                "print_ready_sha256": robust_row["sha256"],
                "print_ready_triangles": robust_row["triangles"],
                "high_detail_file": high_target.name,
                "high_detail_sha256": high_row["sha256"],
                "high_detail_triangles": high_row["triangles"],
                "pose_proof": proof_target.name,
            }
        )

    missing = [hero for hero in expected_heroes if hero not in heroes_found]
    unexpected = [hero for hero in heroes_found if hero not in expected_heroes]
    if missing or unexpected:
        raise RuntimeError(f"Hero set mismatch. Missing={missing}; unexpected={unexpected}")

    manifest.sort(key=lambda row: row["hero"].casefold())
    robust_validation.sort(key=lambda row: row["hero"].casefold())
    high_validation.sort(key=lambda row: row["hero"].casefold())

    validation = {
        "expected_heroes": args.expected,
        "found_heroes": len(manifest),
        "passed": True,
        "requirements": {
            "photo_pose_animation_selected": True,
            "coarse_animation_frames_tested": ">=2",
            "refined_animation_frames_tested": ">=1",
            "watertight": True,
            "winding_consistent": True,
            "finite_coordinates": True,
            "positive_volume": True,
            "connected_bodies": 1,
            "dimension_range_mm": [2.0, 350.0],
            "print_ready_max_triangles": 500000,
            "high_detail_max_triangles": 600000,
        },
        "print_ready": robust_validation,
        "high_detail": high_validation,
    }
    (reports_out / "final_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    (reports_out / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    with (reports_out / "package_manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    for name, rows in (
        ("print_ready_validation.csv", robust_validation),
        ("high_detail_validation.csv", high_validation),
    ):
        with (reports_out / name).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    readme = f"""HOTS-04 PHOTO-POSED PRINTABLE STL COLLECTION
================================================

This package contains {len(manifest)} Heroes of the Storm character models.

01_Print_Ready_Photo_Posed
    Conservative, single-body, watertight models intended for slicing.

02_High_Detail_Photo_Posed
    Higher-detail, independently watertight versions. These have more triangles
    and may require more slicer memory.

03_Pose_Proofs
    Side-by-side proof sheets showing the four supplied JPG references and the
    silhouettes used to select the closest available game animation frame.

Reports
    Per-character pose-search records, selected animation/frame, source mesh,
    skin selection, repair method, silhouette retention, validation, and hashes.

Every included STL passed automated checks for watertightness, consistent winding,
finite coordinates, positive volume, physical dimensions, triangle count, and one
connected body. The pose search used the selected game skin mesh and compatible M3A
animation banks, compared many frames against all four supplied JPG silhouettes,
and permanently baked the best-scoring frame before STL repair.

STL does not preserve textures, colours, transparency, glow, particles, or normal
maps. Those appearance details must be reproduced through physical painting.
Intended for personal, non-commercial fan use.
"""
    (output / "README_FIRST.txt").write_text(readme, encoding="utf-8")
    print(json.dumps(validation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
