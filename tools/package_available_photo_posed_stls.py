#!/usr/bin/env python3
"""Collect every retained, strictly valid HOTS-04 photo-posed STL artifact.

This script is intended for GitHub Actions. It deliberately excludes the older
non-posed/T-pose artifact family and packages only outputs whose artifact names
identify the genuine photo-pose pipeline.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import traceback
import urllib.request
import zipfile

import numpy as np
import trimesh


API_VERSION = "2022-11-28"
POSE_ARTIFACT_PATTERN = re.compile(
    r"(photo[ _-]*pose|photo[ _-]*posed|posed[ _-]*stl)", re.IGNORECASE
)


def log(message: str) -> None:
    print(f"[posed-packager] {message}", flush=True)


def api_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": API_VERSION,
        "User-Agent": "HOTS04-photo-pose-packager",
    }


def fetch_json(url: str, token: str) -> dict:
    request = urllib.request.Request(url, headers=api_headers(token))
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def list_artifacts(repository: str, token: str) -> list[dict]:
    records: list[dict] = []
    page = 1
    while True:
        data = fetch_json(
            f"https://api.github.com/repos/{repository}/actions/artifacts"
            f"?per_page=100&page={page}",
            token,
        )
        batch = list(data.get("artifacts", []))
        records.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return records


def download_artifact(
    repository: str,
    token: str,
    artifact: dict,
    destination_root: Path,
) -> dict:
    artifact_id = int(artifact["id"])
    name = str(artifact.get("name", artifact_id))
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_.")
    zip_path = destination_root / f"{artifact_id}_{safe_name}.zip"
    extract_dir = destination_root / f"artifact_{artifact_id}"
    extract_dir.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/actions/artifacts/{artifact_id}/zip",
        headers=api_headers(token),
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response, zip_path.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
        stls = list(extract_dir.rglob("*.stl"))
        log(f"Downloaded {name}: {len(stls)} STL file(s)")
        return {
            "id": artifact_id,
            "name": name,
            "created_at": artifact.get("created_at"),
            "updated_at": artifact.get("updated_at"),
            "size_in_bytes": artifact.get("size_in_bytes"),
            "workflow_run": artifact.get("workflow_run"),
            "extract_directory": str(extract_dir),
            "stl_count": len(stls),
            "download_status": "success",
        }
    except Exception as exc:
        log(f"WARNING: {name} could not be downloaded: {exc}")
        return {
            "id": artifact_id,
            "name": name,
            "created_at": artifact.get("created_at"),
            "extract_directory": str(extract_dir),
            "stl_count": 0,
            "download_status": "failed",
            "error": str(exc),
        }
    finally:
        zip_path.unlink(missing_ok=True)


def safe_hero_name(stem: str) -> str:
    value = re.sub(
        r"(?i)(?:_print_ready|_watertight|_robust|_high_detail|_posed)$",
        "",
        stem,
    )
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_.")
    return value or "Hero"


def variant_rank(path: Path) -> tuple[int, str]:
    lower = path.as_posix().casefold()
    stem = path.stem.casefold()
    if "/robust/" in lower or "robust" in stem:
        return 500, "robust"
    if "print_ready" in lower or "print-ready" in lower or "/01_print_ready" in lower:
        return 450, "print-ready"
    if "/highdetail/" in lower or "/high_detail/" in lower:
        return 300, "high-detail"
    if "/initial/" in lower:
        return 200, "initial"
    if "/raw/" in lower:
        return 100, "raw"
    return 150, "other"


def validate_stl(path: Path) -> dict:
    result: dict = {
        "source_file": path.name,
        "source_path": str(path),
        "bytes": path.stat().st_size,
    }
    try:
        loaded = trimesh.load(path, force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            geometries = list(loaded.geometry.values())
            if not geometries:
                raise ValueError("STL scene contained no geometry")
            mesh = trimesh.util.concatenate(geometries)
        else:
            mesh = loaded

        extents = np.asarray(mesh.extents, dtype=float)
        finite = bool(np.isfinite(mesh.vertices).all() and np.isfinite(extents).all())
        vertices = int(len(mesh.vertices))
        triangles = int(len(mesh.faces))
        volume = float(abs(mesh.volume)) if finite else 0.0
        watertight = bool(mesh.is_watertight)
        winding = bool(mesh.is_winding_consistent)
        connected_bodies = int(mesh.body_count)
        passed = bool(
            finite
            and vertices >= 50
            and 100 <= triangles <= 750_000
            and extents.shape == (3,)
            and np.all(extents > 0.1)
            and np.all(extents < 500.0)
            and volume > 1.0
            and watertight
            and winding
            and connected_bodies == 1
        )
        result.update(
            {
                "passed": passed,
                "vertices": vertices,
                "triangles": triangles,
                "watertight": watertight,
                "winding_consistent": winding,
                "finite_coordinates": finite,
                "volume_mm3": volume,
                "x_mm": float(extents[0]),
                "y_mm": float(extents[1]),
                "z_mm": float(extents[2]),
                "connected_bodies": connected_bodies,
            }
        )
    except Exception as exc:
        result.update(
            {
                "passed": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    return result


def write_github_output(values: dict[str, str]) -> None:
    target = os.environ.get("GITHUB_OUTPUT")
    if not target:
        return
    with open(target, "a", encoding="utf-8") as handle:
        for key, value in values.items():
            clean = str(value).replace("\n", " ").replace("\r", " ")
            handle.write(f"{key}={clean}\n")


def create_archive(source: Path, target: Path) -> None:
    with zipfile.ZipFile(
        target,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=7,
        allowZip64=True,
    ) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, source.name / path.relative_to(source))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--token", default=os.environ.get("GH_TOKEN"))
    parser.add_argument("--working-directory", type=Path, required=True)
    parser.add_argument("--package-directory", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()

    if not args.repository:
        raise SystemExit("GitHub repository was not supplied")
    if not args.token:
        raise SystemExit("GitHub token was not supplied")

    working = args.working_directory.resolve()
    package = args.package_directory.resolve()
    archive = args.archive.resolve()
    shutil.rmtree(working, ignore_errors=True)
    shutil.rmtree(package, ignore_errors=True)
    working.mkdir(parents=True, exist_ok=True)
    (package / "STL").mkdir(parents=True, exist_ok=True)
    (package / "Reports").mkdir(parents=True, exist_ok=True)

    all_artifacts = list_artifacts(args.repository, args.token)
    matched = [
        item
        for item in all_artifacts
        if not item.get("expired", False)
        and POSE_ARTIFACT_PATTERN.search(str(item.get("name", "")))
    ]
    matched.sort(key=lambda item: str(item.get("created_at", "")))
    log(f"Matched {len(matched)} photo-pose artifact(s) from {len(all_artifacts)} retained artifact records")

    inventory = [
        download_artifact(args.repository, args.token, artifact, working)
        for artifact in matched
    ]
    inventory_document = {
        "repository": args.repository,
        "matched_artifact_count": len(inventory),
        "matched_artifacts": inventory,
    }
    inventory_path = package / "Reports" / "artifact_inventory.json"
    inventory_path.write_text(json.dumps(inventory_document, indent=2), encoding="utf-8")

    successful = [item for item in inventory if item.get("download_status") == "success"]
    if not any(int(item.get("stl_count", 0)) > 0 for item in successful):
        raise SystemExit("No retained photo-posed artifact contained an STL file")

    candidates: dict[str, list[dict]] = {}
    for artifact in successful:
        root = Path(artifact["extract_directory"])
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.casefold() != ".stl":
                continue
            hero = safe_hero_name(path.stem)
            rank, variant = variant_rank(path)
            candidates.setdefault(hero.casefold(), []).append(
                {
                    "hero": hero,
                    "path": path,
                    "rank": rank,
                    "variant": variant,
                    "artifact": artifact,
                }
            )

    selected_rows: list[dict] = []
    rejected_rows: list[dict] = []
    for folded_hero, hero_candidates in sorted(candidates.items()):
        hero_candidates.sort(
            key=lambda item: (
                int(item["rank"]),
                str(item["artifact"].get("created_at") or ""),
                int(item["path"].stat().st_size),
            ),
            reverse=True,
        )
        chosen: tuple[dict, dict] | None = None
        for candidate in hero_candidates:
            validation = validate_stl(candidate["path"])
            validation["variant"] = candidate["variant"]
            if validation["passed"]:
                chosen = candidate, validation
                break
            rejected_rows.append(
                {
                    "hero": candidate["hero"],
                    "artifact_id": candidate["artifact"].get("id"),
                    "artifact_name": candidate["artifact"].get("name"),
                    **validation,
                }
            )

        if chosen is None:
            log(f"No strict printable candidate passed for {hero_candidates[0]['hero']}")
            continue

        candidate, validation = chosen
        destination_name = f"{candidate['hero']}.stl"
        destination = package / "STL" / destination_name
        shutil.copy2(candidate["path"], destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        artifact_root = Path(candidate["artifact"]["extract_directory"])
        selected_rows.append(
            {
                "hero": candidate["hero"],
                "file": destination_name,
                "variant": candidate["variant"],
                "artifact_id": candidate["artifact"].get("id"),
                "artifact_name": candidate["artifact"].get("name"),
                "artifact_created_at": candidate["artifact"].get("created_at"),
                "source_path_inside_artifact": str(candidate["path"].relative_to(artifact_root)),
                "bytes": destination.stat().st_size,
                "sha256": digest,
                "vertices": validation.get("vertices"),
                "triangles": validation.get("triangles"),
                "watertight": validation.get("watertight"),
                "winding_consistent": validation.get("winding_consistent"),
                "finite_coordinates": validation.get("finite_coordinates"),
                "volume_mm3": validation.get("volume_mm3"),
                "x_mm": validation.get("x_mm"),
                "y_mm": validation.get("y_mm"),
                "z_mm": validation.get("z_mm"),
                "connected_bodies": validation.get("connected_bodies"),
            }
        )
        log(f"Selected {candidate['hero']}: {candidate['variant']} from {candidate['artifact'].get('name')}")

    if not selected_rows:
        raise SystemExit("No photo-posed STL passed strict printable validation")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository": args.repository,
        "description": "Strictly validated photo-posed HOTS-04 STL files recoverable at packaging time.",
        "hero_count": len(selected_rows),
        "heroes": [row["hero"] for row in selected_rows],
        "selection_policy": [
            "Only retained action artifacts identifying the genuine photo-pose pipeline were considered.",
            "The earlier generic V2/T-pose artifact family was excluded.",
            "Robust and print-ready variants were preferred.",
            "Each included STL passed watertightness, winding, finite-coordinate, volume, dimension, and single-connected-body checks.",
        ],
        "files": selected_rows,
    }
    reports = package / "Reports"
    manifest_json = reports / "manifest.json"
    manifest_csv = reports / "manifest.csv"
    rejected_json = reports / "rejected_candidates.json"
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rejected_json.write_text(json.dumps(rejected_rows, indent=2), encoding="utf-8")

    with manifest_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0].keys()))
        writer.writeheader()
        writer.writerows(selected_rows)

    names = "\n".join(f"- {row['hero']} ({row['variant']})" for row in selected_rows)
    readme = (
        "HOTS-04 PHOTO-POSED STL FILES AVAILABLE SO FAR\n\n"
        f"This archive contains {len(selected_rows)} recoverable character STL file(s) "
        "produced by the genuine photo-pose pipeline. It does not contain the earlier "
        "generic T-pose collection.\n\n"
        "The pipeline used the selected HOTS game mesh for proportions and skin geometry, "
        "searched compatible skeletal animation frames, compared projected silhouettes "
        "against all four supplied JPG views, froze the best matching pose, and repaired "
        "the model for printing.\n\n"
        f"INCLUDED\n{names}\n\n"
        "Every included STL is watertight, winding-consistent, finite, positive-volume, "
        "and one connected body. The Reports folder records provenance, dimensions, "
        "triangle counts, checksums, and rejected alternatives.\n\n"
        "STL stores geometry only. Colours, textures, glow, transparency, particles, and "
        "normal-map details are not embedded. Inspect orientation and supports in your "
        "slicer before printing. These fan models are for personal, non-commercial use.\n"
    )
    (package / "README_FIRST.txt").write_text(readme, encoding="utf-8")

    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.unlink(missing_ok=True)
    create_archive(package, archive)
    archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum_path = archive.with_name(archive.name + ".sha256")
    checksum_path.write_text(f"{archive_digest}  {archive.name}\n", encoding="utf-8")

    result = {
        "hero_count": len(selected_rows),
        "heroes": [row["hero"] for row in selected_rows],
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_digest,
    }
    log(json.dumps(result, indent=2))
    write_github_output(
        {
            "hero_count": str(len(selected_rows)),
            "heroes": ", ".join(row["hero"] for row in selected_rows),
            "archive": str(archive),
            "checksum": str(checksum_path),
            "manifest_json": str(manifest_json),
            "manifest_csv": str(manifest_csv),
            "archive_sha256": archive_digest,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
