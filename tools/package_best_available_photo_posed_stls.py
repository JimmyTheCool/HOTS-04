#!/usr/bin/env python3
"""Package the best available photo-posed STL for every retained HOTS-04 hero.

Unlike the strict packager, this script does not discard a posed model merely because
it still needs manifold or component repair. It prefers robust/print-ready variants,
requires the mesh to be parseable and geometrically meaningful, and records every
remaining defect in the package manifest.
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
import traceback
from urllib.parse import urlsplit
import urllib.request
import zipfile

import numpy as np
import trimesh


POSE_PATTERN = re.compile(r"photo[ _-]*posed", re.IGNORECASE)
ARTIFACT_HERO_PATTERN = re.compile(r"^HOTS04-photo-posed-v\d+-(.+)$", re.IGNORECASE)


class StripSensitiveHeadersOnCrossHostRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        if urlsplit(req.full_url).netloc.casefold() != urlsplit(newurl).netloc.casefold():
            for name in ("Authorization", "X-GitHub-Api-Version", "Accept"):
                redirected.remove_header(name)
        return redirected


urllib.request.install_opener(
    urllib.request.build_opener(StripSensitiveHeadersOnCrossHostRedirect())
)


def log(message: str) -> None:
    print(f"[best-available-posed] {message}", flush=True)


def headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "HOTS04-best-available-photo-pose-packager",
    }


def fetch_json(url: str, token: str) -> dict:
    request = urllib.request.Request(url, headers=headers(token))
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def list_artifacts(repository: str, token: str) -> list[dict]:
    result: list[dict] = []
    page = 1
    while True:
        data = fetch_json(
            f"https://api.github.com/repos/{repository}/actions/artifacts"
            f"?per_page=100&page={page}",
            token,
        )
        batch = list(data.get("artifacts", []))
        result.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return result


def artifact_hero(name: str) -> str:
    match = ARTIFACT_HERO_PATTERN.match(name)
    value = match.group(1) if match else name
    value = value.replace("Lt_Morales", "Lt_Morales").replace("Zul_jin", "Zul_jin")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_.") or "Hero"


def download_artifact(repository: str, token: str, artifact: dict, root: Path) -> dict:
    artifact_id = int(artifact["id"])
    name = str(artifact["name"])
    hero = artifact_hero(name)
    archive_path = root / f"{artifact_id}_{hero}.zip"
    extract_path = root / f"artifact_{artifact_id}_{hero}"
    extract_path.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/actions/artifacts/{artifact_id}/zip",
        headers=headers(token),
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response, archive_path.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_path)
        stls = sorted(extract_path.rglob("*.stl"))
        log(f"Downloaded {name}: {len(stls)} STL files")
        return {
            "id": artifact_id,
            "name": name,
            "hero": hero,
            "created_at": artifact.get("created_at"),
            "updated_at": artifact.get("updated_at"),
            "size_in_bytes": artifact.get("size_in_bytes"),
            "extract_directory": str(extract_path),
            "stl_count": len(stls),
            "download_status": "success",
        }
    except Exception as exc:
        log(f"WARNING: failed to download {name}: {exc}")
        return {
            "id": artifact_id,
            "name": name,
            "hero": hero,
            "created_at": artifact.get("created_at"),
            "extract_directory": str(extract_path),
            "stl_count": 0,
            "download_status": "failed",
            "error": str(exc),
        }
    finally:
        archive_path.unlink(missing_ok=True)


def rank_variant(path: Path) -> tuple[int, str]:
    lower = path.as_posix().casefold()
    stem = path.stem.casefold()
    if "/robust/" in lower or "robust" in stem:
        return 600, "robust"
    if "/printready/" in lower or "/print_ready/" in lower or "print_ready" in lower:
        return 550, "print-ready"
    if "/repaired/" in lower or "repaired" in stem:
        return 500, "repaired"
    if "/highdetail/" in lower or "/high_detail/" in lower:
        return 400, "high-detail"
    if "/initial/" in lower:
        return 300, "initial"
    if "/raw/" in lower:
        return 200, "raw"
    return 100, "other"


def inspect_stl(path: Path) -> dict:
    row: dict = {
        "source_file": path.name,
        "source_path": str(path),
        "bytes": path.stat().st_size,
    }
    try:
        loaded = trimesh.load(path, force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            geometries = list(loaded.geometry.values())
            if not geometries:
                raise ValueError("STL scene contains no geometry")
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
        bodies = int(mesh.body_count)
        usable = bool(
            finite
            and vertices >= 50
            and triangles >= 100
            and extents.shape == (3,)
            and np.all(extents > 0.1)
            and np.all(extents < 1000.0)
            and volume > 1.0
        )
        strict = bool(usable and watertight and winding and bodies == 1)
        warnings: list[str] = []
        if not watertight:
            warnings.append("not watertight")
        if not winding:
            warnings.append("inconsistent winding")
        if bodies != 1:
            warnings.append(f"{bodies} connected bodies")
        if triangles > 750_000:
            warnings.append(f"high triangle count: {triangles}")
        row.update(
            {
                "usable_as_generated": usable,
                "strict_print_ready": strict,
                "vertices": vertices,
                "triangles": triangles,
                "watertight": watertight,
                "winding_consistent": winding,
                "finite_coordinates": finite,
                "volume_mm3": volume,
                "x_mm": float(extents[0]),
                "y_mm": float(extents[1]),
                "z_mm": float(extents[2]),
                "connected_bodies": bodies,
                "warnings": warnings,
            }
        )
    except Exception as exc:
        row.update(
            {
                "usable_as_generated": False,
                "strict_print_ready": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    return row


def copy_selected_reports(artifact_root: Path, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    preferred_names = {
        "photo_pose_quality_summary.json",
        "pose_match.json",
        "job_config.json",
        "source_resolution.json",
        "robust_validation.json",
        "high_detail_validation.json",
        "reference_masks.json",
        "HERO_COMPLETE.txt",
    }
    for path in artifact_root.rglob("*"):
        if not path.is_file() or path.name not in preferred_names:
            continue
        target = destination / path.name
        if target.exists():
            target = destination / f"{path.parent.name}_{path.name}"
        shutil.copy2(path, target)
        copied.append(target.name)
    return sorted(copied)


def write_github_output(values: dict[str, str]) -> None:
    target = os.environ.get("GITHUB_OUTPUT")
    if not target:
        return
    with open(target, "a", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}={str(value).replace(chr(10), ' ').replace(chr(13), ' ')}\n")


def create_zip(source: Path, target: Path) -> None:
    target.unlink(missing_ok=True)
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
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    if not args.repository or not args.token:
        raise SystemExit("GitHub repository and token are required")

    work = args.work.resolve()
    package = args.package.resolve()
    archive = args.archive.resolve()
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(package, ignore_errors=True)
    work.mkdir(parents=True)
    best_dir = package / "01_Posed_Best_Available"
    reports_dir = package / "Reports"
    per_hero_reports = reports_dir / "Per_Hero"
    best_dir.mkdir(parents=True)
    per_hero_reports.mkdir(parents=True)

    artifacts = list_artifacts(args.repository, args.token)
    posed = [
        artifact
        for artifact in artifacts
        if not artifact.get("expired", False)
        and POSE_PATTERN.search(str(artifact.get("name", "")))
        and re.search(r"photo-posed-v5-", str(artifact.get("name", "")), re.IGNORECASE)
    ]
    posed.sort(key=lambda item: str(item.get("created_at", "")))
    log(f"Found {len(posed)} retained V5 photo-posed artifacts")
    inventory = [download_artifact(args.repository, args.token, item, work) for item in posed]
    (reports_dir / "artifact_inventory.json").write_text(
        json.dumps({"artifacts": inventory}, indent=2), encoding="utf-8"
    )

    rows: list[dict] = []
    rejected: list[dict] = []
    for artifact in inventory:
        if artifact.get("download_status") != "success":
            continue
        root = Path(artifact["extract_directory"])
        candidates: list[tuple[int, str, Path, dict]] = []
        for path in root.rglob("*.stl"):
            rank, variant = rank_variant(path)
            inspection = inspect_stl(path)
            if inspection.get("usable_as_generated"):
                candidates.append((rank, variant, path, inspection))
            else:
                rejected.append(
                    {
                        "hero": artifact["hero"],
                        "artifact_name": artifact["name"],
                        "variant": variant,
                        **inspection,
                    }
                )
        if not candidates:
            log(f"No parseable posed STL available for {artifact['hero']}")
            continue
        candidates.sort(
            key=lambda value: (
                bool(value[3].get("strict_print_ready")),
                value[0],
                int(value[3].get("triangles", 0)),
            ),
            reverse=True,
        )
        rank, variant, source, inspection = candidates[0]
        destination = best_dir / f"{artifact['hero']}.stl"
        shutil.copy2(source, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        report_files = copy_selected_reports(root, per_hero_reports / artifact["hero"])
        warnings = list(inspection.get("warnings", []))
        row = {
            "hero": artifact["hero"],
            "file": destination.name,
            "variant": variant,
            "strict_print_ready": bool(inspection.get("strict_print_ready")),
            "requires_slicer_or_mesh_repair": not bool(inspection.get("strict_print_ready")),
            "warnings": "; ".join(warnings) if warnings else "",
            "artifact_id": artifact["id"],
            "artifact_name": artifact["name"],
            "artifact_created_at": artifact.get("created_at"),
            "source_path_inside_artifact": str(source.relative_to(root)),
            "bytes": destination.stat().st_size,
            "sha256": digest,
            "vertices": inspection.get("vertices"),
            "triangles": inspection.get("triangles"),
            "watertight": inspection.get("watertight"),
            "winding_consistent": inspection.get("winding_consistent"),
            "finite_coordinates": inspection.get("finite_coordinates"),
            "volume_mm3": inspection.get("volume_mm3"),
            "x_mm": inspection.get("x_mm"),
            "y_mm": inspection.get("y_mm"),
            "z_mm": inspection.get("z_mm"),
            "connected_bodies": inspection.get("connected_bodies"),
            "copied_report_files": "; ".join(report_files),
        }
        rows.append(row)
        log(
            f"Selected {artifact['hero']}: {variant}; strict={row['strict_print_ready']}; "
            f"warnings={row['warnings'] or 'none'}"
        )

    if not rows:
        raise SystemExit("No usable posed STL was recovered")

    rows.sort(key=lambda row: row["hero"].casefold())
    strict_count = sum(bool(row["strict_print_ready"]) for row in rows)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository": args.repository,
        "description": "Best available V5 photo-posed HOTS-04 models generated so far.",
        "hero_count": len(rows),
        "strict_print_ready_count": strict_count,
        "requires_repair_count": len(rows) - strict_count,
        "important_notice": (
            "These are the posed models currently available, supplied at the user's request. "
            "Models marked requires_slicer_or_mesh_repair did not pass the strict watertight, "
            "consistent-winding, one-connected-body gate and should be inspected or repaired before printing."
        ),
        "heroes": [row["hero"] for row in rows],
        "files": rows,
    }
    manifest_json = reports_dir / "manifest.json"
    manifest_csv = reports_dir / "manifest.csv"
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (reports_dir / "rejected_unusable_variants.json").write_text(
        json.dumps(rejected, indent=2), encoding="utf-8"
    )
    with manifest_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    hero_lines = "\n".join(
        f"- {row['hero']}: {row['variant']} — "
        + ("strict print-ready" if row["strict_print_ready"] else f"needs inspection/repair ({row['warnings']})")
        for row in rows
    )
    readme = f"""HOTS-04 PHOTO-POSED STL MODELS AVAILABLE SO FAR

This package contains {len(rows)} V5 photo-posed hero models recovered from completed
GitHub Actions build artifacts. The earlier generic T-pose collection is excluded.

The models use the selected game-mesh skin and dimensions. Their poses were chosen by
searching compatible in-game skeletal animation frames and comparing projected
silhouettes against the four supplied JPG views.

IMPORTANT PRINTING NOTICE
Only {strict_count} of {len(rows)} model(s) passed the strict watertight, consistent-
winding, one-connected-body gate. The remaining models are included because you asked
for everything usable that exists so far. Inspect and repair them in a slicer or mesh
repair program before printing. The manifest identifies every detected issue.

INCLUDED MODELS
{hero_lines}

FOLDERS
01_Posed_Best_Available — one selected posed STL per recovered hero
Reports/manifest.csv — concise validation and source information
Reports/manifest.json — complete machine-readable validation information
Reports/Per_Hero — available pose-selection and build reports

STL preserves geometry only. It does not preserve game textures, colours, glow,
transparency, particle effects, or normal-map detail. These fan models are intended
for personal, non-commercial use.
"""
    (package / "README_FIRST.txt").write_text(readme, encoding="utf-8")

    archive.parent.mkdir(parents=True, exist_ok=True)
    create_zip(package, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum = archive.with_name(archive.name + ".sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    log(
        json.dumps(
            {
                "hero_count": len(rows),
                "strict_print_ready_count": strict_count,
                "heroes": [row["hero"] for row in rows],
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": digest,
            },
            indent=2,
        )
    )
    write_github_output(
        {
            "hero_count": str(len(rows)),
            "strict_count": str(strict_count),
            "heroes": ", ".join(row["hero"] for row in rows),
            "archive": str(archive),
            "checksum": str(checksum),
            "manifest_json": str(manifest_json),
            "manifest_csv": str(manifest_csv),
            "archive_sha256": digest,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
