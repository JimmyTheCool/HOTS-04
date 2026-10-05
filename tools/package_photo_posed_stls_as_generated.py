#!/usr/bin/env python3
"""Package one best as-generated photo-posed STL from each retained V5 artifact.

This deliberately does not claim print-readiness. It selects Robust first, followed by
print-ready/repaired/high-detail/initial/raw, and records the exact source path.
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
from urllib.parse import urlsplit
import urllib.request
import zipfile


ARTIFACT_PATTERN = re.compile(r"^HOTS04-photo-posed-v5-(.+)$", re.IGNORECASE)


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        if urlsplit(req.full_url).netloc.casefold() != urlsplit(newurl).netloc.casefold():
            for key in ("Authorization", "Accept", "X-GitHub-Api-Version"):
                redirected.remove_header(key)
        return redirected


urllib.request.install_opener(urllib.request.build_opener(SafeRedirect()))


def log(message: str) -> None:
    print(f"[as-generated-packager] {message}", flush=True)


def api_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "HOTS04-as-generated-posed-packager",
    }


def fetch_json(url: str, token: str) -> dict:
    request = urllib.request.Request(url, headers=api_headers(token))
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def list_artifacts(repository: str, token: str) -> list[dict]:
    result: list[dict] = []
    page = 1
    while True:
        payload = fetch_json(
            f"https://api.github.com/repos/{repository}/actions/artifacts?per_page=100&page={page}",
            token,
        )
        batch = list(payload.get("artifacts", []))
        result.extend(batch)
        if len(batch) < 100:
            return result
        page += 1


def hero_from_artifact(name: str) -> str:
    match = ARTIFACT_PATTERN.match(name)
    value = match.group(1) if match else name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_.") or "Hero"


def download(repository: str, token: str, artifact: dict, destination: Path) -> Path:
    artifact_id = int(artifact["id"])
    hero = hero_from_artifact(str(artifact["name"]))
    archive_path = destination / f"{artifact_id}_{hero}.zip"
    extract_path = destination / f"{artifact_id}_{hero}"
    extract_path.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/actions/artifacts/{artifact_id}/zip",
        headers=api_headers(token),
    )
    with urllib.request.urlopen(request, timeout=900) as response, archive_path.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_path)
    archive_path.unlink(missing_ok=True)
    log(f"Downloaded {artifact['name']}")
    return extract_path


def rank(path: Path) -> tuple[int, int, str]:
    value = path.as_posix().casefold()
    if "/robust/" in value:
        return 700, path.stat().st_size, "Robust"
    if "print_ready" in value or "print-ready" in value or "/printready/" in value:
        return 650, path.stat().st_size, "PrintReady"
    if "/repaired/" in value:
        return 600, path.stat().st_size, "Repaired"
    if "/highdetail/" in value or "/high_detail/" in value:
        return 500, path.stat().st_size, "HighDetail"
    if "/initial/" in value:
        return 400, path.stat().st_size, "Initial"
    if "/raw/" in value:
        return 300, path.stat().st_size, "Raw"
    return 100, path.stat().st_size, "Other"


def copy_reports(source: Path, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    wanted = {
        "photo_pose_quality_summary.json",
        "pose_match.json",
        "photo_posed_build.json",
        "job_config.json",
        "source_resolution.json",
        "robust_validation.json",
        "high_detail_validation.json",
        "reference_masks.json",
        "HERO_COMPLETE.txt",
    }
    copied: list[str] = []
    for path in source.rglob("*"):
        if not path.is_file() or path.name not in wanted:
            continue
        target = destination / path.name
        counter = 2
        while target.exists():
            target = destination / f"{path.stem}_{counter}{path.suffix}"
            counter += 1
        shutil.copy2(path, target)
        copied.append(target.name)
    return sorted(copied)


def github_output(values: dict[str, str]) -> None:
    target = os.environ.get("GITHUB_OUTPUT")
    if not target:
        return
    with open(target, "a", encoding="utf-8") as handle:
        for key, value in values.items():
            value = str(value).replace("\r", " ").replace("\n", " ")
            handle.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--token", default=os.environ.get("GH_TOKEN"))
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    if not args.repository or not args.token:
        raise SystemExit("Repository and token are required")

    work = args.work.resolve()
    package = args.package.resolve()
    archive_path = args.archive.resolve()
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(package, ignore_errors=True)
    work.mkdir(parents=True)
    model_dir = package / "STL_Photo_Posed_As_Generated"
    report_dir = package / "Reports"
    model_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    artifacts = [
        artifact
        for artifact in list_artifacts(args.repository, args.token)
        if not artifact.get("expired", False)
        and ARTIFACT_PATTERN.match(str(artifact.get("name", "")))
    ]
    artifacts.sort(key=lambda item: str(item.get("created_at", "")))
    if not artifacts:
        raise SystemExit("No retained V5 photo-posed artifacts were found")

    rows: list[dict] = []
    download_errors: list[dict] = []
    for artifact in artifacts:
        hero = hero_from_artifact(str(artifact["name"]))
        try:
            extracted = download(args.repository, args.token, artifact, work)
        except Exception as exc:
            download_errors.append(
                {"hero": hero, "artifact_id": artifact.get("id"), "error": str(exc)}
            )
            continue
        candidates = [path for path in extracted.rglob("*.stl") if path.is_file() and path.stat().st_size > 84]
        if not candidates:
            download_errors.append(
                {"hero": hero, "artifact_id": artifact.get("id"), "error": "artifact contained no STL"}
            )
            continue
        candidates.sort(key=lambda path: rank(path), reverse=True)
        source = candidates[0]
        score, _, variant = rank(source)
        target = model_dir / f"{hero}.stl"
        shutil.copy2(source, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        available_reports = copy_reports(extracted, report_dir / hero)
        rows.append(
            {
                "hero": hero,
                "file": target.name,
                "variant": variant,
                "status": "posed model as generated; topology repair may be required",
                "artifact_id": artifact.get("id"),
                "artifact_name": artifact.get("name"),
                "artifact_created_at": artifact.get("created_at"),
                "source_path_inside_artifact": str(source.relative_to(extracted)),
                "available_stl_variants_in_artifact": len(candidates),
                "bytes": target.stat().st_size,
                "sha256": digest,
                "copied_report_files": "; ".join(available_reports),
            }
        )
        log(f"Selected {hero}: {variant} ({target.stat().st_size:,} bytes)")

    if not rows:
        raise SystemExit("No posed STL files could be recovered")
    rows.sort(key=lambda row: row["hero"].casefold())

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository": args.repository,
        "hero_count": len(rows),
        "heroes": [row["hero"] for row in rows],
        "purpose": "Delivery of all best available photo-posed models generated so far.",
        "important_warning": (
            "These files are supplied as generated. Previous strict validation found that the posed "
            "outputs did not pass the complete watertight/single-connected-body gate. They may contain "
            "disconnected parts, open edges, or non-manifold geometry and should be repaired before printing."
        ),
        "files": rows,
        "download_errors": download_errors,
    }
    manifest_json = report_dir / "manifest.json"
    manifest_csv = report_dir / "manifest.csv"
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with manifest_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    hero_list = "\n".join(f"- {row['hero']}: {row['variant']}" for row in rows)
    (package / "README_FIRST.txt").write_text(
        f"""HOTS-04 PHOTO-POSED STL FILES — AVAILABLE SO FAR

This ZIP contains {len(rows)} posed models recovered from completed V5 photo-pose
builds. The older generic T-pose files are not included.

INCLUDED
{hero_list}

IMPORTANT
These are the best models generated so far, not certified final print-ready meshes.
The strict validation run found topology defects in the posed outputs. They may have
open edges, disconnected components, or non-manifold areas. Run your slicer's repair
function or a mesh repair tool before printing. The Reports folder includes available
pose-selection and validation records.

The selected game mesh supplies the character dimensions and skin geometry. The pose
pipeline searches compatible in-game skeletal animation frames and compares their
four projected silhouettes with the four JPG reference views before freezing the
closest pose into the STL.

STL contains geometry only, not colour, textures, glow, transparency, particles, or
normal-map details. Personal, non-commercial fan use only.
""",
        encoding="utf-8",
    )

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.unlink(missing_ok=True)
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=7,
        allowZip64=True,
    ) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                archive.write(path, package.name / path.relative_to(package))
    archive_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    checksum = archive_path.with_name(archive_path.name + ".sha256")
    checksum.write_text(f"{archive_digest}  {archive_path.name}\n", encoding="utf-8")
    log(
        json.dumps(
            {
                "hero_count": len(rows),
                "heroes": [row["hero"] for row in rows],
                "archive_bytes": archive_path.stat().st_size,
                "archive_sha256": archive_digest,
            },
            indent=2,
        )
    )
    github_output(
        {
            "hero_count": str(len(rows)),
            "heroes": ", ".join(row["hero"] for row in rows),
            "archive": str(archive_path),
            "checksum": str(checksum),
            "manifest_json": str(manifest_json),
            "manifest_csv": str(manifest_csv),
            "archive_sha256": archive_digest,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
