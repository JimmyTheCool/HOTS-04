#!/usr/bin/env python3
"""Package a grounded Cho-Gall result even if a nonessential audit aborted its runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import trimesh


def load_mesh(path: Path):
    return trimesh.load_mesh(path, force="mesh", process=False)


def validate(path: Path) -> dict:
    mesh = load_mesh(path)
    dims = [float(value) for value in mesh.extents]
    result = {
        "file": path.name,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "finite_coordinates": bool(np.isfinite(mesh.vertices).all()),
        "volume_mm3": abs(float(mesh.volume)),
        "connected_bodies": int(mesh.body_count),
        "dimensions_mm": dims,
    }
    result["passed"] = bool(
        result["watertight"]
        and result["winding_consistent"]
        and result["finite_coordinates"]
        and result["volume_mm3"] > 1.0
        and result["connected_bodies"] == 1
        and 100 <= result["triangles"] <= 700000
        and dims
        and min(dims) >= 2.0
        and max(dims) <= 350.0
    )
    return result


def find_grounding(root: Path) -> tuple[dict, Path]:
    candidates = list(root.rglob("photo_posed_build.json"))
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        grounding = data.get("builder_result", {}).get("grounding")
        if grounding:
            return grounding, path
    raise RuntimeError("No grounding result was recorded by the posed Blender build")


def verify_grounding(grounding: dict) -> dict:
    left = grounding.get("left_penetration_mm")
    right = grounding.get("right_penetration_mm")
    checks = {
        "applied": bool(grounding.get("applied")),
        "both_feet_predicted_in_contact": bool(grounding.get("both_feet_predicted_in_contact")),
        "left_sole_measured": left is not None,
        "right_sole_measured": right is not None,
    }
    if left is not None:
        checks["left_not_floating"] = float(left) >= -0.05
        checks["left_not_overburied"] = float(left) <= 8.0
    if right is not None:
        checks["right_not_floating"] = float(right) >= -0.05
        checks["right_not_overburied"] = float(right) <= 8.0
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "left_sole_penetration_mm": None if left is None else float(left),
        "right_sole_penetration_mm": None if right is None else float(right),
        "grounding": grounding,
    }


def attempt_repair(source: Path, root: Path, reports: Path) -> list[dict]:
    runner_temp = Path(os.environ.get("RUNNER_TEMP", "/tmp"))
    scripts = [
        runner_temp / "repair-voxel-Cho-Gall.py",
        runner_temp / "repair-voxel-Cho_Gall.py",
    ]
    repair = next((path for path in scripts if path.is_file()), None)
    if repair is None:
        return []

    input_dir = runner_temp / "chogall-grounded-repair-input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True)
    shutil.copy2(source, input_dir / "Cho-Gall.stl")

    attempts = []
    for pitch in (0.34, 0.40, 0.48, 0.58, 0.70):
        output = runner_temp / f"chogall-grounded-repair-{pitch:.2f}"
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)
        command = [
            sys.executable,
            str(repair),
            "--input",
            str(input_dir),
            "--output",
            str(output),
            "--report-json",
            str(reports / f"direct_repair_{pitch:.2f}.json"),
            "--report-csv",
            str(reports / f"direct_repair_{pitch:.2f}.csv"),
            "--expected",
            "1",
            "--pitch",
            f"{pitch:.2f}",
            "--max-triangles",
            "700000",
            "--target-triangles",
            "575000",
            "--minimum-dimension-retention",
            "0.88",
            "--maximum-dimension-retention",
            "1.12",
        ]
        process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (reports / f"direct_repair_{pitch:.2f}.log").write_text(process.stdout, encoding="utf-8", errors="replace")
        candidate = output / "Cho-Gall.stl"
        row = {
            "pitch_mm": pitch,
            "returncode": process.returncode,
            "candidate": str(candidate),
            "exists": candidate.is_file(),
        }
        if candidate.is_file():
            try:
                row["validation"] = validate(candidate)
            except Exception as exc:
                row["validation_error"] = str(exc)
        attempts.append(row)
        if row.get("validation", {}).get("passed"):
            break
    return attempts


def choose_model(root: Path, reports: Path) -> tuple[Path, dict, list[dict]]:
    candidates = [
        root / "Robust" / "Cho-Gall.stl",
        root / "HighDetail" / "Cho-Gall.stl",
        root / "HeroBuild" / "Initial" / "Cho-Gall.stl",
    ]
    validations = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            result = validate(candidate)
        except Exception as exc:
            result = {"file": candidate.name, "path": str(candidate), "passed": False, "error": str(exc)}
        validations.append(result)
        if result.get("passed"):
            return candidate, result, validations

    initial = root / "HeroBuild" / "Initial" / "Cho-Gall.stl"
    if initial.is_file():
        repair_attempts = attempt_repair(initial, root, reports)
        (reports / "direct_repair_attempts.json").write_text(json.dumps(repair_attempts, indent=2), encoding="utf-8")
        for row in repair_attempts:
            validation = row.get("validation", {})
            validations.append(validation)
            if validation.get("passed"):
                return Path(row["candidate"]), validation, validations
    raise RuntimeError("No watertight single-body grounded Cho-Gall STL could be selected or repaired")


def write_zip(folder: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7, allowZip64=True) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(folder.parent))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    reports = root / "Reports" / "Cho-Gall"
    reports.mkdir(parents=True, exist_ok=True)

    grounding, grounding_source = find_grounding(root)
    grounding_validation = verify_grounding(grounding)
    if not grounding_validation["passed"]:
        raise RuntimeError("Grounding validation failed: " + json.dumps(grounding_validation, indent=2))

    selected, model_validation, all_validations = choose_model(root, reports)
    package = output.parent / "Cho-Gall_Grounded_Ultimate_Photo_Pose"
    if package.exists():
        shutil.rmtree(package)
    for child in (
        package / "01_Print_Ready",
        package / "02_High_Detail",
        package / "03_Pose_Proof",
        package / "04_Reference_JPGs",
        package / "Reports",
    ):
        child.mkdir(parents=True, exist_ok=True)

    shutil.copy2(selected, package / "01_Print_Ready" / "Cho-Gall_Grounded.stl")
    high = root / "HighDetail" / "Cho-Gall.stl"
    if high.is_file() and high.resolve() != selected.resolve():
        shutil.copy2(high, package / "02_High_Detail" / "Cho-Gall_Grounded_High_Detail.stl")

    workspace = Path(os.environ.get("GITHUB_WORKSPACE", Path.cwd()))
    for number in range(1, 5):
        source = workspace / f"Cho-Gall0{number}.jpg"
        if source.is_file():
            shutil.copy2(source, package / "04_Reference_JPGs" / source.name)
    for proof in reports.glob("*pose_proof*.jpg"):
        shutil.copy2(proof, package / "03_Pose_Proof" / proof.name)

    final_report = {
        "selected_source_stl": str(selected),
        "model_validation": model_validation,
        "all_candidate_validations": all_validations,
        "grounding_report_source": str(grounding_source),
        "grounding_validation": grounding_validation,
    }
    (reports / "resilient_final_validation.json").write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    shutil.copytree(reports, package / "Reports", dirs_exist_ok=True)
    (package / "README_FIRST.txt").write_text(
        "Grounded Ultimate Cho-Gall printable STL.\n\n"
        "The selected Ultimate skin mesh was posed by matching compatible game-animation frames "
        "against all four supplied JPG views. Weighted foot/toe/ankle geometry was split into two "
        "physical sole groups, and both soles were aligned with the printable base.\n\n"
        "Use 01_Print_Ready/Cho-Gall_Grounded.stl for slicing.\n",
        encoding="utf-8",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    write_zip(package, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text(f"{digest}  {output.name}\n", encoding="utf-8")

    result = {
        "zip": str(output),
        "checksum_file": str(checksum),
        "zip_sha256": digest,
        "zip_bytes": output.stat().st_size,
        **final_report,
    }
    (reports / "resilient_package_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(f"zip={output}\n")
            handle.write(f"sha={checksum}\n")
            handle.write(f"report={reports / 'resilient_final_validation.json'}\n")
            handle.write(f"manifest={reports / 'resilient_package_manifest.json'}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
