#!/usr/bin/env python3
"""Run one end-to-end HOTS photo-posed STL build on a GitHub Actions runner."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import zlib


def log(message: str) -> None:
    print(f"[photo-pose-job] {message}", flush=True)


def run(command: list[str], *, cwd: Path | None = None, check: bool = True,
        env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    log("RUN " + shlex.join(command))
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=check,
        env=env,
        text=True,
    )


def run_capture(command: list[str], *, cwd: Path | None = None) -> str:
    log("RUN " + shlex.join(command))
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def run_tee(command: list[str], log_path: Path, *, cwd: Path | None = None,
            check: bool = True, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log("RUN " + shlex.join(command))
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        status = process.wait()
    if check and status != 0:
        raise subprocess.CalledProcessError(status, command)
    return status


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_.")
    return value or "hero"


def verify_pose_builder(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"Uncompressed source SHA-256:\n([0-9a-f]{64})", text)
    if not match:
        raise RuntimeError("Pose builder payload checksum header is missing")
    expected = match.group(1)
    tree = ast.parse(text)
    payload = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "_PAYLOAD":
                payload = ast.literal_eval(node.value)
                break
    if payload is None:
        raise RuntimeError("Pose builder payload is missing")
    source = zlib.decompress(base64.b85decode(payload)).decode("utf-8")
    actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Pose builder payload checksum mismatch: {actual} != {expected}")
    compile(source, str(path), "exec")
    return actual


def write_env(name: str, value: str) -> None:
    target = os.environ.get("GITHUB_ENV")
    if not target:
        return
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hero", required=True)
    parser.add_argument("--max-animations", type=int, default=22)
    parser.add_argument("--max-coarse-frames", type=int, default=144)
    parser.add_argument("--refine-candidates", type=int, default=9)
    args = parser.parse_args()

    workspace = Path(os.environ.get("GITHUB_WORKSPACE", Path.cwd())).resolve()
    runner_temp = Path(os.environ.get("RUNNER_TEMP", "/tmp")).resolve()
    plan_path = workspace / "reference_analysis" / "selected_skin_plan.json"
    plan = load_json(plan_path)["heroes"]
    if args.hero not in plan:
        raise KeyError(f"Hero not found in selected skin plan: {args.hero}")
    selected = plan[args.hero]
    hero = args.hero
    safe = safe_name(hero)
    source_repo = selected["repository"]
    model_path = selected["model_path"].replace("\\", "/")
    skin_key = selected["skin_key"]
    photos = [workspace / item for item in selected["reference_images"]]
    if len(photos) != 4 or any(not path.is_file() for path in photos):
        raise FileNotFoundError(f"Expected four reference photos for {hero}: {photos}")

    root = runner_temp / f"HOTS04-photo-posed-v3-{safe}"
    reports = root / "Reports" / safe
    high_detail = root / "HighDetail"
    robust = root / "Robust"
    raw = root / "Raw"
    hero_build = root / "HeroBuild"
    for directory in (reports, high_detail, robust, raw, hero_build):
        directory.mkdir(parents=True, exist_ok=True)

    config = {
        "hero": hero,
        "safe": safe,
        "repository": source_repo,
        "model_path": model_path,
        "skin_key": skin_key,
        "photos": [str(path.relative_to(workspace)) for path in photos],
        "selected_skin_plan": selected,
    }
    (reports / "job_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_env("HERO_ARTIFACT_ROOT", str(root))
    write_env("HERO_SAFE", safe)
    log(json.dumps(config, indent=2))

    tools_root = runner_temp / "HOTS-03-tools"
    run([
        "git", "clone", "--filter=blob:none", "--sparse", "--depth", "1",
        "--branch", "chatgpt-stl-builder",
        "https://github.com/JimmyTheCool/HOTS-03.git", str(tools_root),
    ])
    run(["git", "-C", str(tools_root), "sparse-checkout", "set", "ci", "tools/hots_stl"])

    source_root = runner_temp / f"source-{safe}"
    run([
        "git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1",
        "--branch", "main", f"https://github.com/JimmyTheCool/{source_repo}.git", str(source_root),
    ])
    resolution_path = reports / "source_resolution.json"
    run([
        sys.executable, str(workspace / "tools" / "resolve_pose_source_v3.py"),
        "--repo", str(source_root), "--model", model_path, "--hero", hero,
        "--output", str(resolution_path),
    ])
    resolution = load_json(resolution_path)
    sparse_dirs = [str(value) for value in resolution["sparse_directories"]]
    animation_rel = [str(value) for value in resolution["animation_paths"]]
    if not animation_rel:
        log(f"WARNING: no external M3A bank resolved for {hero}; embedded model animations will be tested")
    run(["git", "-C", str(source_root), "sparse-checkout", "init", "--cone"])
    run(["git", "-C", str(source_root), "sparse-checkout", "set", *sparse_dirs])
    run(["git", "-C", str(source_root), "checkout", "main"])
    model = source_root / model_path
    animation_paths = [source_root / path for path in animation_rel]
    if not model.is_file() or any(not path.is_file() for path in animation_paths):
        raise FileNotFoundError(f"Sparse source checkout incomplete for {hero}")

    run(["sudo", "apt-get", "update", "-qq"])
    run([
        "sudo", "apt-get", "install", "-y", "--no-install-recommends",
        "ca-certificates", "curl", "git", "xz-utils", "p7zip-full",
        "libx11-6", "libxi6", "libxrender1", "libxfixes3", "libxkbcommon0",
        "libsm6", "libice6", "libgl1", "libegl1",
    ])

    blender_version = os.environ.get("BLENDER_VERSION", "3.6.23")
    blender_dir = runner_temp / f"blender-{blender_version}-linux-x64"
    blender = blender_dir / "blender"
    archive = runner_temp / f"blender-{blender_version}-linux-x64.tar.xz"
    if not blender.is_file():
        run([
            "curl", "--fail", "--location", "--retry", "5", "--retry-delay", "5",
            f"https://download.blender.org/release/Blender3.6/{archive.name}",
            "--output", str(archive),
        ])
        run(["tar", "-xf", str(archive), "-C", str(runner_temp)])
    if not os.access(blender, os.X_OK):
        raise RuntimeError(f"Blender executable is missing: {blender}")

    addon_commit = os.environ.get("M3ADDON_COMMIT", "58935005c4465a219a740f6b740253533eda773f")
    addon_source = runner_temp / "m3addon-src"
    run(["git", "clone", "https://github.com/SC2Mapster/m3addon.git", str(addon_source)])
    run(["git", "-C", str(addon_source), "checkout", addon_commit])
    addon_dir = Path.home() / ".config" / "blender" / "3.6" / "scripts" / "addons"
    addon_dir.mkdir(parents=True, exist_ok=True)
    addon_link = addon_dir / "m3addon"
    if addon_link.exists() or addon_link.is_symlink():
        addon_link.unlink()
    addon_link.symlink_to(addon_source, target_is_directory=True)

    run([
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--quiet",
        "numpy==2.2.6", "scipy==1.15.3", "pillow==11.1.0", "trimesh==4.8.3",
        "scikit-image==0.25.2", "pymeshfix==0.18.1",
        "fast-simplification==0.1.12", "matplotlib==3.10.1",
        "opencv-python-headless==4.12.0.88",
    ])

    pose_builder = workspace / "tools" / "build_one_photo_posed_v3.py"
    payload_hash = verify_pose_builder(pose_builder)
    log(f"Verified pose builder payload {payload_hash}")

    masks_npz = reports / "reference_masks.npz"
    run([
        sys.executable, str(workspace / "tools" / "prepare_pose_reference_masks_v4.py"),
        "--hero", hero, "--photos", *[str(path) for path in photos],
        "--output-npz", str(masks_npz),
        "--preview", str(reports / f"{safe}_reference_masks.jpg"),
        "--report", str(reports / "reference_masks.json"),
    ])

    repair_normal = runner_temp / f"repair-normal-{safe}.py"
    repair_voxel = runner_temp / f"repair-voxel-{safe}.py"
    shutil.copy2(tools_root / "ci" / "repair_stl_collection.py", repair_normal)
    for patch in (
        "patch_repair_dimension_retention.py",
        "patch_repair_voxel_translation.py",
        "patch_repair_face_connectivity.py",
    ):
        run([sys.executable, str(tools_root / "ci" / patch), str(repair_normal)])
    shutil.copy2(repair_normal, repair_voxel)
    run([sys.executable, str(tools_root / "ci" / "patch_repair_force_voxel.py"), str(repair_voxel)])

    builder = hero_build / "builder.py"
    shutil.copy2(tools_root / "tools" / "hots_stl" / "runtime_extracted" / "build_hots_stls.py", builder)
    run([sys.executable, str(tools_root / "tools" / "hots_stl" / "runtime_patch_v2.py"), str(builder)])
    run([sys.executable, str(tools_root / "ci" / "patch_builder_external_validation.py"), str(builder)])

    blender_command = [
        str(blender), "--background", "--factory-startup", "--python", str(pose_builder), "--",
        "--builder", str(builder), "--model", str(model),
    ]
    for path in animation_paths:
        blender_command.extend(["--animation", str(path)])
    blender_command.extend([
        "--hero", hero, "--skin-key", skin_key, "--reference-npz", str(masks_npz),
        "--output", str(hero_build), "--target-height-mm", "120",
        "--base-height-mm", "4.5", "--base-margin-mm", "5.0",
        "--voxel-mm", "0.30", "--thin-part-mm", "0.70", "--max-triangles", "600000",
        "--max-animations", str(args.max_animations),
        "--max-coarse-frames", str(args.max_coarse_frames),
        "--refine-candidates", str(args.refine_candidates),
    ])
    run_tee(blender_command, reports / "photo_pose_builder.log")

    raw_source = hero_build / "Raw" / f"{safe}.stl"
    initial_source = hero_build / "Initial" / f"{safe}.stl"
    if not raw_source.is_file() or not initial_source.is_file():
        raise RuntimeError(f"Photo-pose builder did not emit both STL variants for {hero}")
    shutil.copy2(raw_source, raw / f"{safe}.stl")
    shutil.copy2(initial_source, high_detail / f"{safe}.stl")
    for item in (hero_build / "Reports").iterdir():
        target = reports / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    run([
        sys.executable, str(workspace / "tools" / "make_pose_proof_sheet_v3.py"),
        "--hero", hero, "--photos", *[str(path) for path in photos],
        "--reference-npz", str(masks_npz), "--matched-dir", str(reports),
        "--pose-report", str(reports / "pose_match.json"),
        "--output", str(reports / f"{safe}_pose_proof.jpg"),
    ])

    one_input = runner_temp / f"one-{safe}"
    if one_input.exists():
        shutil.rmtree(one_input)
    one_input.mkdir(parents=True)
    shutil.copy2(initial_source, one_input / f"{safe}.stl")
    adaptive_status = run_tee([
        sys.executable, str(tools_root / "ci" / "repair_one_adaptive.py"),
        "--input-file", str(one_input / f"{safe}.stl"),
        "--output", str(robust), "--report-dir", str(reports / "Adaptive"),
        "--normal-script", str(repair_normal), "--voxel-script", str(repair_voxel),
        "--validator", str(tools_root / "ci" / "validate_stl_collection.py"),
        "--silhouette", str(tools_root / "ci" / "compare_stl_silhouette.py"),
        "--max-triangles", "600000", "--target-triangles", "500000",
        "--minimum-dimension-retention", "0.90", "--maximum-dimension-retention", "1.10",
    ], reports / "adaptive_repair.log", check=False)
    robust_file = robust / f"{safe}.stl"
    if adaptive_status != 0 or not robust_file.is_file():
        fallback = runner_temp / f"fallback-{safe}"
        if fallback.exists():
            shutil.rmtree(fallback)
        fallback.mkdir(parents=True)
        run_tee([
            sys.executable, str(repair_voxel), "--input", str(one_input), "--output", str(fallback),
            "--report-json", str(reports / "fallback.json"),
            "--report-csv", str(reports / "fallback.csv"), "--expected", "1",
            "--pitch", "0.32", "--max-triangles", "600000", "--target-triangles", "500000",
        ], reports / "fallback.log")
        shutil.copy2(fallback / f"{safe}.stl", robust_file)
    if not robust_file.is_file():
        raise RuntimeError(f"No robust print-ready STL was produced for {hero}")

    validator = tools_root / "ci" / "validate_stl_collection.py"
    comparer = tools_root / "ci" / "compare_stl_silhouette.py"
    run_tee([
        sys.executable, str(validator), "--input", str(high_detail),
        "--report-json", str(reports / "high_detail_validation.json"),
        "--report-csv", str(reports / "high_detail_validation.csv"),
        "--checksums", str(reports / "HIGH_DETAIL_SHA256SUMS.txt"),
        "--expected", "1", "--max-triangles", "600000",
    ], reports / "high_detail_validation.log")
    run_tee([
        sys.executable, str(validator), "--input", str(robust),
        "--report-json", str(reports / "robust_validation.json"),
        "--report-csv", str(reports / "robust_validation.csv"),
        "--checksums", str(reports / "ROBUST_SHA256SUMS.txt"),
        "--expected", "1", "--max-triangles", "500000",
    ], reports / "robust_validation.log")
    run_tee([
        sys.executable, str(comparer), "--raw", str(raw), "--repaired", str(high_detail),
        "--report-json", str(reports / "raw_to_high_detail.json"),
        "--report-csv", str(reports / "raw_to_high_detail.csv"),
        "--image", str(reports / "raw_to_high_detail.png"), "--expected", "1",
        "--pixel-mm", "0.40", "--tolerance-mm", "1.20",
        "--minimum-raw-coverage", "0.940", "--minimum-repaired-coverage", "0.850",
        "--minimum-iou", "0.780", "--maximum-points", "800000",
    ], reports / "raw_to_high_detail.log")
    run_tee([
        sys.executable, str(comparer), "--raw", str(high_detail), "--repaired", str(robust),
        "--report-json", str(reports / "high_detail_to_robust.json"),
        "--report-csv", str(reports / "high_detail_to_robust.csv"),
        "--image", str(reports / "high_detail_to_robust.png"), "--expected", "1",
        "--pixel-mm", "0.45", "--tolerance-mm", "1.35",
        "--minimum-raw-coverage", "0.950", "--minimum-repaired-coverage", "0.850",
        "--minimum-iou", "0.780", "--maximum-points", "800000",
    ], reports / "high_detail_to_robust.log")

    pose_report = load_json(reports / "pose_match.json")
    pose = pose_report["selected"]
    if not str(pose.get("animation_name", "")).strip():
        raise RuntimeError(f"The selected {hero} pose has no animation name")
    if int(pose_report.get("coarse_frames_tested", 0)) < 2:
        raise RuntimeError(f"The {hero} pose search did not test enough animated frames")
    if int(pose_report.get("refined_frames_tested", 0)) < 1:
        raise RuntimeError(f"The {hero} pose search did not refine any animated frame")
    animation_imports = pose_report.get("animation_imports", [])
    external_animation_added = any(
        int(item.get("added", 0)) > 0
        for item in animation_imports
        if item.get("status") == "success"
    )
    if not external_animation_added and not pose_report.get("animation_inventory"):
        raise RuntimeError(f"No skeletal animations were available for {hero}")

    high_validation = load_json(reports / "high_detail_validation.json")
    robust_validation = load_json(reports / "robust_validation.json")
    quality = {
        "hero": hero,
        "safe": safe,
        "skin_key": skin_key,
        "model_path": model_path,
        "selected_animation": pose.get("animation_name"),
        "selected_frame": pose.get("frame"),
        "pose_score": pose.get("pose_score"),
        "rest_pose_baseline_score": (
            pose_report.get("rest_pose_baseline", {}).get("score")
            if pose_report.get("rest_pose_baseline") else None
        ),
        "improvement_over_rest_pose": pose_report.get("improvement_over_rest_pose"),
        "camera": pose.get("camera"),
        "coarse_frames_tested": pose_report.get("coarse_frames_tested"),
        "refined_frames_tested": pose_report.get("refined_frames_tested"),
        "high_detail_validation_passed": high_validation.get("passed"),
        "robust_validation_passed": robust_validation.get("passed"),
        "source_resolution": resolution,
        "payload_sha256": payload_hash,
    }
    (reports / "photo_pose_quality_summary.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    (root / "HERO_COMPLETE.txt").write_text(
        f"{hero}\n{skin_key}\n{pose.get('animation_name')}\nframe={pose.get('frame')}\nscore={pose.get('pose_score')}\n",
        encoding="utf-8",
    )
    log("COMPLETED\n" + json.dumps(quality, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
