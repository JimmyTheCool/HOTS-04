#!/usr/bin/env python3
"""Prepare V4 pose matching while retaining the proven stable repair settings."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import prepare_photo_pose_v4 as v4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-bootstrap", required=True, type=Path)
    parser.add_argument("--pose-source", required=True, type=Path)
    parser.add_argument("--output-runner", required=True, type=Path)
    parser.add_argument("--output-pose", required=True, type=Path)
    args = parser.parse_args()

    pose = v4.patch_pose_source(args.pose_source.read_text(encoding="utf-8"))
    args.output_pose.write_text(pose, encoding="utf-8")

    runner = v4.decode_bootstrap(args.runner_bootstrap)
    runner = v4.replace_once(
        runner,
        r'    pose_builder\s*=\s*workspace\s*/\s*"tools"\s*/\s*"build_one_photo_posed_v3\.py"\n',
        f'    pose_builder = Path({str(args.output_pose.resolve())!r})\n',
        "pose builder path",
    )
    runner = v4.replace_once(
        runner,
        r'    payload_hash\s*=\s*verify_pose_builder\(pose_builder\)\n',
        '    payload_hash = hashlib.sha256(pose_builder.read_bytes()).hexdigest()\n',
        "pose builder verification",
    )
    compile(runner, "run_photo_posed_job_v4_safe.py", "exec")
    args.output_runner.write_text(runner, encoding="utf-8")

    print({
        "runner_sha256": hashlib.sha256(runner.encode("utf-8")).hexdigest(),
        "pose_sha256": hashlib.sha256(pose.encode("utf-8")).hexdigest(),
        "repair_policy": "original stable v3.2 dynamic pitch and adaptive fallback",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
