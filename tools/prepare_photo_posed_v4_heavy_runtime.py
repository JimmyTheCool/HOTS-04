#!/usr/bin/env python3
"""Prepare a memory-safe V4 runtime for spatially large HOTS hero meshes."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one target, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    source_plan = root / "reference_analysis" / "selected_skin_plan.json"
    target_plan = root / "reference_analysis" / "selected_skin_plan_v4.json"
    plan = json.loads(source_plan.read_text(encoding="utf-8"))
    plan["selection_policy"] = (
        "User-corrected skin selection plus four-JPG animation-frame matching; "
        "memory-safe heavy-model reconstruction."
    )
    plan["heroes"]["Rexxar"].update(
        {
            "skin_key": "base",
            "directory": "storm_hero_rexxar_base",
            "repository": "HOTS-03",
            "model_path": "storm_hero_rexxar_base/storm_hero_rexxar_base.m3",
            "selection_method": "user correction: photographed Rexxar is the base skin",
        }
    )
    plan["heroes"]["Dehaka"].update(
        {
            "skin_key": "mecha",
            "directory": "storm_hero_dehaka_mecha",
            "repository": "HOTS-02",
            "model_path": "storm_hero_dehaka_mecha/storm_hero_dehaka_mecha.m3",
            "selection_method": "user-confirmed Mecha skin",
        }
    )
    target_plan.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    runner = root / "tools" / "run_photo_posed_job_v4_heavy_runtime.py"
    builder = root / "tools" / "build_one_photo_posed_v4_heavy_runtime.py"
    shutil.copy2(root / "tools" / "exposed_r2" / "run_photo_posed_job_v3_1_source.py", runner)
    shutil.copy2(root / "tools" / "exposed_r2" / "build_one_photo_posed_v3_source.py", builder)
    subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "patch_photo_pose_v4.py"),
            "--runner",
            str(runner),
            "--builder",
            str(builder),
        ],
        check=True,
    )

    text = runner.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '"--voxel-mm", "0.30", "--thin-part-mm", "0.70", "--max-triangles", "600000",',
        '"--voxel-mm", "0.55", "--thin-part-mm", "0.70", "--max-triangles", "600000",',
        "memory-safe interim voxel pitch",
    )
    text = replace_once(
        text,
        '''    if any(token in safe.casefold() for token in ("fenix", "dehaka", "zagara", "cho-gall", "cho_gall")):
        fine_pitch = "0.24"
    else:
        fine_pitch = "0.20"
''',
        '''    folded_safe = safe.casefold()
    if "fenix" in folded_safe:
        fine_pitch = "0.42"
    elif "dehaka" in folded_safe:
        fine_pitch = "0.36"
    elif "rexxar" in folded_safe:
        fine_pitch = "0.30"
    else:
        fine_pitch = "0.28"
''',
        "heavy-model fine pitch selection",
    )
    text = replace_once(
        text,
        'shutil.copy2(initial_source, fine_input / f"{safe}.stl")',
        'shutil.copy2(raw_source, fine_input / f"{safe}.stl")',
        "frozen raw mesh high-detail input",
    )
    text = replace_once(
        text,
        '"--expected", "1", "--pitch", fine_pitch, "--max-triangles", "1500000", "--target-triangles", "1200000",',
        '"--expected", "1", "--pitch", fine_pitch, "--max-triangles", "2000000", "--target-triangles", "2000000",',
        "high-detail triangle budget",
    )
    text = replace_once(
        text,
        '"--expected", "1", "--max-triangles", "1500000",',
        '"--expected", "1", "--max-triangles", "2000000",',
        "high-detail validation triangle budget",
    )
    text = replace_once(
        text,
        '"high_detail_triangle_limit": 1500000,',
        '"high_detail_triangle_limit": 2000000,',
        "quality-report triangle budget",
    )
    runner.write_text(text, encoding="utf-8")
    compile(text, str(runner), "exec")
    compile(builder.read_text(encoding="utf-8"), str(builder), "exec")
    print(f"Prepared heavy runner: {runner}")
    print(f"Prepared heavy builder: {builder}")
    print(f"Prepared corrected plan: {target_plan}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
