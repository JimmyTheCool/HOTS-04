#!/usr/bin/env python3
"""Apply rig-aware foot grounding with spatial clustering fallback."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: patch_builder_ground_weighted_feet_v4.py BUILDER.py")
    builder = Path(sys.argv[1]).resolve()
    v3 = Path(__file__).resolve().with_name("patch_builder_ground_weighted_feet_v3.py")
    subprocess.run([sys.executable, str(v3), str(builder)], check=True)
    text = builder.read_text(encoding="utf-8")

    old = '''    left = sole(points["left"])
    right = sole(points["right"])
    unknown = sole(points["unknown"])
    combined_values = points["left"] + points["right"] + points["unknown"]
'''
    new = '''    left = sole(points["left"])
    right = sole(points["right"])
    unknown = sole(points["unknown"])

    # Some Blizzard rigs use names such as Bip01LFoot or use no explicit side
    # marker at all. When semantic side labels are unavailable, split the
    # weighted foot vertices into two spatial clusters along the wider ground
    # axis. This identifies the two physical soles without relying on bone names.
    clustered = False
    cluster_sizes = None
    if (left is None or right is None) and len(points["unknown"]) >= 24:
        unknown_points = list(points["unknown"])
        x_values = [float(item[0]) for item in unknown_points]
        y_values = [float(item[1]) for item in unknown_points]
        axis = 0 if (max(x_values) - min(x_values)) >= (max(y_values) - min(y_values)) else 1
        values = [float(item[axis]) for item in unknown_points]
        centre_a = min(values)
        centre_b = max(values)
        group_a = []
        group_b = []
        for _iteration in range(16):
            group_a = []
            group_b = []
            for item, value in zip(unknown_points, values):
                if abs(value - centre_a) <= abs(value - centre_b):
                    group_a.append(item)
                else:
                    group_b.append(item)
            if not group_a or not group_b:
                break
            next_a = sum(float(item[axis]) for item in group_a) / len(group_a)
            next_b = sum(float(item[axis]) for item in group_b) / len(group_b)
            if abs(next_a - centre_a) < 1e-7 and abs(next_b - centre_b) < 1e-7:
                centre_a, centre_b = next_a, next_b
                break
            centre_a, centre_b = next_a, next_b
        if len(group_a) >= 8 and len(group_b) >= 8 and abs(centre_b - centre_a) > 1e-5:
            sole_a = sole(group_a)
            sole_b = sole(group_b)
            if left is None and right is None:
                left, right = sole_a, sole_b
            elif left is None:
                left = max(sole_a, sole_b) if right is not None else sole_a
            elif right is None:
                right = max(sole_a, sole_b) if left is not None else sole_b
            clustered = True
            cluster_sizes = [len(group_a), len(group_b)]

    combined_values = points["left"] + points["right"] + points["unknown"]
'''
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one foot measurement block; found {text.count(old)}")
    text = text.replace(old, new, 1)

    old_report = '''        "matching_groups": sorted(set(group_names)),
        "both_feet_detected": left is not None and right is not None,
'''
    new_report = '''        "matching_groups": sorted(set(group_names)),
        "unlabelled_vertices_clustered": clustered,
        "unlabelled_cluster_sizes": cluster_sizes,
        "both_feet_detected": left is not None and right is not None,
'''
    if text.count(old_report) != 1:
        raise RuntimeError(f"Expected one grounding report block; found {text.count(old_report)}")
    text = text.replace(old_report, new_report, 1)

    builder.write_text(text, encoding="utf-8")
    compile(text, str(builder), "exec")
    print(f"Applied clustered two-sole grounding to {builder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
