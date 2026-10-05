#!/usr/bin/env python3
"""Apply robust rig-aware foot grounding to a patched HOTS STL builder."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label}; found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: patch_builder_ground_weighted_feet_v3.py BUILDER.py")
    builder = Path(sys.argv[1]).resolve()
    v2 = Path(__file__).resolve().with_name("patch_builder_ground_weighted_feet_v2.py")
    subprocess.run([sys.executable, str(v2), str(builder)], check=True)
    text = builder.read_text(encoding="utf-8")

    old_side = '''def _ground_side_from_group(name: str) -> str:
    value = name.casefold().replace("-", "_").replace(".", "_")
    tokens = [token for token in value.split("_") if token]
    left_tokens = {"l", "lf", "left", "lft"}
    right_tokens = {"r", "rt", "right", "rgt"}
    if any(token in left_tokens for token in tokens) or value.endswith(("left", "_l", " l")):
        return "left"
    if any(token in right_tokens for token in tokens) or value.endswith(("right", "_r", " r")):
        return "right"
    return "unknown"
'''
    new_side = '''def _ground_side_from_group(name: str) -> str:
    raw = name.casefold().strip()
    value = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    tokens = [token for token in value.split("_") if token]
    left_tokens = {"l", "lf", "left", "lft", "lhs"}
    right_tokens = {"r", "rt", "right", "rgt", "rhs"}
    if any(token in left_tokens for token in tokens):
        return "left"
    if any(token in right_tokens for token in tokens):
        return "right"
    compact = re.sub(r"[^a-z0-9]+", "", raw)
    semantic = r"(?:foot|toe|ankle|heel|hoof)"
    if re.search(rf"(?:left|lft){semantic}|{semantic}(?:left|lft|l)$|^l{semantic}", compact):
        return "left"
    if re.search(rf"(?:right|rgt){semantic}|{semantic}(?:right|rgt|r)$|^r{semantic}", compact):
        return "right"
    return "unknown"
'''
    text = replace_once(text, old_side, new_side, "foot side classifier")

    text = replace_once(
        text,
        '"method": "evaluated vertex groups",\n        "left_raw_z": left,',
        '"method": "evaluated vertex groups",\n        "coordinate_space": "source",\n        "left_raw_z": left,',
        "semantic coordinate-space marker",
    )
    text = replace_once(
        text,
        'return {"method": "geometric fallback", "combined_raw_z": float(minimum.z), "both_feet_detected": False}',
        'return {"method": "geometric fallback", "coordinate_space": "normalized", "combined_raw_z": float(minimum.z), "both_feet_detected": False}',
        "empty fallback coordinate marker",
    )
    text = replace_once(
        text,
        '"method": "lower-body geometric fallback",\n        "left_raw_z": percentile(left),',
        '"method": "lower-body geometric fallback",\n        "coordinate_space": "normalized",\n        "left_raw_z": percentile(left),',
        "fallback coordinate-space marker",
    )
    text = replace_once(
        text,
        '''    def normalized(value):
        return None if value is None else float(value) * factor - before_min_z
''',
        '''    def normalized(value):
        if value is None:
            return None
        if estimate.get("coordinate_space") == "normalized":
            return float(value)
        return float(value) * factor - before_min_z
''',
        "coordinate-aware normalizer",
    )

    builder.write_text(text, encoding="utf-8")
    compile(text, str(builder), "exec")
    print(f"Applied robust rig-aware foot grounding to {builder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
