#!/usr/bin/env python3
"""Apply rig-aware grounding and correct the builder's scaled-bound conversion."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: patch_builder_ground_weighted_feet_v2.py BUILDER.py")
    builder = Path(sys.argv[1]).resolve()
    base_patch = Path(__file__).resolve().with_name("patch_builder_ground_weighted_feet.py")
    subprocess.run([sys.executable, str(base_patch), str(builder)], check=True)
    text = builder.read_text(encoding="utf-8")
    old = 'return None if value is None else (float(value) - before_min_z) * factor'
    new = 'return None if value is None else float(value) * factor - before_min_z'
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one sole scaling expression; found {text.count(old)}")
    text = text.replace(old, new, 1)
    builder.write_text(text, encoding="utf-8")
    compile(text, str(builder), "exec")
    print(f"Corrected scaled sole-height conversion in {builder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
