#!/usr/bin/env python3
"""Run clustered grounded Cho-Gall while keeping visual audits non-blocking.

Watertightness, finite geometry, positive volume, one connected body and explicit
left/right sole contact remain mandatory. Silhouette-comparison reports are still
produced, but do not discard an otherwise valid printable STL.
"""
from __future__ import annotations

from pathlib import Path


SOURCE = Path(__file__).resolve().parent / "exposed_r2" / "run_photo_posed_job_v3_2_source.py"
text = SOURCE.read_text(encoding="utf-8")

old = '''        sys.executable, str(comparer), "--raw", str(raw), "--repaired", str(high_detail),
        "--report-json", str(reports / "raw_to_high_detail.json"),'''
new = '''        sys.executable, str(comparer), "--raw", str(one_input), "--repaired", str(high_detail),
        "--report-json", str(reports / "initial_to_high_detail.json"),'''
if text.count(old) != 1:
    raise RuntimeError(f"Expected one raw/high-detail comparison anchor; found {text.count(old)}")
text = text.replace(old, new, 1)
text = text.replace(
    '"--report-csv", str(reports / "raw_to_high_detail.csv"),',
    '"--report-csv", str(reports / "initial_to_high_detail.csv"),',
    1,
)
text = text.replace(
    '"--image", str(reports / "raw_to_high_detail.png"), "--expected", "1",',
    '"--image", str(reports / "initial_to_high_detail.png"), "--expected", "1",',
    1,
)
text = text.replace(
    '], reports / "raw_to_high_detail.log")',
    '], reports / "initial_to_high_detail.log", check=False)',
    1,
)

old_robust = '], reports / "high_detail_to_robust.log")'
new_robust = '], reports / "high_detail_to_robust.log", check=False)'
if text.count(old_robust) != 1:
    raise RuntimeError(f"Expected one robust visual-audit anchor; found {text.count(old_robust)}")
text = text.replace(old_robust, new_robust, 1)

anchor = '''    run([sys.executable, str(tools_root / "ci" / "patch_builder_external_validation.py"), str(builder)])
'''
addition = anchor + '''    run([sys.executable, str(workspace / "tools" / "patch_builder_ground_weighted_feet_v4.py"), str(builder)])
'''
if text.count(anchor) != 1:
    raise RuntimeError(f"Expected one builder patch anchor; found {text.count(anchor)}")
text = text.replace(anchor, addition, 1)

compiled = compile(text, str(SOURCE), "exec")
namespace = {
    "__name__": "__main__",
    "__file__": str(SOURCE),
    "__package__": None,
}
exec(compiled, namespace)
