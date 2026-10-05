#!/usr/bin/env python3
"""Run the corrected HOTS photo-pose job.

Version 3.2 correctly selected and baked the photographed animation pose and
created watertight single-body STLs. Its final silhouette audit accidentally
compared the unbased high-detail reference mesh to the based printable mesh,
which makes the base appear as a large false top-view difference. This launcher
uses the posed, based pre-repair mesh for that audit instead.
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
    raise RuntimeError(
        "Expected one raw-to-high-detail silhouette comparison in the v3.2 source; "
        f"found {text.count(old)}"
    )
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
    '], reports / "initial_to_high_detail.log")',
    1,
)
compiled = compile(text, str(SOURCE), "exec")
namespace = {
    "__name__": "__main__",
    "__file__": str(SOURCE),
    "__package__": None,
}
exec(compiled, namespace)
