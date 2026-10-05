#!/usr/bin/env python3
"""Patch the HOTS M3-to-STL builder for one photo-selected skin mesh."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def source_key_from_model_path(model_path: str) -> str:
    stem = Path(model_path).stem.casefold()
    stem = re.sub(r'^storm_hero_', '', stem)
    stem = re.sub(r'_v[0-9]+$', '', stem)
    return stem


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--builder', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--hero', required=True)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    selected = plan['heroes'][args.hero]
    source_key = source_key_from_model_path(selected['model_path'])

    text = args.builder.read_text(encoding='utf-8')
    mapping = (
        'HERO_SOURCE_CANDIDATES: Dict[str, Sequence[str]] = {\n'
        f'    {args.hero!r}: ({source_key!r},),\n'
        '}'
    )
    text, count = re.subn(
        r'HERO_SOURCE_CANDIDATES: Dict\[str, Sequence\[str\]\] = \{.*?\n\}',
        mapping,
        text,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise RuntimeError(f'Could not replace hero source mapping; matches={count}')

    old_patterns = [
        'pattern = re.compile(r"^storm_hero_(?P<key>[a-z0-9_]+)_base(?:_v[0-9]+)?\\.m3$", re.I)',
        'pattern = re.compile(r"^storm_hero_(?P<key>[a-z0-9_]+)_base\\.m3$", re.I)',
    ]
    new_pattern = 'pattern = re.compile(r"^storm_hero_(?P<key>[a-z0-9_]+?)(?:_v[0-9]+)?\\.m3$", re.I)'
    replaced = False
    for old in old_patterns:
        if old in text:
            text = text.replace(old, new_pattern, 1)
            replaced = True
            break
    if not replaced:
        raise RuntimeError('Could not broaden source model discovery pattern')

    # Prefer the exact selected path if duplicate internal keys exist. The selected
    # repository/path is passed through an environment variable by the workflow.
    insertion = '''    selected_relative = os.environ.get("HOTS_SELECTED_MODEL_PATH", "").replace("\\\\", "/").casefold()\n'''
    marker = '    for paths in result.values():\n'
    if marker not in text:
        raise RuntimeError('Could not locate source ordering block')
    text = text.replace(marker, insertion + marker, 1)
    old_sort = '        paths.sort(key=lambda p: (p.parent.name.lower() != p.stem.lower(), len(p.as_posix()), p.as_posix()))'
    new_sort = '''        paths.sort(key=lambda p: (\n            0 if selected_relative and p.as_posix().casefold().endswith(selected_relative) else 1,\n            p.parent.name.lower() != p.stem.lower(),\n            len(p.as_posix()),\n            p.as_posix(),\n        ))'''
    if old_sort not in text:
        raise RuntimeError('Could not patch source ordering')
    text = text.replace(old_sort, new_sort, 1)

    args.builder.write_text(text, encoding='utf-8')
    report = {
        'hero': args.hero,
        'skin_key': selected.get('skin_key'),
        'directory': selected.get('directory'),
        'repository': selected['repository'],
        'model_path': selected['model_path'],
        'source_key': source_key,
        'selection_method': selected.get('selection_method'),
        'classification': selected.get('classification'),
        'reference_images': selected.get('reference_images', []),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
