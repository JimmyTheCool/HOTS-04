#!/usr/bin/env python3
"""Build one exact photo-selected HOTS M3 model inside Blender.

This script bypasses fuzzy roster/source discovery. The workflow supplies the
exact repository path selected from the four-view photo analysis, then this
script calls the proven HOTS-03 print-preparation pipeline directly.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import traceback


def blender_argv() -> list[str]:
    argv = sys.argv
    return argv[argv.index('--') + 1:] if '--' in argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--builder', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--hero', required=True)
    parser.add_argument('--skin-key', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--target-height-mm', type=float, default=120.0)
    parser.add_argument('--base-height-mm', type=float, default=4.5)
    parser.add_argument('--base-margin-mm', type=float, default=5.0)
    parser.add_argument('--voxel-mm', type=float, default=0.42)
    parser.add_argument('--thin-part-mm', type=float, default=0.80)
    parser.add_argument('--max-triangles', type=int, default=600000)
    parser.add_argument('--skip-animation', action='store_true')
    parser.add_argument('--keep-blend', action='store_true')
    return parser.parse_args(blender_argv())


def safe_name(value: str) -> str:
    value = re.sub(r'[^A-Za-z0-9._-]+', '_', value.strip())
    value = re.sub(r'_+', '_', value).strip('_.')
    return value or 'hero'


def source_key(model: Path) -> str:
    value = model.stem.casefold()
    value = re.sub(r'^storm_hero_', '', value)
    return re.sub(r'_v[0-9]+$', '', value)


def load_builder(path: Path):
    spec = importlib.util.spec_from_file_location('hots_selected_builder', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load builder module: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    args.builder = args.builder.resolve()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    if not args.builder.is_file():
        raise FileNotFoundError(args.builder)
    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    hero_safe = safe_name(args.hero)
    args.output.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output / 'Raw'
    initial_dir = args.output / 'Initial'
    reports_dir = args.output / 'Reports'
    raw_dir.mkdir(parents=True, exist_ok=True)
    initial_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    builder = load_builder(args.builder)
    builder_args = argparse.Namespace(
        target_height_mm=args.target_height_mm,
        base_height_mm=args.base_height_mm,
        base_margin_mm=args.base_margin_mm,
        voxel_mm=args.voxel_mm,
        thin_part_mm=args.thin_part_mm,
        max_triangles=args.max_triangles,
        skip_animation=args.skip_animation,
        keep_blend=args.keep_blend,
    )

    report = {
        'hero': args.hero,
        'hero_safe': hero_safe,
        'skin_key': args.skin_key,
        'model_path': str(args.model),
        'source_key': source_key(args.model),
        'settings': vars(builder_args),
    }
    try:
        builder.enable_m3_addon()
        result = builder.process_source_group(
            report['source_key'],
            [args.model],
            args.output,
            builder_args,
        )
        detail_source = Path(result['detail']['path'])
        initial_source = Path(result['print_ready']['path'])
        detail_target = raw_dir / f'{hero_safe}.stl'
        initial_target = initial_dir / f'{hero_safe}.stl'
        shutil.copy2(detail_source, detail_target)
        shutil.copy2(initial_source, initial_target)
        report.update({
            'status': 'success',
            'detail_stl': str(detail_target),
            'initial_print_stl': str(initial_target),
            'builder_result': result,
        })
        print(json.dumps(report, indent=2))
    except Exception as exc:
        report.update({
            'status': 'failed',
            'error': str(exc),
            'traceback': traceback.format_exc(),
        })
        print(report['traceback'])
        (reports_dir / 'exact_source_build.json').write_text(
            json.dumps(report, indent=2), encoding='utf-8'
        )
        return 2

    (reports_dir / 'exact_source_build.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
