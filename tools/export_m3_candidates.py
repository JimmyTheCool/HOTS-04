#!/usr/bin/env python3
"""Blender-side batch exporter for skin candidate M3 geometry."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import traceback


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--builder', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--target-height-mm', type=float, default=120.0)
    parser.add_argument('--skip-animation', action='store_true')
    return parser.parse_args(argv)


def load_builder(path: Path):
    spec = importlib.util.spec_from_file_location('hots_candidate_builder', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to load builder module from {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    builder = load_builder(args.builder)
    entries = json.loads(args.manifest.read_text(encoding='utf-8'))
    report = {'target_height_mm': args.target_height_mm, 'entries': []}
    failures = 0

    builder.enable_m3_addon()
    for index, entry in enumerate(entries, start=1):
        result = dict(entry)
        try:
            model_path = Path(entry['absolute_model_path']).resolve()
            output_path = Path(entry['output_path']).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            source_key = entry['source_key']
            print(f"[{index}/{len(entries)}] Exporting {entry['hero']} / {entry['skin_key']}: {model_path}")

            builder.reset_scene()
            import_info = builder.import_source_group(
                [model_path],
                source_key,
                args.skip_animation,
            )
            mesh_objects = builder.evaluated_mesh_objects()
            if not mesh_objects:
                raise RuntimeError('M3 import created no printable evaluated mesh objects')

            baked = set(mesh_objects)
            import bpy
            for obj in list(bpy.context.scene.objects):
                if obj not in baked:
                    bpy.data.objects.remove(obj, do_unlink=True)
            builder.apply_world_transforms(mesh_objects)
            detail = builder.join_meshes(mesh_objects, f"DETAIL_{source_key}")
            builder.clean_mesh_object(detail, merge_distance=0.002)
            normalization = builder.normalize_to_height(detail, args.target_height_mm)
            builder.triangulate_object(detail)
            export = builder.write_binary_stl(detail, output_path, source_key)
            export['validation'] = builder.validate_binary_stl(output_path)
            result.update({
                'status': 'success',
                'import': import_info,
                'normalization': normalization,
                'export': export,
            })
        except Exception as exc:
            failures += 1
            result.update({
                'status': 'failed',
                'error': str(exc),
                'traceback': traceback.format_exc(),
            })
            print(result['traceback'])
        report['entries'].append(result)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'total': len(entries), 'failures': failures}, indent=2))
    return 0 if failures == 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())
