#!/usr/bin/env python3
"""Create a Blender batch manifest for one or more HOTS heroes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

ALIASES = {
    'Alarak': ('alarak',),
    'Arthas': ('arthas',),
    'Blaze': ('firebat', 'blaze'),
    'Cho-Gall': ('chogall',),
    'Dehaka': ('dehaka',),
    'ETC': ('etc',),
    'Fenix': ('fenix',),
    'Leoric': ('kingleoric', 'leoric'),
    'Lt Morales': ('medic', 'morales'),
    'Lunara': ('dryad', 'lunara'),
    "Mal'Ganis": ('malganis',),
    'Nova': ('nova',),
    'Ragnaros': ('ragnaros',),
    'Rexxar': ('rexxar',),
    'Stukov': ('stukov',),
    'Tychus': ('tychus',),
    'Zagara': ('zagara',),
    "Zul'jin": ('zuljin',),
}

HELPER_TOKENS = (
    '_replace_mat', '_ghost_mat', '_shadow_mat', '_shield', '_holo',
    '_morph_', '_birth', '_short',
)


def safe_name(value: str) -> str:
    value = re.sub(r'[^A-Za-z0-9._-]+', '_', value.strip())
    return re.sub(r'_+', '_', value).strip('_.') or 'hero'


def source_key(model_path: str) -> str:
    value = Path(model_path).stem.casefold()
    value = re.sub(r'^storm_hero_', '', value)
    return re.sub(r'_v[0-9]+$', '', value)


def skin_key(hero: str, directory: str) -> str:
    value = re.sub(r'^storm_hero_', '', directory.casefold())
    for alias in sorted(ALIASES[hero], key=len, reverse=True):
        if value == alias:
            return 'base'
        if value.startswith(alias + '_'):
            value = value[len(alias) + 1:]
            break
    return value or 'base'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True, type=Path)
    parser.add_argument('--heroes', required=True, help='Pipe-separated display names')
    parser.add_argument('--hots02', required=True, type=Path)
    parser.add_argument('--hots03', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args()

    data = json.loads(args.candidates.read_text(encoding='utf-8'))['heroes']
    entries = []
    for hero in args.heroes.split('|'):
        seen_geometry = set()
        for item in data[hero]:
            directory_lower = item['directory'].casefold()
            if any(token in directory_lower for token in HELPER_TOKENS):
                continue
            key = skin_key(hero, item['directory'])
            # Internal suffixes such as material helpers should already be excluded;
            # this protects against duplicate files with the same geometry family.
            family = re.sub(r'_(replace_mat|ghost_mat|shadow_mat|shield|holo)$', '', directory_lower)
            if family in seen_geometry:
                continue
            seen_geometry.add(family)
            root = args.hots02 if item['repository'] == 'HOTS-02' else args.hots03
            absolute = (root / item['model_path']).resolve()
            if not absolute.exists():
                raise FileNotFoundError(absolute)
            output = args.output_root / safe_name(hero) / f'{key}.stl'
            entries.append({
                'hero': hero,
                'hero_safe': safe_name(hero),
                'skin_key': key,
                'directory': item['directory'],
                'repository': item['repository'],
                'model_path': item['model_path'],
                'absolute_model_path': str(absolute),
                'source_key': source_key(item['model_path']),
                'output_path': str(output.resolve()),
            })

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(entries, indent=2), encoding='utf-8')
    print(json.dumps({'heroes': args.heroes.split('|'), 'candidate_count': len(entries)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
