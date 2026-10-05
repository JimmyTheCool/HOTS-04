#!/usr/bin/env python3
"""Combine multi-view appearance and geometry reports into a build plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip', required=True, type=Path)
    parser.add_argument('--inventory', required=True, type=Path)
    parser.add_argument('--geometry-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    clip = json.loads(args.clip.read_text(encoding='utf-8'))
    inventory = json.loads(args.inventory.read_text(encoding='utf-8'))
    image_map = {entry['hero']: entry['images'] for entry in inventory['heroes']}
    plan = {
        'hero_count': len(image_map),
        'selection_policy': (
            'Multi-angle geometry verification where available; otherwise four-view '
            'CLIP appearance classification with BLIP captions.'
        ),
        'heroes': {},
    }

    for hero in image_map:
        result = clip['heroes'][hero]
        selected_key = result['recommended_skin_key']
        method = 'four-view CLIP appearance classification'
        geometry_path = args.geometry_dir / f'{hero}.json'
        geometry = None
        if geometry_path.exists():
            geometry = json.loads(geometry_path.read_text(encoding='utf-8'))
            selected_key = geometry['selected_skin_key']
            method = geometry['selection_method']

        ranked = result['ranked_candidates']
        matches = [row for row in ranked if row['skin_key'] == selected_key]
        if not matches:
            raise RuntimeError(f'No source mesh for {hero} skin key {selected_key}')
        selected = matches[0]
        plan['heroes'][hero] = {
            'skin_key': selected_key,
            'directory': selected['canonical_directory'],
            'repository': selected['repository'],
            'model_path': selected['model_path'],
            'selection_method': method,
            'reference_images': image_map[hero],
            'classification': {
                'confidence': result['confidence'],
                'score_margin': result['score_margin'],
                'captions': result['captions'],
                'top_candidates': [
                    {
                        'skin_key': row['skin_key'],
                        'combined_score': row['combined_score'],
                        'probability': row['probability'],
                    }
                    for row in ranked[:5]
                ],
            },
            'geometry_verification': geometry,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps({
        hero: selected['skin_key']
        for hero, selected in plan['heroes'].items()
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
