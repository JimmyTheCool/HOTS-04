#!/usr/bin/env python3
"""Classify HOTS-04 screenshot sets against available HOTS skin meshes.

The classifier combines four-view CLIP image similarity with BLIP-generated
captions. It only selects geometry families: material-only, shield-only,
hologram, and transformation helper models are folded into their parent skin.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import math
import re
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
)

ROOT = Path('.')
OUT = ROOT / 'reference_analysis'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Internal source names are geometry identifiers, not always public skin names.
# Descriptions are deliberately visual because STL selection depends on shape.
VISUAL_DESCRIPTIONS = {
    'base': 'the default original appearance and standard armour',
    'ultimate': 'the ornate master skin with upgraded elaborate armour',
    'dark18': 'a dark nexus cosmic purple black armoured appearance',
    'hunter': 'a rugged hunter themed armoured appearance',
    'craft20': 'a craft wars science fiction crossover appearance',
    'crimsoncount': 'a crimson vampire count appearance with gothic armour',
    'dragonbone': 'a skeletal dragon bone armour appearance with horns and bones',
    'guanyu': 'a Chinese Guan Yu warrior appearance with ornate eastern armour',
    'lichprince': 'a younger human lich prince appearance',
    'army19': 'a military army themed appearance',
    'felreaver': 'a demonic fel reaver mechanical appearance with horns',
    'altered20': 'an altered fate arcane mutant appearance with unusual magical armour',
    'corruptor': 'a heavily corrupted demonic appearance with spikes and monstrous armour',
    'pumpkin': 'a Halloween pumpkin headed harvest monster appearance',
    'mecha': 'a futuristic robotic mecha appearance',
    'toys18': 'a colourful plastic toy appearance',
    'vhaka': 'a savage dinosaur beast appearance',
    'glam': 'a glamorous rock star appearance with extravagant stage clothing',
    'marine': 'a StarCraft space marine power armour appearance',
    'purecountry': 'a country music cowboy appearance',
    'summer18': 'a summer beach appearance with swimwear and tropical accessories',
    'dark20': 'a dark nexus cosmic purple black futuristic appearance',
    'guardian': 'a celestial guardian armoured appearance with ornate protoss shapes',
    'janitor18': 'a janitor cleaner appearance carrying cleaning equipment',
    'kingvrykul': 'a massive vrykul king appearance with Nordic armour and crown',
    'vrykul': 'a vrykul barbarian appearance with Nordic armour',
    'space': 'a futuristic space lord appearance with science fiction armour',
    'apoth': 'an apothecary plague doctor medical appearance',
    'enforcer': 'a heavily armoured futuristic enforcer appearance',
    'love': 'a pink heart themed love doctor appearance',
    'sentinel': 'a night elf sentinel appearance with dark moon armour',
    'warden': 'a dark armoured warden appearance',
    'winter': 'a winter frost appearance with icy antlers and cold weather details',
    'pilothawk': 'a pilot hawk aviator appearance with flight gear',
    'summer20': 'a humorous summer beach appearance',
    'madaxe18': 'a post apocalyptic mad axe warrior appearance',
    'novazon': 'an Amazon warrior appearance with spear and tribal armour',
    'rollerderby': 'a roller derby athlete appearance with skates and sports gear',
    'spectre': 'a dark stealth spectre operative appearance',
    'widow': 'a Widowmaker inspired sniper appearance',
    'lil': 'a small cute miniature chibi version',
    'frost': 'a frost covered northern hunter appearance',
    'army18': 'a military army officer appearance',
    'pirate': 'a pirate captain appearance',
    'infested': 'an infested zerg mutant appearance with organic growths',
    'prisoner': 'a rugged prisoner appearance with reduced armour',
    'watergun': 'a summer water gun appearance with beach equipment',
    'cryptqueen': 'an undead crypt queen appearance with skeletal armour',
    'insectoid': 'an insectoid alien appearance with hard shell armour',
    'luxqueen': 'an elegant luxury queen appearance with ornate royal armour',
    'fire': 'a fiery tribal warrior appearance',
    'lunar': 'a lunar festival eastern warrior appearance',
}

REMOVE_SUFFIXES = (
    '_replace_mat', '_ghost_mat', '_shadow_mat', '_shield', '_holo',
    '_morph_odin_birth', '_morph_odin', '_short',
)

HERO_ALIASES = {
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


def canonical_directory(directory: str) -> str:
    value = directory.casefold()
    for suffix in REMOVE_SUFFIXES:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return value


def skin_key(hero: str, directory: str) -> str:
    value = canonical_directory(directory)
    value = re.sub(r'^storm_hero_', '', value)
    for alias in sorted(HERO_ALIASES[hero], key=len, reverse=True):
        if value == alias:
            return 'base'
        if value.startswith(alias + '_'):
            value = value[len(alias) + 1:]
            break
    return value or 'base'


def humanise(value: str) -> str:
    value = re.sub(r'(?<=\D)(\d{2})$', r' \1', value)
    value = value.replace('_', ' ').replace('-', ' ')
    return re.sub(r'\s+', ' ', value).strip()


def candidate_prompts(hero: str, key: str) -> list[str]:
    description = VISUAL_DESCRIPTIONS.get(key, humanise(key) + ' themed appearance')
    display_hero = {
        'ETC': 'E.T.C. the tauren rock musician',
        'Lt Morales': 'Lieutenant Morales the StarCraft medic',
        'Cho-Gall': "Cho'Gall the two-headed ogre",
    }.get(hero, hero)
    return [
        f'a full body game character screenshot of {display_hero} wearing {description}',
        f'{display_hero} from Heroes of the Storm, {description}',
        f'a 3D video game model of {display_hero}, {description}',
        f'the {humanise(key)} skin for {display_hero} in Heroes of the Storm',
        f'{display_hero}, silhouette and costume showing {description}',
    ]


def softmax(values: np.ndarray, temperature: float = 0.025) -> np.ndarray:
    scaled = values / temperature
    scaled -= scaled.max()
    exp = np.exp(scaled)
    return exp / exp.sum()


def main() -> None:
    inventory = json.loads((OUT / 'inventory.json').read_text(encoding='utf-8'))
    source = json.loads((OUT / 'skin_candidates.json').read_text(encoding='utf-8'))

    # Fold material/effect duplicates into a single geometry family.
    families: dict[str, list[dict[str, Any]]] = {}
    for hero_entry in inventory['heroes']:
        hero = hero_entry['hero']
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in source['heroes'].get(hero, []):
            grouped[canonical_directory(candidate['directory'])].append(candidate)
        hero_families = []
        for canonical, items in grouped.items():
            # Prefer the plain model, not material/effect helper variants.
            preferred = sorted(
                items,
                key=lambda item: (
                    any(token in item['directory'].casefold() for token in REMOVE_SUFFIXES),
                    len(item['model_path']),
                    item['model_path'].casefold(),
                ),
            )[0]
            key = skin_key(hero, canonical)
            hero_families.append({
                'skin_key': key,
                'canonical_directory': canonical,
                'repository': preferred['repository'],
                'model_path': preferred['model_path'],
                'source_variants': items,
                'prompts': candidate_prompts(hero, key),
            })
        families[hero] = sorted(hero_families, key=lambda item: item['skin_key'])

    print(f'Loading CLIP on {DEVICE}...')
    clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(DEVICE).eval()

    captions_available = True
    try:
        print('Loading BLIP captioner...')
        blip_processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
        blip_model = BlipForConditionalGeneration.from_pretrained(
            'Salesforce/blip-image-captioning-base'
        ).to(DEVICE).eval()
    except Exception as exc:  # CLIP remains authoritative if BLIP is unavailable.
        print(f'BLIP unavailable: {exc}')
        captions_available = False
        blip_processor = None
        blip_model = None

    results: dict[str, Any] = {
        'model': 'openai/clip-vit-base-patch32',
        'caption_model': 'Salesforce/blip-image-captioning-base' if captions_available else None,
        'device': DEVICE,
        'heroes': {},
    }

    for hero_entry in inventory['heroes']:
        hero = hero_entry['hero']
        images = [Image.open(ROOT / name).convert('RGB') for name in hero_entry['images']]
        captions = []
        if captions_available and blip_processor and blip_model:
            for image in images:
                inputs = blip_processor(images=image, return_tensors='pt').to(DEVICE)
                with torch.inference_mode():
                    generated = blip_model.generate(
                        **inputs,
                        max_new_tokens=35,
                        num_beams=5,
                    )
                captions.append(blip_processor.decode(generated[0], skip_special_tokens=True))

        candidate_families = families[hero]
        all_prompts = [prompt for family in candidate_families for prompt in family['prompts']]
        image_inputs = clip_processor(images=images, return_tensors='pt').to(DEVICE)
        text_inputs = clip_processor(
            text=all_prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
        ).to(DEVICE)
        with torch.inference_mode():
            image_features = clip_model.get_image_features(**image_inputs)
            text_features = clip_model.get_text_features(**text_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_mean = image_features.mean(dim=0, keepdim=True)
        image_mean = image_mean / image_mean.norm(dim=-1, keepdim=True)

        caption_feature = None
        if captions:
            caption_inputs = clip_processor(
                text=captions,
                return_tensors='pt',
                padding=True,
                truncation=True,
            ).to(DEVICE)
            with torch.inference_mode():
                caption_feature = clip_model.get_text_features(**caption_inputs)
            caption_feature = caption_feature / caption_feature.norm(dim=-1, keepdim=True)
            caption_feature = caption_feature.mean(dim=0, keepdim=True)
            caption_feature = caption_feature / caption_feature.norm(dim=-1, keepdim=True)

        rows = []
        prompt_offset = 0
        for family in candidate_families:
            count = len(family['prompts'])
            family_text = text_features[prompt_offset: prompt_offset + count]
            prompt_offset += count
            family_text_mean = family_text.mean(dim=0, keepdim=True)
            family_text_mean = family_text_mean / family_text_mean.norm(dim=-1, keepdim=True)
            image_score = float((image_mean @ family_text_mean.T).item())
            view_scores = (image_features @ family_text_mean.T).squeeze(1).cpu().tolist()
            caption_score = (
                float((caption_feature @ family_text_mean.T).item())
                if caption_feature is not None else image_score
            )
            final_score = 0.86 * image_score + 0.14 * caption_score
            rows.append({
                **{k: v for k, v in family.items() if k != 'prompts'},
                'prompts': family['prompts'],
                'image_cosine': image_score,
                'caption_cosine': caption_score,
                'combined_score': final_score,
                'per_view_image_cosine': view_scores,
            })

        rows.sort(key=lambda row: row['combined_score'], reverse=True)
        probabilities = softmax(np.array([row['combined_score'] for row in rows], dtype=float))
        for row, probability in zip(rows, probabilities):
            row['probability'] = float(probability)
        margin = rows[0]['combined_score'] - rows[1]['combined_score'] if len(rows) > 1 else 1.0
        entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())
        results['heroes'][hero] = {
            'images': hero_entry['images'],
            'captions': captions,
            'recommended_skin_key': rows[0]['skin_key'],
            'recommended_directory': rows[0]['canonical_directory'],
            'recommended_repository': rows[0]['repository'],
            'recommended_model_path': rows[0]['model_path'],
            'score_margin': float(margin),
            'probability_entropy': entropy,
            'confidence': (
                'high' if margin >= 0.025 else
                'medium' if margin >= 0.010 else
                'low'
            ),
            'ranked_candidates': rows,
        }
        print(hero, '->', rows[0]['skin_key'], f'margin={margin:.4f}', captions)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'skin_match_clip.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    with (OUT / 'skin_match_clip.tsv').open('w', encoding='utf-8') as handle:
        handle.write('Hero\tSelected Skin Key\tDirectory\tRepository\tModel Path\tConfidence\tMargin\tCaptions\tTop Candidates\n')
        for hero, result in results['heroes'].items():
            top = ' | '.join(
                f"{row['skin_key']}={row['combined_score']:.5f}"
                for row in result['ranked_candidates'][:5]
            )
            handle.write(
                f"{hero}\t{result['recommended_skin_key']}\t{result['recommended_directory']}\t"
                f"{result['recommended_repository']}\t{result['recommended_model_path']}\t"
                f"{result['confidence']}\t{result['score_margin']:.6f}\t"
                f"{' | '.join(result['captions'])}\t{top}\n"
            )


if __name__ == '__main__':
    main()
