#!/usr/bin/env python3
"""Resolve compatible HOTS M3A animation sources for a selected skin mesh.

The source repository is cloned with --no-checkout. This script inspects the Git
index without downloading every game asset, ranks available *requiredanims*.m3a
files against the selected model directory, and emits the paths that should be
included in sparse checkout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Iterable


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def key_from_directory(directory: str) -> str:
    key = PurePosixPath(directory).name.casefold()
    key = re.sub(r"^storm_hero_", "", key)
    return key


def animation_key(path: str) -> str:
    directory = PurePosixPath(path).parent.name.casefold()
    directory = re.sub(r"^storm_hero_", "", directory)
    directory = re.sub(r"_requiredanims(?:_[a-z0-9]+)?$", "", directory)
    return directory


def common_prefix_tokens(left: str, right: str) -> int:
    left_tokens = left.split("_")
    right_tokens = right.split("_")
    count = 0
    for a, b in zip(left_tokens, right_tokens):
        if a != b:
            break
        count += 1
    return count


def common_prefix_chars(left: str, right: str) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def score_candidate(model_key: str, path: str, size: int, hint: str | None) -> tuple[float, dict]:
    key = animation_key(path)
    model_tokens = set(model_key.split("_"))
    candidate_tokens = set(key.split("_"))
    token_overlap = len(model_tokens & candidate_tokens)
    token_union = max(1, len(model_tokens | candidate_tokens))
    jaccard = token_overlap / token_union
    prefix_tokens = common_prefix_tokens(model_key, key)
    prefix_chars = common_prefix_chars(model_key, key)

    score = 0.0
    reasons: list[str] = []
    if key == model_key:
        score += 20_000
        reasons.append("exact selected-skin animation key")
    if model_key.startswith(key + "_"):
        score += 8_000 + len(key) * 20
        reasons.append("generic animation key is a prefix of selected skin")
    if key.startswith(model_key + "_"):
        score += 6_000 + len(model_key) * 15
        reasons.append("animation key extends selected skin key")
    if prefix_tokens:
        score += prefix_tokens * 1_500
        reasons.append(f"{prefix_tokens} leading token(s) match")
    score += prefix_chars * 12
    score += jaccard * 800

    if hint:
        hint_cf = hint.casefold()
        if key == hint_cf:
            score += 15_000
            reasons.append("exact animation hint")
        elif key.startswith(hint_cf) or hint_cf.startswith(key):
            score += 4_000
            reasons.append("animation hint prefix match")

    lower = path.casefold()
    for bad in ("facial", "death", "ragdoll", "portrait_low", "low.m3a"):
        if bad in lower:
            score -= 20_000
            reasons.append(f"penalized {bad}")
    score += min(600.0, max(0.0, size) / 50_000.0)
    score -= abs(len(model_key) - len(key)) * 2.0
    return score, {
        "path": path,
        "directory": str(PurePosixPath(path).parent),
        "animation_key": key,
        "bytes": size,
        "score": round(score, 3),
        "reasons": reasons,
    }


def parse_tree(lines: Iterable[str]) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for raw in lines:
        raw = raw.rstrip("\n")
        if not raw or "\t" not in raw:
            continue
        metadata, path = raw.split("\t", 1)
        parts = metadata.split()
        size = 0
        if len(parts) >= 4 and parts[3].isdigit():
            size = int(parts[3])
        rows.append((size, path))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--hero", required=True)
    parser.add_argument("--animation-hint")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    repo = args.repo.resolve()
    model = args.model.replace("\\", "/").lstrip("/")
    rows = parse_tree(run_git(repo, "ls-tree", "-rl", "HEAD").splitlines())
    paths = {path for _, path in rows}
    if model not in paths:
        raise FileNotFoundError(f"Selected model does not exist in Git tree: {model}")

    model_dir = str(PurePosixPath(model).parent)
    model_key = key_from_directory(model_dir)
    ranked: list[dict] = []
    for size, path in rows:
        lower = path.casefold()
        if not lower.endswith(".m3a") or "requiredanims" not in lower:
            continue
        score, detail = score_candidate(model_key, path, size, args.animation_hint)
        ranked.append(detail)
    ranked.sort(key=lambda row: (-float(row["score"]), -int(row["bytes"]), str(row["path"])))

    chosen = ranked[0] if ranked and float(ranked[0]["score"]) > 0 else None
    compatible: list[dict] = []
    seen_directories: set[str] = set()
    if chosen:
        best_score = float(chosen["score"])
        for row in ranked:
            directory = str(row["directory"])
            if directory in seen_directories or float(row["score"]) <= 0:
                continue
            if len(compatible) >= 3 or (compatible and float(row["score"]) < best_score - 9000):
                break
            compatible.append(row)
            seen_directories.add(directory)

    sparse_directories = [model_dir]
    sparse_directories.extend(str(row["directory"]) for row in compatible)

    result = {
        "hero": args.hero,
        "model_path": model,
        "model_directory": model_dir,
        "model_key": model_key,
        "animation_hint": args.animation_hint,
        "animation_path": chosen["path"] if chosen else None,
        "animation_directory": chosen["directory"] if chosen else None,
        "animation_paths": [str(row["path"]) for row in compatible],
        "animation_directories": [str(row["directory"]) for row in compatible],
        "sparse_directories": list(dict.fromkeys(sparse_directories)),
        "top_animation_candidates": ranked[:12],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
