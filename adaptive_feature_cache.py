#!/usr/bin/env python3
"""Utilities for persistent SuperPoint feature caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from adaptive_cascade_inloc import InLocPair, SuperPointFeatures


FEATURES_DIR_NAME = "features"
MANIFEST_NAME = "manifest.json"


def normalized_path(path: Path) -> str:
    return str(path.resolve())


def feature_key(path: Path) -> str:
    return hashlib.sha1(normalized_path(path).lower().encode("utf-8")).hexdigest()


def feature_file_path(cache_dir: Path, image_path: Path) -> Path:
    return cache_dir / FEATURES_DIR_NAME / f"{feature_key(image_path)}.npz"


def manifest_path(cache_dir: Path) -> Path:
    return cache_dir / MANIFEST_NAME


def save_feature_file(path: Path, features: SuperPointFeatures, descriptor_dtype: str = "float32") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptors = features.descriptors.numpy()
    if descriptor_dtype == "float16":
        descriptors = descriptors.astype(np.float16)
    elif descriptor_dtype == "float32":
        descriptors = descriptors.astype(np.float32)
    else:
        raise ValueError("descriptor_dtype must be float32 or float16")

    np.savez_compressed(
        path,
        keypoints=features.keypoints.numpy().astype(np.float32),
        scores=features.scores.numpy().astype(np.float32),
        descriptors=descriptors,
        image_hw=np.array(features.image_hw, dtype=np.int32),
    )


def load_feature_file(path: Path) -> SuperPointFeatures:
    with np.load(path) as data:
        keypoints = torch.from_numpy(data["keypoints"].astype(np.float32))
        scores = torch.from_numpy(data["scores"].astype(np.float32))
        descriptors = torch.from_numpy(data["descriptors"].astype(np.float32))
        image_hw = tuple(int(v) for v in data["image_hw"].tolist())
    if len(image_hw) != 2:
        raise ValueError(f"Invalid image_hw in {path}: {image_hw}")
    return SuperPointFeatures(
        keypoints=keypoints,
        scores=scores,
        descriptors=descriptors,
        image_hw=(image_hw[0], image_hw[1]),
    )


def load_manifest(cache_dir: Path) -> Dict[str, object]:
    path = manifest_path(cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"Feature cache manifest not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest(cache_dir: Path, manifest: Dict[str, object]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path(cache_dir).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def load_feature_index(cache_dir: Path) -> Dict[str, Path]:
    manifest = load_manifest(cache_dir)
    images = manifest.get("images", [])
    if not isinstance(images, list):
        raise ValueError(f"Invalid manifest images field: {manifest_path(cache_dir)}")

    index: Dict[str, Path] = {}
    for item in images:
        if not isinstance(item, dict):
            continue
        image_path = item.get("image_path")
        feature_file = item.get("feature_file")
        if not isinstance(image_path, str) or not isinstance(feature_file, str):
            continue
        index[image_path] = cache_dir / feature_file
    return index


def collect_database_images(grouped_queries: Iterable[Tuple[str, List[InLocPair]]], sg_topk: int) -> List[Path]:
    output: List[Path] = []
    seen: set[Path] = set()
    for _, candidates in grouped_queries:
        for pair in candidates:
            if pair.rank > sg_topk or pair.image1 in seen:
                continue
            seen.add(pair.image1)
            output.append(pair.image1)
    return output

