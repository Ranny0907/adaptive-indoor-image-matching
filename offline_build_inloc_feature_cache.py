#!/usr/bin/env python3
"""Build a persistent database SuperPoint feature cache for InLoc candidates."""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import torch

from adaptive_cascade_inloc import (
    DATASETS_DIR,
    ROOT,
    RESULTS_DIR,
    extract_superpoint_features,
    group_by_query,
    load_pairs,
)
from adaptive_feature_cache import (
    collect_database_images,
    feature_file_path,
    load_feature_file,
    normalized_path,
    save_feature_file,
    write_manifest,
)
from superglue_benchmark_lib import create_matching, load_and_prepare_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline-build a persistent SuperPoint cache for InLoc database images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pairs_csv",
        type=Path,
        default=DATASETS_DIR / "baseline" / "standardized_20260525" / "pairs" / "pairs_inloc_netvlad40.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=RESULTS_DIR / "adaptive_feature_cache" / "inloc_superpoint_top40",
    )
    parser.add_argument("--sg_topk", type=int, default=40)
    parser.add_argument("--warmup_pairs", type=int, default=5)
    parser.add_argument("--query_name", type=str, default=None, help="Optional query filter for demo-sized caches.")
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--descriptor_dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--superglue_weights", choices=["indoor", "outdoor"], default="indoor")
    parser.add_argument("--max_keypoints", type=int, default=1024)
    parser.add_argument("--keypoint_threshold", type=float, default=0.005)
    parser.add_argument("--nms_radius", type=int, default=4)
    parser.add_argument("--sinkhorn_iterations", type=int, default=20)
    parser.add_argument("--match_threshold", type=float, default=0.20)
    parser.add_argument("--print_every", type=int, default=100)
    return parser.parse_args()


def write_rows_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    sg_args = SimpleNamespace(
        nms_radius=args.nms_radius,
        keypoint_threshold=args.keypoint_threshold,
        max_keypoints=args.max_keypoints,
        sinkhorn_iterations=args.sinkhorn_iterations,
        match_threshold=args.match_threshold,
    )
    matching = create_matching(device, args.superglue_weights, sg_args)

    pairs = load_pairs(args.pairs_csv)
    formal_pairs = pairs[min(args.warmup_pairs, len(pairs)) :]
    grouped = group_by_query(formal_pairs, args.max_queries, args.query_name)
    db_images = collect_database_images(grouped, args.sg_topk)
    if args.max_images is not None:
        db_images = db_images[: args.max_images]

    print(f"Device: {device}")
    print(f"Pairs CSV: {args.pairs_csv}")
    print(f"Output dir: {args.output_dir}")
    print(f"Query filter: {args.query_name or 'all'}")
    print(f"SuperGlue Top-K cache scope: {args.sg_topk}")
    print(f"Database images to cache: {len(db_images)}")

    start = time.perf_counter()
    rows: List[Dict[str, object]] = []
    manifest_images: List[Dict[str, object]] = []
    extracted = 0
    reused = 0

    for index, image_path in enumerate(db_images, start=1):
        cache_path = feature_file_path(args.output_dir, image_path)
        status = "reused"
        runtime_ms = 0.0
        if args.overwrite or not cache_path.exists():
            image = load_and_prepare_image(image_path)
            features, runtime_ms = extract_superpoint_features(matching, device, image)
            save_feature_file(cache_path, features, args.descriptor_dtype)
            extracted += 1
            status = "extracted"
        else:
            features = load_feature_file(cache_path)
            reused += 1

        row = {
            "index": index,
            "status": status,
            "image_path": normalized_path(image_path),
            "feature_file": str(cache_path.relative_to(args.output_dir)),
            "num_keypoints": int(features.keypoints.shape[0]),
            "image_height": int(features.image_hw[0]),
            "image_width": int(features.image_hw[1]),
            "runtime_ms": runtime_ms,
        }
        rows.append(row)
        manifest_images.append({k: row[k] for k in ["image_path", "feature_file", "num_keypoints", "image_height", "image_width"]})

        if index == 1 or index == len(db_images) or (args.print_every > 0 and index % args.print_every == 0):
            print(f"[{index}/{len(db_images)}] {status} {image_path.name} keypoints={features.keypoints.shape[0]}")

    total_runtime_ms = (time.perf_counter() - start) * 1000.0
    manifest: Dict[str, object] = {
        "cache_type": "InLoc SuperPoint database feature cache",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pairs_csv": str(args.pairs_csv),
        "query_name": args.query_name,
        "sg_topk": args.sg_topk,
        "warmup_pairs": args.warmup_pairs,
        "num_images": len(manifest_images),
        "extracted_images": extracted,
        "reused_images": reused,
        "descriptor_dtype": args.descriptor_dtype,
        "device": device,
        "superglue_weights": args.superglue_weights,
        "superpoint_config": {
            "max_keypoints": args.max_keypoints,
            "keypoint_threshold": args.keypoint_threshold,
            "nms_radius": args.nms_radius,
        },
        "total_runtime_ms": total_runtime_ms,
        "images": manifest_images,
    }
    write_manifest(args.output_dir, manifest)
    write_rows_csv(args.output_dir / "cache_index.csv", rows)

    print("\n===== Feature Cache Summary =====")
    print(f"images={len(manifest_images)} extracted={extracted} reused={reused}")
    print(f"total_runtime_ms={total_runtime_ms:.1f}")
    print(f"manifest={args.output_dir / 'manifest.json'}")
    print(f"cache_index={args.output_dir / 'cache_index.csv'}")


if __name__ == "__main__":
    main()
