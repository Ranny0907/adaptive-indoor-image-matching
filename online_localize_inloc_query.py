#!/usr/bin/env python3
"""Online single-query localization prototype for the adaptive InLoc matcher."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import kornia.feature as KF

from adaptive_cascade_inloc import (
    DATASETS_DIR,
    ROOT,
    RESULTS_DIR,
    InLocPair,
    finite_or_none,
    group_by_query,
    infer_pair_from_superpoint_features,
    load_pairs,
    run_loftr_pair,
    extract_superpoint_features,
)
from adaptive_feature_cache import load_feature_file, load_feature_index, load_manifest, normalized_path
from superglue_benchmark_lib import MAX_FINAL_MATCHES, create_matching, infer_pair, load_and_prepare_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one online InLoc query with persistent SuperPoint database cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pairs_csv",
        type=Path,
        default=DATASETS_DIR / "baseline" / "standardized_20260525" / "pairs" / "pairs_inloc_netvlad40.csv",
    )
    parser.add_argument("--query_name", type=str, default="IMG_0994.JPG")
    parser.add_argument(
        "--feature_cache_dir",
        type=Path,
        default=RESULTS_DIR / "adaptive_feature_cache" / "inloc_superpoint_top40",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=RESULTS_DIR / "adaptive_online_outputs",
    )
    parser.add_argument("--sg_topk", type=int, default=20)
    parser.add_argument("--loftr_topk", type=int, default=40)
    parser.add_argument("--warmup_pairs", type=int, default=5)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--allow_cache_miss", action="store_true", help="Extract missing database features online.")
    parser.add_argument("--superglue_weights", choices=["indoor", "outdoor"], default="indoor")
    parser.add_argument("--loftr_weights", choices=["indoor", "outdoor"], default="indoor")
    parser.add_argument("--max_keypoints", type=int, default=1024)
    parser.add_argument("--keypoint_threshold", type=float, default=0.005)
    parser.add_argument("--nms_radius", type=int, default=4)
    parser.add_argument("--sinkhorn_iterations", type=int, default=20)
    parser.add_argument("--match_threshold", type=float, default=0.20)
    parser.add_argument("--max_final_matches", type=int, default=MAX_FINAL_MATCHES)
    return parser.parse_args()


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def result_to_attempt(
    pair: InLocPair,
    stage: str,
    result,
    cache_source: str,
) -> Dict[str, object]:
    return {
        "stage": stage,
        "pair_id": pair.pair_id,
        "scene": pair.scene,
        "rank": pair.rank,
        "query_image": normalized_path(pair.image0),
        "candidate_image": normalized_path(pair.image1),
        "cache_source": cache_source,
        "num_matches": int(result.num_matches),
        "num_inliers": int(result.num_inliers),
        "inlier_ratio": float(result.inlier_ratio),
        "median_reproj_error": finite_or_none(result.median_reproj_error),
        "mean_reproj_error": finite_or_none(result.mean_reproj_error),
        "success": int(result.success),
        "runtime_ms": float(result.runtime_ms),
        "H_found": int(result.H_found),
    }


def select_query_candidates(pairs_csv: Path, warmup_pairs: int, query_name: str) -> List[InLocPair]:
    pairs = load_pairs(pairs_csv)
    formal_pairs = pairs[min(warmup_pairs, len(pairs)) :]
    grouped = dict(group_by_query(formal_pairs, None, None))
    if query_name not in grouped:
        available = ", ".join(list(grouped.keys())[:8])
        raise KeyError(f"query_name not found: {query_name}. examples: {available}")
    return grouped[query_name]


def load_cached_db_feature(
    pair: InLocPair,
    feature_index: Dict[str, Path],
    matching,
    device: str,
    allow_cache_miss: bool,
) -> Tuple[object, str, float]:
    image_key = normalized_path(pair.image1)
    cache_path = feature_index.get(image_key)
    if cache_path is not None and cache_path.exists():
        return load_feature_file(cache_path), "persistent_cache", 0.0

    if not allow_cache_miss:
        raise FileNotFoundError(
            "Missing cached feature for candidate image. "
            f"image={pair.image1} cache_dir missing entry. "
            "Build it with offline_build_inloc_feature_cache.py or pass --allow_cache_miss."
        )

    image = load_and_prepare_image(pair.image1)
    features, runtime_ms = extract_superpoint_features(matching, device, image)
    return features, "online_cache_miss", runtime_ms


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
    superglue = create_matching(device, args.superglue_weights, sg_args) if args.sg_topk > 0 else None
    loftr = KF.LoFTR(pretrained=args.loftr_weights).to(device).eval() if args.loftr_topk > 0 else None

    manifest = load_manifest(args.feature_cache_dir) if args.sg_topk > 0 else {}
    feature_index = load_feature_index(args.feature_cache_dir) if args.sg_topk > 0 else {}
    candidates = select_query_candidates(args.pairs_csv, args.warmup_pairs, args.query_name)
    scene = candidates[0].scene

    print(f"Device: {device}")
    print(f"Query: {args.query_name} scene={scene}")
    print(f"Feature cache: {args.feature_cache_dir}")
    print(f"Cached database images in manifest: {len(feature_index)}")
    print(f"Strategy: SG Top{args.sg_topk} -> LoFTR Top{args.loftr_topk}")

    start = time.perf_counter()
    attempts: List[Dict[str, object]] = []
    success = 0
    final_stage = "failed"
    final_rank: Optional[int] = None
    final_attempt: Optional[Dict[str, object]] = None
    sg_runtime_ms = 0.0
    loftr_runtime_ms = 0.0
    cache_miss_feature_runtime_ms = 0.0
    sg_attempts = 0
    loftr_attempts = 0

    query_image = load_and_prepare_image(candidates[0].image0)
    image0_cache = {candidates[0].image0: query_image}

    if superglue is not None:
        query_features, query_feature_runtime_ms = extract_superpoint_features(superglue, device, query_image)
        sg_runtime_ms += query_feature_runtime_ms
        for pair in [p for p in candidates if p.rank <= args.sg_topk]:
            db_features, cache_source, miss_runtime_ms = load_cached_db_feature(
                pair,
                feature_index,
                superglue,
                device,
                args.allow_cache_miss,
            )
            cache_miss_feature_runtime_ms += miss_runtime_ms
            sg_runtime_ms += miss_runtime_ms
            result, _ = infer_pair_from_superpoint_features(
                superglue,
                device,
                query_features,
                db_features,
                "inloc",
                args.max_final_matches,
            )
            sg_attempts += 1
            sg_runtime_ms += result.runtime_ms
            row = result_to_attempt(pair, "SuperGlue", result, cache_source)
            attempts.append(row)
            print(
                f"SG rank={pair.rank:02d} success={result.success} "
                f"matches={result.num_matches} inliers={result.num_inliers} time={result.runtime_ms:.1f} ms"
            )
            if result.success:
                success = 1
                final_stage = "SuperGlue"
                final_rank = pair.rank
                final_attempt = row
                break

    loftr_triggered = int(success == 0 and loftr is not None)
    if success == 0 and loftr is not None:
        for pair in [p for p in candidates if p.rank <= args.loftr_topk]:
            if pair.image0 not in image0_cache:
                image0_cache[pair.image0] = load_and_prepare_image(pair.image0)
            image1 = load_and_prepare_image(pair.image1)
            result = run_loftr_pair(loftr, device, image0_cache[pair.image0], image1, args.max_final_matches)
            loftr_attempts += 1
            loftr_runtime_ms += result.runtime_ms
            row = result_to_attempt(pair, "LoFTR", result, "not_used")
            attempts.append(row)
            print(
                f"LoFTR rank={pair.rank:02d} success={result.success} "
                f"matches={result.num_matches} inliers={result.num_inliers} time={result.runtime_ms:.1f} ms"
            )
            if result.success:
                success = 1
                final_stage = "LoFTR"
                final_rank = pair.rank
                final_attempt = row
                break

    total_wall_runtime_ms = (time.perf_counter() - start) * 1000.0
    if final_attempt is None and attempts:
        final_attempt = max(attempts, key=lambda row: (int(row["success"]), int(row["num_inliers"]), int(row["num_matches"])))

    result_json: Dict[str, object] = {
        "system": "adaptive_sparse_dense_inloc_localizer",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "query_name": args.query_name,
        "query_image": normalized_path(candidates[0].image0),
        "scene": scene,
        "strategy": {
            "superglue_topk": args.sg_topk,
            "loftr_topk": args.loftr_topk,
            "persistent_superpoint_cache": args.sg_topk > 0,
            "allow_cache_miss": args.allow_cache_miss,
        },
        "success": int(success),
        "final_stage": final_stage,
        "final_rank": final_rank,
        "loftr_triggered": loftr_triggered,
        "pose_estimation_ready": bool(success),
        "timing_ms": {
            "superglue_total": sg_runtime_ms,
            "loftr_total": loftr_runtime_ms,
            "cache_miss_feature_extraction": cache_miss_feature_runtime_ms,
            "online_model_total": sg_runtime_ms + loftr_runtime_ms,
            "wall_clock_total": total_wall_runtime_ms,
        },
        "attempt_counts": {
            "superglue": sg_attempts,
            "loftr": loftr_attempts,
            "total": len(attempts),
        },
        "feature_cache": {
            "cache_dir": str(args.feature_cache_dir),
            "manifest_num_images": len(feature_index),
            "manifest_query_name": manifest.get("query_name"),
            "manifest_sg_topk": manifest.get("sg_topk"),
        },
        "accepted_candidate": final_attempt,
        "attempts": attempts,
    }

    stem = Path(args.query_name).stem
    result_path = args.output_dir / f"{stem}_sg{args.sg_topk}_loftr{args.loftr_topk}_result.json"
    attempts_path = args.output_dir / f"{stem}_sg{args.sg_topk}_loftr{args.loftr_topk}_attempts.csv"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    write_csv(attempts_path, attempts)

    print("\n===== Online Localization Result =====")
    print(json.dumps({k: result_json[k] for k in ["query_name", "success", "final_stage", "final_rank", "timing_ms"]}, ensure_ascii=False, indent=2))
    print(f"result_json={result_path}")
    print(f"attempts_csv={attempts_path}")


if __name__ == "__main__":
    main()
