#!/usr/bin/env python3
"""Run an adaptive SuperGlue -> LoFTR cascade on InLoc pairs.

The script evaluates a retrieval-after-verification setting:
for each query image, run SuperPoint+SuperGlue on candidate images first.
If no candidate passes geometric verification within the configured Top-K,
fall back to LoFTR on the same candidate list and stop at the first success.

Runtime metrics follow the existing benchmark convention: image loading and
resize are excluded from per-method runtime; model inference plus RANSAC are
included.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import kornia.feature as KF


ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
METHODS_DIR = ROOT / "methods"
RESULTS_DIR = ROOT / "results"
SUPERGLUE_DIR = METHODS_DIR / "superglue"
if str(SUPERGLUE_DIR) not in sys.path:
    sys.path.insert(0, str(SUPERGLUE_DIR))

from superglue_benchmark_lib import (  # noqa: E402
    MAX_FINAL_MATCHES,
    PairResult,
    create_matching,
    frame_to_tensor,
    infer_pair,
    load_and_prepare_image,
)


RANSAC_REPROJ_THRESHOLD = 3.0
RANSAC_MAX_ITERS = 2000
RANSAC_CONFIDENCE = 0.995
INLOC_MIN_INLIERS = 15
INLOC_MAX_MEDIAN_ERROR = 5.0


@dataclass(frozen=True)
class InLocPair:
    pair_id: str
    scene: str
    image0: Path
    image1: Path
    rank: int
    retrieval_score: str = ""

    @property
    def query_name(self) -> str:
        return self.image0.name


@dataclass
class LoFTRResult:
    method: str
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    median_reproj_error: float
    mean_reproj_error: float
    success: int
    runtime_ms: float
    H_found: int


@dataclass
class SuperPointFeatures:
    keypoints: torch.Tensor
    scores: torch.Tensor
    descriptors: torch.Tensor
    image_hw: Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real SuperGlue -> LoFTR adaptive cascade on InLoc.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pairs_csv",
        type=Path,
        default=DATASETS_DIR / "baseline" / "standardized_20260525" / "pairs" / "pairs_inloc_netvlad40.csv",
        help="CSV with pair_id,scene,img0_path,img1_path,retrieval_rank.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=RESULTS_DIR / "adaptive_cascade_results" / "inloc_sg_loftr",
    )
    parser.add_argument("--sg_topk", type=int, default=40, help="Run SuperGlue up to this retrieval rank. Use 0 to skip.")
    parser.add_argument("--loftr_topk", type=int, default=40, help="Run LoFTR up to this retrieval rank after SG failure.")
    parser.add_argument("--warmup_pairs", type=int, default=5, help="Warm-up pairs excluded from formal statistics.")
    parser.add_argument("--max_queries", type=int, default=None, help="Limit formal query count for smoke tests.")
    parser.add_argument("--query_name", type=str, default=None, help="Run only one query image name, e.g. IMG_0994.JPG.")
    parser.add_argument("--print_every", type=int, default=1, help="Print one progress line every N queries. Use 0 for summary only.")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--superglue_weights", choices=["indoor", "outdoor"], default="indoor")
    parser.add_argument("--loftr_weights", choices=["indoor", "outdoor"], default="indoor")
    parser.add_argument(
        "--cache_superpoint_features",
        action="store_true",
        help=(
            "Precompute database SuperPoint features and extract each query feature once. "
            "This simulates an online localization system with an offline feature map."
        ),
    )
    parser.add_argument(
        "--loftr_policy",
        choices=["fixed_topk", "confidence_dispatch", "confidence_then_fixed"],
        default="fixed_topk",
        help=(
            "fixed_topk runs LoFTR over the configured Top-K after SuperGlue fails. "
            "confidence_dispatch runs LoFTR only for candidates selected by SuperGlue confidence cues. "
            "confidence_then_fixed adds a fixed Top-K fallback if confidence dispatch does not solve the query."
        ),
    )
    parser.add_argument("--policy_early_rank", type=int, default=5)
    parser.add_argument("--policy_min_matches", type=int, default=30)
    parser.add_argument("--policy_min_inliers", type=int, default=8)
    parser.add_argument("--policy_late_min_matches", type=int, default=50)
    parser.add_argument("--policy_late_min_inliers", type=int, default=8)
    parser.add_argument("--policy_max_loftr_attempts", type=int, default=5)
    parser.add_argument("--max_keypoints", type=int, default=1024)
    parser.add_argument("--keypoint_threshold", type=float, default=0.005)
    parser.add_argument("--nms_radius", type=int, default=4)
    parser.add_argument("--sinkhorn_iterations", type=int, default=20)
    parser.add_argument("--match_threshold", type=float, default=0.20)
    parser.add_argument("--max_final_matches", type=int, default=MAX_FINAL_MATCHES)
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        if path.exists():
            return path
        try:
            return resolve_path(str(path.relative_to(ROOT)))
        except ValueError:
            return path

    parts = path.parts
    candidates: List[Path] = []
    if parts:
        if parts[0] in {"data", "baseline"}:
            candidates.append(DATASETS_DIR / path)
        elif parts[0] in {"adaptive_cascade_results", "adaptive_demo_outputs"}:
            candidates.append(RESULTS_DIR / path)
    candidates.append(ROOT / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_pairs(pairs_csv: Path) -> List[InLocPair]:
    pairs: List[InLocPair] = []
    with pairs_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"pair_id", "scene", "img0_path", "img1_path", "retrieval_rank"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{pairs_csv} must contain columns: {sorted(required)}")
        for row in reader:
            pairs.append(
                InLocPair(
                    pair_id=row["pair_id"],
                    scene=row["scene"],
                    image0=resolve_path(row["img0_path"]),
                    image1=resolve_path(row["img1_path"]),
                    rank=int(row["retrieval_rank"]),
                    retrieval_score=row.get("retrieval_score", ""),
                )
            )
    if not pairs:
        raise RuntimeError(f"No pairs loaded from {pairs_csv}")
    return pairs


def group_by_query(
    pairs: Sequence[InLocPair],
    max_queries: Optional[int],
    query_name_filter: Optional[str],
) -> List[Tuple[str, List[InLocPair]]]:
    grouped: DefaultDict[str, List[InLocPair]] = defaultdict(list)
    order: List[str] = []
    for pair in pairs:
        if pair.query_name not in grouped:
            order.append(pair.query_name)
        grouped[pair.query_name].append(pair)

    output: List[Tuple[str, List[InLocPair]]] = []
    for query_name in order:
        if query_name_filter is not None and query_name != query_name_filter:
            continue
        candidates = sorted(grouped[query_name], key=lambda p: p.rank)
        output.append((query_name, candidates))
        if max_queries is not None and len(output) >= max_queries:
            break
    return output


def finite_or_none(value: float) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def compute_reprojection_errors(homography: np.ndarray, points0: np.ndarray, points1: np.ndarray) -> np.ndarray:
    projected = cv2.perspectiveTransform(
        points0.reshape(-1, 1, 2).astype(np.float32),
        homography.astype(np.float64),
    ).reshape(-1, 2)
    return np.linalg.norm(projected - points1.reshape(-1, 2).astype(np.float32), axis=1)


def extract_superpoint_features(
    matching,
    device: str,
    image: np.ndarray,
) -> Tuple[SuperPointFeatures, float]:
    timer_start = time.perf_counter()
    with torch.inference_mode():
        output = matching.superpoint({"image": frame_to_tensor(image, device)})
    runtime_ms = (time.perf_counter() - timer_start) * 1000.0
    return (
        SuperPointFeatures(
            keypoints=output["keypoints"][0].detach().cpu(),
            scores=output["scores"][0].detach().cpu(),
            descriptors=output["descriptors"][0].detach().cpu(),
            image_hw=image.shape[:2],
        ),
        runtime_ms,
    )


def feature_inputs(
    features: SuperPointFeatures,
    suffix: str,
    device: str,
) -> Dict[str, torch.Tensor]:
    height, width = features.image_hw
    return {
        f"image{suffix}": torch.empty((1, 1, height, width), dtype=torch.float32, device=device),
        f"keypoints{suffix}": features.keypoints.to(device)[None],
        f"scores{suffix}": features.scores.to(device)[None],
        f"descriptors{suffix}": features.descriptors.to(device)[None],
    }


def infer_pair_from_superpoint_features(
    matching,
    device: str,
    features0: SuperPointFeatures,
    features1: SuperPointFeatures,
    dataset: str,
    max_final_matches: int,
) -> Tuple[PairResult, Dict[str, np.ndarray]]:
    timer_start = time.perf_counter()
    data: Dict[str, torch.Tensor] = {}
    data.update(feature_inputs(features0, "0", device))
    data.update(feature_inputs(features1, "1", device))
    with torch.inference_mode():
        pred = matching(data)

    matches0 = pred["matches0"][0].detach().cpu().numpy()
    scores0 = pred["matching_scores0"][0].detach().cpu().numpy()
    kpts0 = features0.keypoints.numpy()
    kpts1 = features1.keypoints.numpy()

    valid = matches0 > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches0[valid]]
    mconf = scores0[valid]

    if len(mconf) > max_final_matches:
        keep = np.argsort(-mconf)[:max_final_matches]
        mkpts0 = mkpts0[keep]
        mkpts1 = mkpts1[keep]
        mconf = mconf[keep]

    estimated_homography = None
    mask = None
    inlier_mask = np.zeros((len(mkpts0),), dtype=bool)
    median_reproj_error = float("nan")
    mean_reproj_error = float("nan")

    if len(mkpts0) >= 4:
        estimated_homography, mask = cv2.findHomography(
            mkpts0,
            mkpts1,
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
            maxIters=RANSAC_MAX_ITERS,
            confidence=RANSAC_CONFIDENCE,
        )

    if estimated_homography is not None and mask is not None:
        inlier_mask = mask.reshape(-1).astype(bool)
        if np.any(inlier_mask):
            errors = compute_reprojection_errors(estimated_homography, mkpts0[inlier_mask], mkpts1[inlier_mask])
            median_reproj_error = float(np.median(errors))
            mean_reproj_error = float(np.mean(errors))

    runtime_ms = (time.perf_counter() - timer_start) * 1000.0
    num_matches = int(len(mkpts0))
    num_inliers = int(inlier_mask.sum())
    inlier_ratio = float(num_inliers / num_matches) if num_matches > 0 else 0.0
    success = int(
        estimated_homography is not None
        and num_inliers >= INLOC_MIN_INLIERS
        and not math.isnan(median_reproj_error)
        and median_reproj_error <= INLOC_MAX_MEDIAN_ERROR
    )

    result = PairResult(
        pair_id="",
        dataset=dataset,
        method="SuperPoint+SuperGlue",
        num_matches=num_matches,
        num_inliers=num_inliers,
        inlier_ratio=inlier_ratio,
        median_reproj_error=median_reproj_error,
        mean_reproj_error=mean_reproj_error,
        corner_error=float("nan"),
        success=success,
        runtime_ms=runtime_ms,
        H_found=int(estimated_homography is not None),
        H_gt_found=0,
    )
    aux = {
        "kpts0": kpts0,
        "kpts1": kpts1,
        "mkpts0": mkpts0,
        "mkpts1": mkpts1,
        "mconf": mconf,
        "inlier_mask": inlier_mask,
        "homography": estimated_homography,
        "eval_homography": None,
    }
    return result, aux


def precompute_database_features(
    matching,
    device: str,
    grouped_queries: Sequence[Tuple[str, List[InLocPair]]],
    sg_topk: int,
) -> Tuple[Dict[Path, SuperPointFeatures], float]:
    db_paths: List[Path] = []
    seen: set[Path] = set()
    for _, candidates in grouped_queries:
        for pair in candidates:
            if pair.rank > sg_topk or pair.image1 in seen:
                continue
            seen.add(pair.image1)
            db_paths.append(pair.image1)

    cache: Dict[Path, SuperPointFeatures] = {}
    total_runtime_ms = 0.0
    print(f"Precomputing SuperPoint features for {len(db_paths)} database images...")
    for index, path in enumerate(db_paths, start=1):
        image = load_and_prepare_image(path)
        features, runtime_ms = extract_superpoint_features(matching, device, image)
        cache[path] = features
        total_runtime_ms += runtime_ms
        if index == 1 or index == len(db_paths) or index % 100 == 0:
            print(f"  [{index}/{len(db_paths)}] cached {path.name} ({runtime_ms:.1f} ms)")

    return cache, total_runtime_ms


def run_loftr_pair(
    loftr: KF.LoFTR,
    device: str,
    image0: np.ndarray,
    image1: np.ndarray,
    max_final_matches: int,
) -> LoFTRResult:
    timer_start = time.perf_counter()
    with torch.inference_mode():
        batch = {
            "image0": frame_to_tensor(image0, device),
            "image1": frame_to_tensor(image1, device),
        }
        output = loftr(batch)

    result = output if isinstance(output, dict) else batch
    points0 = result["keypoints0"].detach().cpu().numpy()
    points1 = result["keypoints1"].detach().cpu().numpy()
    confidence = result["confidence"].detach().cpu().numpy()

    raw_matches = len(points0)
    if raw_matches == 0:
        runtime_ms = (time.perf_counter() - timer_start) * 1000.0
        return LoFTRResult("LoFTR", 0, 0, 0.0, float("nan"), float("nan"), 0, runtime_ms, 0)

    order = np.argsort(-confidence)
    points0 = points0[order]
    points1 = points1[order]
    if len(points0) > max_final_matches:
        points0 = points0[:max_final_matches]
        points1 = points1[:max_final_matches]

    homography = None
    mask = None
    if len(points0) >= 4:
        homography, mask = cv2.findHomography(
            points0.reshape(-1, 1, 2),
            points1.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
            maxIters=RANSAC_MAX_ITERS,
            confidence=RANSAC_CONFIDENCE,
        )

    num_matches = int(len(points0))
    num_inliers = 0
    median_reproj_error = float("nan")
    mean_reproj_error = float("nan")
    if homography is not None and mask is not None:
        inlier_mask = mask.reshape(-1).astype(bool)
        num_inliers = int(inlier_mask.sum())
        if num_inliers > 0:
            errors = compute_reprojection_errors(homography, points0[inlier_mask], points1[inlier_mask])
            median_reproj_error = float(np.median(errors))
            mean_reproj_error = float(np.mean(errors))

    runtime_ms = (time.perf_counter() - timer_start) * 1000.0
    inlier_ratio = float(num_inliers / num_matches) if num_matches else 0.0
    success = int(
        homography is not None
        and num_inliers >= INLOC_MIN_INLIERS
        and not math.isnan(median_reproj_error)
        and median_reproj_error <= INLOC_MAX_MEDIAN_ERROR
    )
    return LoFTRResult(
        method="LoFTR",
        num_matches=num_matches,
        num_inliers=num_inliers,
        inlier_ratio=inlier_ratio,
        median_reproj_error=median_reproj_error,
        mean_reproj_error=mean_reproj_error,
        success=success,
        runtime_ms=runtime_ms,
        H_found=int(homography is not None),
    )


def result_to_attempt_row(
    query_index: int,
    query_name: str,
    pair: InLocPair,
    stage: str,
    result: PairResult | LoFTRResult,
) -> Dict[str, object]:
    return {
        "query_index": query_index,
        "query_img": query_name,
        "pair_id": pair.pair_id,
        "scene": pair.scene,
        "rank": pair.rank,
        "stage": stage,
        "image0_path": str(pair.image0),
        "image1_path": str(pair.image1),
        "num_matches": result.num_matches,
        "num_inliers": result.num_inliers,
        "inlier_ratio": result.inlier_ratio,
        "median_reproj_error": finite_or_none(result.median_reproj_error),
        "mean_reproj_error": finite_or_none(result.mean_reproj_error),
        "success": result.success,
        "runtime_ms": result.runtime_ms,
        "H_found": result.H_found,
    }


def should_dispatch_loftr(pair: InLocPair, result: PairResult | LoFTRResult, args: argparse.Namespace) -> bool:
    if result.success:
        return False
    if pair.rank > args.loftr_topk:
        return False
    if pair.rank <= args.policy_early_rank:
        return bool(
            result.H_found
            or result.num_matches >= args.policy_min_matches
            or result.num_inliers >= args.policy_min_inliers
        )
    return bool(
        result.num_inliers >= args.policy_late_min_inliers
        or (result.H_found and result.num_matches >= args.policy_late_min_matches)
    )


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def summarize(query_rows: Sequence[Dict[str, object]], args: argparse.Namespace, device: str) -> Dict[str, object]:
    num_queries = len(query_rows)
    success_count = sum(int(row["success"]) for row in query_rows)
    loftr_trigger_count = sum(int(row["loftr_triggered"]) for row in query_rows)
    final_stage_counts = Counter(str(row["final_stage"]) for row in query_rows)
    return {
        "method": "SuperGlue->LoFTR adaptive cascade",
        "dataset": "InLoc",
        "num_queries": num_queries,
        "success_count": success_count,
        "success_rate": success_count / num_queries if num_queries else float("nan"),
        "avg_total_runtime_ms": mean(float(row["total_runtime_ms"]) for row in query_rows),
        "avg_sg_runtime_ms": mean(float(row["sg_runtime_ms"]) for row in query_rows),
        "avg_sg_query_feature_runtime_ms": mean(float(row["sg_query_feature_runtime_ms"]) for row in query_rows),
        "avg_sg_pair_runtime_ms": mean(float(row["sg_pair_runtime_ms"]) for row in query_rows),
        "avg_loftr_runtime_ms": mean(float(row["loftr_runtime_ms"]) for row in query_rows),
        "avg_sg_attempts": mean(float(row["sg_attempts"]) for row in query_rows),
        "avg_loftr_attempts": mean(float(row["loftr_attempts"]) for row in query_rows),
        "loftr_trigger_count": loftr_trigger_count,
        "loftr_trigger_rate": loftr_trigger_count / num_queries if num_queries else float("nan"),
        "final_stage_counts": dict(final_stage_counts),
        "device": device,
        "sg_topk": args.sg_topk,
        "loftr_topk": args.loftr_topk,
        "loftr_policy": args.loftr_policy,
        "policy_early_rank": args.policy_early_rank,
        "policy_min_matches": args.policy_min_matches,
        "policy_min_inliers": args.policy_min_inliers,
        "policy_late_min_matches": args.policy_late_min_matches,
        "policy_late_min_inliers": args.policy_late_min_inliers,
        "policy_max_loftr_attempts": args.policy_max_loftr_attempts,
        "cache_superpoint_features": args.cache_superpoint_features,
        "db_feature_cache_count": getattr(args, "db_feature_cache_count", 0),
        "db_feature_precompute_runtime_ms": getattr(args, "db_feature_precompute_runtime_ms", 0.0),
        "warmup_pairs": args.warmup_pairs,
        "max_queries": args.max_queries,
        "query_name": args.query_name,
        "pairs_csv": str(args.pairs_csv),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    print(f"Device: {device}")
    print(f"Pairs CSV: {args.pairs_csv}")
    print(f"Output dir: {args.output_dir}")
    print(f"SuperGlue Top-K before LoFTR: {args.sg_topk}")
    print(f"LoFTR fallback Top-K: {args.loftr_topk}")
    print(f"LoFTR policy: {args.loftr_policy}")
    print(f"SuperPoint feature cache: {args.cache_superpoint_features}")

    pairs = load_pairs(args.pairs_csv)
    warmup_count = min(args.warmup_pairs, len(pairs))
    warmup_pairs = pairs[:warmup_count]
    formal_pairs = pairs[warmup_count:]

    sg_args = SimpleNamespace(
        nms_radius=args.nms_radius,
        keypoint_threshold=args.keypoint_threshold,
        max_keypoints=args.max_keypoints,
        sinkhorn_iterations=args.sinkhorn_iterations,
        match_threshold=args.match_threshold,
    )
    superglue = create_matching(device, args.superglue_weights, sg_args) if args.sg_topk > 0 else None
    loftr = KF.LoFTR(pretrained=args.loftr_weights).to(device).eval() if args.loftr_topk > 0 else None

    print(f"Warm-up pairs: {warmup_count}")
    for pair in warmup_pairs:
        image0 = load_and_prepare_image(pair.image0)
        image1 = load_and_prepare_image(pair.image1)
        if superglue is not None:
            infer_pair(superglue, device, image0, image1, "inloc", args.max_final_matches)
        if loftr is not None:
            run_loftr_pair(loftr, device, image0, image1, args.max_final_matches)

    grouped_queries = group_by_query(formal_pairs, args.max_queries, args.query_name)
    print(f"Formal pairs: {len(formal_pairs)}")
    print(f"Formal queries: {len(grouped_queries)}")

    image0_cache: Dict[Path, np.ndarray] = {}
    db_feature_cache: Dict[Path, SuperPointFeatures] = {}
    args.db_feature_cache_count = 0
    args.db_feature_precompute_runtime_ms = 0.0
    if args.cache_superpoint_features and args.sg_topk > 0:
        assert superglue is not None
        db_feature_cache, args.db_feature_precompute_runtime_ms = precompute_database_features(
            superglue,
            device,
            grouped_queries,
            args.sg_topk,
        )
        args.db_feature_cache_count = len(db_feature_cache)

    attempt_rows: List[Dict[str, object]] = []
    query_rows: List[Dict[str, object]] = []

    for query_index, (query_name, candidates) in enumerate(grouped_queries, start=1):
        scene = Counter(pair.scene for pair in candidates).most_common(1)[0][0]
        success = 0
        final_stage = "failed"
        final_rank: Optional[int] = None
        sg_attempts = 0
        loftr_attempts = 0
        sg_runtime_ms = 0.0
        sg_query_feature_runtime_ms = 0.0
        sg_pair_runtime_ms = 0.0
        loftr_runtime_ms = 0.0
        tried_loftr_ranks: set[int] = set()

        def load_pair_images(pair: InLocPair) -> Tuple[np.ndarray, np.ndarray]:
            if pair.image0 not in image0_cache:
                image0_cache[pair.image0] = load_and_prepare_image(pair.image0)
            return image0_cache[pair.image0], load_and_prepare_image(pair.image1)

        if args.sg_topk > 0:
            assert superglue is not None
            query_features: Optional[SuperPointFeatures] = None
            if args.cache_superpoint_features:
                if candidates[0].image0 not in image0_cache:
                    image0_cache[candidates[0].image0] = load_and_prepare_image(candidates[0].image0)
                query_features, runtime_ms = extract_superpoint_features(
                    superglue,
                    device,
                    image0_cache[candidates[0].image0],
                )
                sg_query_feature_runtime_ms += runtime_ms
                sg_runtime_ms += runtime_ms

            for pair in [p for p in candidates if p.rank <= args.sg_topk]:
                if args.cache_superpoint_features:
                    assert query_features is not None
                    result, _ = infer_pair_from_superpoint_features(
                        superglue,
                        device,
                        query_features,
                        db_feature_cache[pair.image1],
                        "inloc",
                        args.max_final_matches,
                    )
                else:
                    image0, image1 = load_pair_images(pair)
                    result, _ = infer_pair(superglue, device, image0, image1, "inloc", args.max_final_matches)
                result.pair_id = pair.pair_id
                sg_attempts += 1
                sg_runtime_ms += result.runtime_ms
                sg_pair_runtime_ms += result.runtime_ms
                attempt_rows.append(result_to_attempt_row(query_index, query_name, pair, "SuperGlue", result))
                if result.success:
                    success = 1
                    final_stage = "SuperGlue"
                    final_rank = pair.rank
                    break
                if (
                    args.loftr_policy in {"confidence_dispatch", "confidence_then_fixed"}
                    and args.loftr_topk > 0
                    and loftr_attempts < args.policy_max_loftr_attempts
                    and should_dispatch_loftr(pair, result, args)
                ):
                    assert loftr is not None
                    image0, image1 = load_pair_images(pair)
                    loftr_result = run_loftr_pair(loftr, device, image0, image1, args.max_final_matches)
                    loftr_attempts += 1
                    tried_loftr_ranks.add(pair.rank)
                    loftr_runtime_ms += loftr_result.runtime_ms
                    attempt_rows.append(result_to_attempt_row(query_index, query_name, pair, "LoFTR", loftr_result))
                    if loftr_result.success:
                        success = 1
                        final_stage = "LoFTR"
                        final_rank = pair.rank
                        break

        if success == 0 and args.loftr_policy in {"fixed_topk", "confidence_then_fixed"} and args.loftr_topk > 0:
            assert loftr is not None
            for pair in [p for p in candidates if p.rank <= args.loftr_topk]:
                if pair.rank in tried_loftr_ranks:
                    continue
                image0, image1 = load_pair_images(pair)
                result = run_loftr_pair(loftr, device, image0, image1, args.max_final_matches)
                loftr_attempts += 1
                tried_loftr_ranks.add(pair.rank)
                loftr_runtime_ms += result.runtime_ms
                attempt_rows.append(result_to_attempt_row(query_index, query_name, pair, "LoFTR", result))
                if result.success:
                    success = 1
                    final_stage = "LoFTR"
                    final_rank = pair.rank
                    break
        loftr_triggered = int(loftr_attempts > 0)

        query_row: Dict[str, object] = {
            "query_index": query_index,
            "query_img": query_name,
            "scene": scene,
            "success": success,
            "final_stage": final_stage,
            "final_rank": final_rank,
            "loftr_triggered": loftr_triggered,
            "sg_attempts": sg_attempts,
            "loftr_attempts": loftr_attempts,
            "sg_runtime_ms": sg_runtime_ms,
            "sg_query_feature_runtime_ms": sg_query_feature_runtime_ms,
            "sg_pair_runtime_ms": sg_pair_runtime_ms,
            "loftr_runtime_ms": loftr_runtime_ms,
            "total_runtime_ms": sg_runtime_ms + loftr_runtime_ms,
            "num_candidates": len(candidates),
        }
        query_rows.append(query_row)

        if args.print_every > 0 and (query_index == 1 or query_index == len(grouped_queries) or query_index % args.print_every == 0):
            print(
                f"[{query_index}/{len(grouped_queries)}] {query_name} | "
                f"success={success} stage={final_stage} rank={final_rank} "
                f"sg_attempts={sg_attempts} loftr_attempts={loftr_attempts} "
                f"time={query_row['total_runtime_ms']:.1f} ms"
            )

    write_csv(args.output_dir / "adaptive_attempts.csv", attempt_rows)
    write_csv(args.output_dir / "adaptive_query_results.csv", query_rows)

    summary = summarize(query_rows, args, device)
    with (args.output_dir / "adaptive_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== Adaptive Cascade Summary =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
