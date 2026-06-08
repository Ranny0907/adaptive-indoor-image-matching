#!/usr/bin/env python3
"""Shared utilities for SuperPoint + SuperGlue homography benchmarks."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.cm as cm
import numpy as np
import torch

from models.matching import Matching
from models.utils import make_matching_plot_fast

torch.set_grad_enabled(False)

RANSAC_REPROJ_THRESHOLD = 3.0
RANSAC_MAX_ITERS = 2000
RANSAC_CONFIDENCE = 0.995
MAX_FINAL_MATCHES = 1000


@dataclass
class PairItem:
    pair_id: str
    dataset: str
    image0: Path
    image1: Path
    gt_homography_path: Optional[Path] = None


@dataclass
class PairResult:
    pair_id: str
    dataset: str
    method: str
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    median_reproj_error: float
    mean_reproj_error: float
    corner_error: float
    success: int
    runtime_ms: float
    H_found: int
    H_gt_found: int


def resize_keep_ratio_long_side_640(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    scale = min(1.0, 640.0 / max(h, w))
    new_w = int(math.floor(w * scale))
    new_h = int(math.floor(h * scale))
    new_w = max(8, (new_w // 8) * 8)
    new_h = max(8, (new_h // 8) * 8)
    if new_w == w and new_h == h:
        return gray
    return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)


def frame_to_tensor(image: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(image / 255.0).float()[None, None].to(device)


def load_and_prepare_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return resize_keep_ratio_long_side_640(image)


def load_and_prepare_image_with_info(path: Path) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    original_hw = image.shape[:2]
    resized = resize_keep_ratio_long_side_640(image)
    resized_hw = resized.shape[:2]
    return resized, original_hw, resized_hw


def resolve_existing_image_stem(stem: Path) -> Path:
    for suffix in [".ppm", ".png", ".jpg", ".jpeg", ".bmp"]:
        candidate = stem.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find image for stem: {stem}")


def resolve_relative_or_absolute_path(path_str: str, *roots: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(f"Referenced image not found: {path}")

    for root in roots:
        candidate = root / path
        if candidate.exists():
            return candidate

    searched = ", ".join(str(root) for root in roots)
    raise FileNotFoundError(f"Referenced image not found: {path}. searched roots: {searched}")


def choose_weights(dataset: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "outdoor" if dataset == "hpatches" else "indoor"


def create_matching(device: str, weights: str, args) -> Matching:
    config = {
        "superpoint": {
            "nms_radius": args.nms_radius,
            "keypoint_threshold": args.keypoint_threshold,
            "max_keypoints": args.max_keypoints,
        },
        "superglue": {
            "weights": weights,
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "match_threshold": args.match_threshold,
        },
    }
    return Matching(config).eval().to(device)


def load_hpatches_homography(path: Path) -> np.ndarray:
    homography = np.loadtxt(path).astype(np.float64)
    if homography.shape != (3, 3):
        raise ValueError(f"Invalid homography shape in {path}: {homography.shape}")
    return homography


def compute_resize_matrix(original_hw: Tuple[int, int], resized_hw: Tuple[int, int]) -> np.ndarray:
    original_h, original_w = original_hw
    resized_h, resized_w = resized_hw
    scale_x = resized_w / float(original_w)
    scale_y = resized_h / float(original_h)
    return np.array(
        [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def scale_homography_to_resized_images(
    homography: np.ndarray,
    original0_hw: Tuple[int, int],
    resized0_hw: Tuple[int, int],
    original1_hw: Tuple[int, int],
    resized1_hw: Tuple[int, int],
) -> np.ndarray:
    s0 = compute_resize_matrix(original0_hw, resized0_hw)
    s1 = compute_resize_matrix(original1_hw, resized1_hw)
    return s1 @ homography @ np.linalg.inv(s0)


def compute_reprojection_errors(
    points0: np.ndarray,
    points1: np.ndarray,
    homography: np.ndarray,
) -> np.ndarray:
    projected = cv2.perspectiveTransform(
        points0.reshape(-1, 1, 2).astype(np.float32),
        homography.astype(np.float64),
    ).reshape(-1, 2)
    target = points1.reshape(-1, 2).astype(np.float32)
    return np.linalg.norm(projected - target, axis=1)


def compute_reprojection_errors_from_mask(
    homography: np.ndarray,
    points0: np.ndarray,
    points1: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float]:
    inlier_mask = mask.reshape(-1).astype(bool)
    if not np.any(inlier_mask):
        return float("nan"), float("nan")

    points0_in = points0[inlier_mask]
    points1_in = points1[inlier_mask]
    errors = compute_reprojection_errors(points0_in, points1_in, homography)
    return float(np.median(errors)), float(np.mean(errors))


def compute_corner_error(
    estimated_homography: np.ndarray,
    gt_homography: np.ndarray,
    src_shape: Tuple[int, int],
) -> float:
    h, w = src_shape
    corners = np.array(
        [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    try:
        est = cv2.perspectiveTransform(corners, estimated_homography.astype(np.float64))
        gt = cv2.perspectiveTransform(corners, gt_homography.astype(np.float64))
        return float(np.mean(np.linalg.norm((est - gt).reshape(-1, 2), axis=1)))
    except Exception:
        return float("nan")


def error_auc(errors: Sequence[float], thresholds: Sequence[float]) -> Dict[str, float]:
    clean = np.array([e for e in errors if not math.isnan(e) and math.isfinite(e)], dtype=np.float64)
    if clean.size == 0:
        return {f"auc@{int(t)}": float("nan") for t in thresholds}

    clean = np.sort(clean)
    recalls = (np.arange(clean.size, dtype=np.float64) + 1.0) / clean.size
    output: Dict[str, float] = {}
    for threshold in thresholds:
        last = np.searchsorted(clean, threshold, side="right")
        if last == 0:
            x = np.array([0.0, float(threshold)], dtype=np.float64)
            y = np.array([0.0, 0.0], dtype=np.float64)
        else:
            x = np.concatenate(([0.0], clean[:last], [float(threshold)]))
            y = np.concatenate(([0.0], recalls[:last], [recalls[last - 1]]))
        output[f"auc@{int(threshold)}"] = float(np.trapz(y, x) / float(threshold))
    return output


def infer_pair(
    matching: Matching,
    device: str,
    image0: np.ndarray,
    image1: np.ndarray,
    dataset: str,
    max_final_matches: int,
    eval_homography: Optional[np.ndarray] = None,
) -> Tuple[PairResult, Dict[str, np.ndarray]]:
    timer_start = time.perf_counter()
    pred = matching(
        {
            "image0": frame_to_tensor(image0, device),
            "image1": frame_to_tensor(image1, device),
        }
    )
    pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}

    kpts0 = pred["keypoints0"]
    kpts1 = pred["keypoints1"]
    matches0 = pred["matches0"]
    scores0 = pred["matching_scores0"]

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
    inlier_mask = np.zeros((len(mkpts0),), dtype=bool)
    median_reproj_error = float("nan")
    mean_reproj_error = float("nan")
    corner_error = float("nan")
    mask = None

    if len(mkpts0) >= 4:
        estimated_homography, mask = cv2.findHomography(
            mkpts0,
            mkpts1,
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
            maxIters=RANSAC_MAX_ITERS,
            confidence=RANSAC_CONFIDENCE,
        )
        if estimated_homography is None:
            estimated_homography = None

    if estimated_homography is not None and mask is not None:
        inlier_mask = mask.reshape(-1).astype(bool)
        median_reproj_error, mean_reproj_error = compute_reprojection_errors_from_mask(
            estimated_homography,
            mkpts0.reshape(-1, 1, 2),
            mkpts1.reshape(-1, 1, 2),
            mask,
        )
        if eval_homography is not None:
            corner_error = compute_corner_error(estimated_homography, eval_homography, image0.shape[:2])

    runtime_ms = (time.perf_counter() - timer_start) * 1000.0
    num_matches = int(len(mkpts0))
    num_inliers = int(inlier_mask.sum())
    inlier_ratio = float(num_inliers / num_matches) if num_matches > 0 else 0.0
    success = int(is_registration_success(dataset, estimated_homography, num_inliers, median_reproj_error))

    result = PairResult(
        pair_id="",
        dataset=dataset,
        method="SuperPoint+SuperGlue",
        num_matches=num_matches,
        num_inliers=num_inliers,
        inlier_ratio=inlier_ratio,
        median_reproj_error=median_reproj_error,
        mean_reproj_error=mean_reproj_error,
        corner_error=corner_error,
        success=success,
        runtime_ms=runtime_ms,
        H_found=int(estimated_homography is not None),
        H_gt_found=int(eval_homography is not None),
    )
    aux = {
        "kpts0": kpts0,
        "kpts1": kpts1,
        "mkpts0": mkpts0,
        "mkpts1": mkpts1,
        "mconf": mconf,
        "inlier_mask": inlier_mask,
        "homography": estimated_homography,
        "eval_homography": eval_homography,
    }
    return result, aux


def is_registration_success(
    dataset: str,
    homography: Optional[np.ndarray],
    num_inliers: int,
    median_reproj_error: float,
) -> bool:
    if homography is None or math.isnan(median_reproj_error):
        return False
    if dataset == "hpatches":
        return num_inliers >= 10 and median_reproj_error <= 3.0
    if dataset == "inloc":
        return num_inliers >= 15 and median_reproj_error <= 5.0
    raise ValueError(f"Unsupported dataset: {dataset}")


def warmup(
    pairs: Sequence[PairItem],
    matching: Matching,
    device: str,
    warmup_pairs: int,
    max_final_matches: int,
) -> None:
    warm_count = min(warmup_pairs, len(pairs))
    if warm_count <= 0:
        return
    print(f"Warm-up on {warm_count} pairs...")
    for pair in pairs[:warm_count]:
        image0 = load_and_prepare_image(pair.image0)
        image1 = load_and_prepare_image(pair.image1)
        infer_pair(matching, device, image0, image1, pair.dataset, max_final_matches)


def make_match_text(result: PairResult) -> List[str]:
    median_str = "NaN" if math.isnan(result.median_reproj_error) else f"{result.median_reproj_error:.3f}px"
    return [
        result.method,
        f"{result.dataset} | {result.pair_id}",
        f"matches={result.num_matches} inliers={result.num_inliers}",
        f"inlier_ratio={result.inlier_ratio:.3f}",
        f"median_reproj={median_str}",
        f"success={result.success}",
    ]


def save_match_visualization(
    output_path: Path,
    image0: np.ndarray,
    image1: np.ndarray,
    aux: Dict[str, np.ndarray],
    result: PairResult,
    show_keypoints: bool,
) -> None:
    colors = cm.jet(aux["mconf"]) if len(aux["mconf"]) > 0 else np.zeros((0, 4))
    make_matching_plot_fast(
        image0,
        image1,
        aux["kpts0"],
        aux["kpts1"],
        aux["mkpts0"],
        aux["mkpts1"],
        colors,
        make_match_text(result),
        path=output_path,
        show_keypoints=show_keypoints,
    )


def draw_text_block(image: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    out = image.copy()
    y = 28
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
    return out


def gray_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def pad_to_same_height(images: Sequence[np.ndarray]) -> List[np.ndarray]:
    max_h = max(image.shape[0] for image in images)
    padded: List[np.ndarray] = []
    for image in images:
        h, w = image.shape[:2]
        if h == max_h:
            padded.append(image)
            continue
        canvas = np.full((max_h, w, image.shape[2]), 255, dtype=image.dtype)
        canvas[:h, :w] = image
        padded.append(canvas)
    return padded


def save_registration_visualization(
    output_path: Path,
    image0: np.ndarray,
    image1: np.ndarray,
    aux: Dict[str, np.ndarray],
    result: PairResult,
) -> None:
    src = gray_to_bgr(image0)
    dst = gray_to_bgr(image1)

    if aux["homography"] is None:
        overlay = dst.copy()
        overlay = draw_text_block(
            overlay,
            [
                "Homography estimation failed",
                f"pair={result.pair_id}",
                f"matches={result.num_matches} inliers={result.num_inliers}",
            ],
        )
    else:
        warped = cv2.warpPerspective(image0, aux["homography"], (image1.shape[1], image1.shape[0]))
        overlay = cv2.addWeighted(gray_to_bgr(warped), 0.5, dst, 0.5, 0.0)
        h0, w0 = image0.shape
        corners = np.array([[0, 0], [w0 - 1, 0], [w0 - 1, h0 - 1], [0, h0 - 1]], dtype=np.float32)
        warped_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), aux["homography"]).reshape(-1, 2)
        cv2.polylines(overlay, [np.round(warped_corners).astype(np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)
        overlay = draw_text_block(
            overlay,
            [
                "Registration overlay",
                f"pair={result.pair_id}",
                f"inliers={result.num_inliers} ratio={result.inlier_ratio:.3f}",
                f"median={format_metric(result.median_reproj_error)} mean={format_metric(result.mean_reproj_error)}",
                f"success={result.success}",
            ],
        )

    src, dst, overlay = pad_to_same_height([src, dst, overlay])
    canvas = np.concatenate([src, dst, overlay], axis=1)
    cv2.imwrite(str(output_path), canvas)


def format_metric(value: float) -> str:
    return "NaN" if math.isnan(value) else f"{value:.3f}px"


def result_to_csv_row(result: PairResult) -> Dict[str, object]:
    return {
        "pair_id": result.pair_id,
        "dataset": result.dataset,
        "method": result.method,
        "num_matches": result.num_matches,
        "num_inliers": result.num_inliers,
        "inlier_ratio": result.inlier_ratio,
        "median_reproj_error": result.median_reproj_error,
        "mean_reproj_error": result.mean_reproj_error,
        "corner_error": result.corner_error,
        "success": result.success,
        "runtime_ms": result.runtime_ms,
        "H_found": result.H_found,
        "H_gt_found": result.H_gt_found,
    }


def write_results_csv(path: Path, results: Sequence[PairResult]) -> None:
    fieldnames = [
        "pair_id",
        "dataset",
        "method",
        "num_matches",
        "num_inliers",
        "inlier_ratio",
        "median_reproj_error",
        "mean_reproj_error",
        "corner_error",
        "success",
        "runtime_ms",
        "H_found",
        "H_gt_found",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result_to_csv_row(result))


def safe_mean(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def safe_median(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmedian(arr))


def summarize_results(
    dataset: str,
    results: Sequence[PairResult],
    weights: str,
    output_path: Path,
) -> Dict[str, object]:
    summary = {
        "dataset": dataset,
        "method": "SuperPoint+SuperGlue",
        "superglue_weights": weights,
        "num_pairs": len(results),
        "avg_num_matches": safe_mean(r.num_matches for r in results),
        "avg_num_inliers": safe_mean(r.num_inliers for r in results),
        "avg_inlier_ratio": safe_mean(r.inlier_ratio for r in results),
        "median_of_median_reproj_error": safe_median(r.median_reproj_error for r in results),
        "avg_mean_reproj_error": safe_mean(r.mean_reproj_error for r in results),
        "mean_corner_error": safe_mean(r.corner_error for r in results),
        "median_corner_error": safe_median(r.corner_error for r in results),
        "success_rate": safe_mean(r.success for r in results),
        "avg_runtime_ms": safe_mean(r.runtime_ms for r in results),
    }
    if dataset == "hpatches":
        aucs = error_auc([r.corner_error for r in results], thresholds=[3, 5, 10])
        summary.update(aucs)
        for key, value in aucs.items():
            if not math.isnan(value):
                summary[f"{key}_pct"] = value * 100.0
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def maybe_visualize(
    pair_idx: int,
    save_viz: bool,
    viz_max_pairs: int,
    show_keypoints: bool,
    image0: np.ndarray,
    image1: np.ndarray,
    aux: Dict[str, np.ndarray],
    result: PairResult,
    viz_dir: Path,
) -> None:
    if not save_viz:
        return
    if viz_max_pairs >= 0 and pair_idx >= viz_max_pairs:
        return
    match_path = viz_dir / f"{result.pair_id}_matches.png"
    reg_path = viz_dir / f"{result.pair_id}_registration.png"
    save_match_visualization(match_path, image0, image1, aux, result, show_keypoints)
    save_registration_visualization(reg_path, image0, image1, aux, result)


def print_summary(summary: Dict[str, object]) -> None:
    print("\nSummary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
