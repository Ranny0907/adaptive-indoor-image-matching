#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORB / SIFT baseline matching pipeline for HPatches / InLoc style experiments.

Features:
- unified resize + grayscale preprocessing
- ORB / SIFT keypoint detection and description
- KNN + ratio test matching
- keep top-N matches before RANSAC
- homography estimation with fixed RANSAC params
- save raw match / inlier match / registration result images
- save per-pair JSON + unified summary CSV + simple run note

Example:
    python baseline_match.py \
        --pairs_csv pairs_hpatches_small.csv \
        --method orb \
        --output_dir results/hpatches/orb
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class PairRecord:
    pair_id: str
    dataset: str
    scene: str
    img0_path: str
    img1_path: str


@dataclass
class PairResult:
    pair_id: str
    dataset: str
    scene: str
    method: str
    img0_name: str
    img1_name: str
    num_keypoints0: int
    num_keypoints1: int
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


# -----------------------------
# Utility functions
# -----------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_image_color(path: str) -> np.ndarray:
    """Robust image reader for Windows paths that may contain non-ASCII chars."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img

    # Fallback for Windows/OpenCV Unicode-path issues.
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size > 0:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    except Exception:
        pass

    raise FileNotFoundError(f"Failed to read image: {path}")


def save_image(path: Path, img: np.ndarray) -> None:
    """Robust image writer for Windows paths that may contain non-ASCII chars."""
    ok = cv2.imwrite(str(path), img)
    if ok:
        return

    ext = path.suffix if path.suffix else '.png'
    success, buf = cv2.imencode(ext, img)
    if not success:
        raise IOError(f"Failed to encode image for saving: {path}")
    try:
        buf.tofile(str(path))
    except Exception as e:
        raise IOError(f"Failed to save image: {path}") from e


def resize_keep_long_edge(
    img: np.ndarray,
    long_edge: int = 640,
    round_to: int = 8,
    no_enlarge: bool = True,
) -> np.ndarray:
    h, w = img.shape[:2]
    current_long = max(h, w)
    scale = 1.0
    if current_long > long_edge:
        scale = long_edge / float(current_long)
    elif not no_enlarge and current_long < long_edge:
        scale = long_edge / float(current_long)

    new_w = max(1, int(math.floor(w * scale)))
    new_h = max(1, int(math.floor(h * scale)))

    if round_to > 1:
        new_w = max(round_to, (new_w // round_to) * round_to)
        new_h = max(round_to, (new_h // round_to) * round_to)

    if new_w == w and new_h == h:
        return img.copy()
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def create_detector(method: str, max_keypoints: int) -> cv2.Feature2D:
    method = method.lower()
    if method == "orb":
        return cv2.ORB_create(nfeatures=max_keypoints)
    if method == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("Current OpenCV build does not support SIFT_create().")
        return cv2.SIFT_create(nfeatures=max_keypoints)
    raise ValueError(f"Unsupported method: {method}")


def create_matcher(method: str) -> cv2.DescriptorMatcher:
    method = method.lower()
    if method == "orb":
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    if method == "sift":
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raise ValueError(f"Unsupported method: {method}")


def detect_and_compute(
    detector: cv2.Feature2D,
    gray: np.ndarray,
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    kpts, desc = detector.detectAndCompute(gray, None)
    if kpts is None:
        kpts = []
    return kpts, desc


def ratio_test_knn(
    matcher: cv2.DescriptorMatcher,
    desc0: np.ndarray,
    desc1: np.ndarray,
    ratio: float,
) -> List[cv2.DMatch]:
    raw_knn = matcher.knnMatch(desc0, desc1, k=2)
    good = []
    for pair in raw_knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def keypoints_to_points(
    kpts0: Sequence[cv2.KeyPoint],
    kpts1: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
) -> Tuple[np.ndarray, np.ndarray]:
    pts0 = np.float32([kpts0[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts1 = np.float32([kpts1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return pts0, pts1


def estimate_homography(
    pts0: np.ndarray,
    pts1: np.ndarray,
    ransac_reproj_threshold: float = 3.0,
    max_iters: int = 2000,
    confidence: float = 0.995,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if len(pts0) < 4 or len(pts1) < 4:
        return None, None
    H, mask = cv2.findHomography(
        pts0,
        pts1,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
        maxIters=max_iters,
        confidence=confidence,
    )
    return H, mask



def load_hpatches_pair_homography(img0_path: str, img1_path: str) -> Optional[np.ndarray]:
    """Infer HPatches GT homography H: img0 -> img1 from scene folder files H_1_x."""
    p0 = Path(img0_path)
    p1 = Path(img1_path)
    if p0.parent != p1.parent:
        return None
    try:
        idx0 = int(p0.stem)
        idx1 = int(p1.stem)
    except ValueError:
        return None

    scene_dir = p0.parent

    def load_h1k(k: int) -> np.ndarray:
        if k == 1:
            return np.eye(3, dtype=np.float64)
        h_path = scene_dir / f"H_1_{k}"
        if not h_path.exists():
            raise FileNotFoundError(str(h_path))
        return np.loadtxt(str(h_path)).astype(np.float64)

    try:
        H_1_0 = load_h1k(idx0)
        H_1_1 = load_h1k(idx1)
        H_0_1 = H_1_1 @ np.linalg.inv(H_1_0)
        return H_0_1
    except Exception:
        return None


def adapt_homography_to_resized_images(
    H_orig: np.ndarray,
    orig0_shape: Tuple[int, int],
    resized0_shape: Tuple[int, int],
    orig1_shape: Tuple[int, int],
    resized1_shape: Tuple[int, int],
) -> np.ndarray:
    """Convert original-image homography to resized-image coordinates."""
    oh0, ow0 = orig0_shape
    rh0, rw0 = resized0_shape
    oh1, ow1 = orig1_shape
    rh1, rw1 = resized1_shape

    sx0, sy0 = rw0 / float(ow0), rh0 / float(oh0)
    sx1, sy1 = rw1 / float(ow1), rh1 / float(oh1)

    S0 = np.array([[sx0, 0.0, 0.0], [0.0, sy0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    S1 = np.array([[sx1, 0.0, 0.0], [0.0, sy1, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return S1 @ H_orig @ np.linalg.inv(S0)


def compute_corner_error(H_est: np.ndarray, H_gt: np.ndarray, src_shape: Tuple[int, int]) -> float:
    """Mean distance between warped source-image corners under H_est and H_gt."""
    h, w = src_shape
    corners = np.array([
        [0.0, 0.0],
        [w - 1.0, 0.0],
        [w - 1.0, h - 1.0],
        [0.0, h - 1.0],
    ], dtype=np.float32).reshape(-1, 1, 2)

    try:
        est = cv2.perspectiveTransform(corners, H_est)
        gt = cv2.perspectiveTransform(corners, H_gt)
        errs = np.linalg.norm((est - gt).reshape(-1, 2), axis=1)
        return float(np.mean(errs))
    except Exception:
        return float("nan")


def error_auc(errors: Sequence[float], thresholds: Sequence[float]) -> Dict[str, float]:
    """Compute AUC up to each threshold, following common HPatches/SuperGlue style."""
    clean = np.array([e for e in errors if not math.isnan(e) and math.isfinite(e)], dtype=np.float64)
    if clean.size == 0:
        return {f"auc@{int(t)}": float("nan") for t in thresholds}

    clean = np.sort(clean)
    recalls = (np.arange(clean.size, dtype=np.float64) + 1.0) / clean.size
    out: Dict[str, float] = {}
    for t in thresholds:
        last = np.searchsorted(clean, t, side="right")
        if last == 0:
            x = np.array([0.0, float(t)], dtype=np.float64)
            y = np.array([0.0, 0.0], dtype=np.float64)
        else:
            x = np.concatenate(([0.0], clean[:last], [float(t)]))
            y = np.concatenate(([0.0], recalls[:last], [recalls[last - 1]]))
        auc = float(np.trapz(y, x) / float(t))
        out[f"auc@{int(t)}"] = auc
    return out

def compute_reprojection_errors(
    H: np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float]:
    inlier_mask = mask.reshape(-1).astype(bool)
    if not np.any(inlier_mask):
        return float("nan"), float("nan")

    pts0_in = pts0[inlier_mask]
    pts1_in = pts1[inlier_mask]
    proj = cv2.perspectiveTransform(pts0_in, H)
    errs = np.linalg.norm((proj - pts1_in).reshape(-1, 2), axis=1)
    return float(np.median(errs)), float(np.mean(errs))


def success_rule(dataset: str, H: Optional[np.ndarray], num_inliers: int, median_err: float) -> int:
    dataset_low = dataset.lower()
    if H is None or math.isnan(median_err):
        return 0
    if dataset_low == "hpatches":
        n_min = 10
        tau = 3.0
    else:
        # Follow your current plan for InLoc / other indoor low-texture tests.
        n_min = 15
        tau = 5.0
    return int(num_inliers >= n_min and median_err <= tau)


def overlay_registration(img0: np.ndarray, img1: np.ndarray, H: Optional[np.ndarray]) -> np.ndarray:
    # Returns a visualization with target / warped / overlay panels.
    h1, w1 = img1.shape[:2]
    if H is None:
        warped = np.zeros_like(img1)
    else:
        warped = cv2.warpPerspective(img0, H, (w1, h1))
    overlay = cv2.addWeighted(img1, 0.5, warped, 0.5, 0)
    panel = np.concatenate([img1, warped, overlay], axis=1)
    return panel


def put_title(img: np.ndarray, title: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def save_match_vis(
    img0: np.ndarray,
    kpts0: Sequence[cv2.KeyPoint],
    img1: np.ndarray,
    kpts1: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    out_path: Path,
    title: Optional[str] = None,
) -> None:
    vis = cv2.drawMatches(
        img0,
        list(kpts0),
        img1,
        list(kpts1),
        list(matches),
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    if title:
        vis = put_title(vis, title)
    save_image(out_path, vis)


def write_summary_csv(csv_path: Path, results: Sequence[PairResult]) -> None:
    ensure_dir(csv_path.parent)
    fieldnames = list(asdict(results[0]).keys()) if results else [
        "pair_id", "dataset", "scene", "method", "img0_name", "img1_name",
        "num_keypoints0", "num_keypoints1", "num_matches", "num_inliers",
        "inlier_ratio", "median_reproj_error", "mean_reproj_error",
        "success", "runtime_ms", "H_found", "H_gt_found"
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))


def write_run_note(note_path: Path, args: argparse.Namespace, results: Sequence[PairResult]) -> None:
    ensure_dir(note_path.parent)
    num_pairs = len(results)
    num_success = sum(r.success for r in results)
    avg_runtime = float(np.mean([r.runtime_ms for r in results])) if results else float("nan")
    avg_matches = float(np.mean([r.num_matches for r in results])) if results else float("nan")
    avg_inliers = float(np.mean([r.num_inliers for r in results])) if results else float("nan")

    corner_vals = [r.corner_error for r in results if not math.isnan(r.corner_error)]
    mean_corner = float(np.mean(corner_vals)) if corner_vals else float("nan")
    median_corner = float(np.median(corner_vals)) if corner_vals else float("nan")

    hpatches_results = [r for r in results if r.dataset.lower() == "hpatches"]
    aucs = error_auc([r.corner_error for r in hpatches_results], thresholds=[3, 5, 10]) if hpatches_results else {}

    lines = [
        "# 实验记录\n",
        f"- 方法: {args.method}\n",
        f"- 输入 pairs_csv: {args.pairs_csv}\n",
        f"- 输出目录: {args.output_dir}\n",
        f"- 图像长边: {args.long_edge}\n",
        f"- 不放大: {not args.allow_enlarge}\n",
        f"- ORB ratio: {args.orb_ratio}\n",
        f"- SIFT ratio: {args.sift_ratio}\n",
        f"- 最大关键点数: {args.max_keypoints}\n",
        f"- RANSAC 前最多匹配数: {args.max_matches}\n",
        f"- RANSAC 阈值: {args.ransac_reproj_threshold}\n",
        f"- RANSAC maxIters: {args.ransac_max_iters}\n",
        f"- RANSAC confidence: {args.ransac_confidence}\n",
        f"- 图像对数量: {num_pairs}\n",
        f"- 成功配准数: {num_success}\n",
        f"- 成功率: {num_success / num_pairs:.4f}\n" if num_pairs else "- 成功率: NaN\n",
        f"- 平均初始匹配数: {avg_matches:.2f}\n",
        f"- 平均内点数: {avg_inliers:.2f}\n",
        f"- 平均单对耗时(ms): {avg_runtime:.2f}\n",
    ]

    if corner_vals:
        lines.append(f"- mean corner error(px): {mean_corner:.4f}\n")
        lines.append(f"- median corner error(px): {median_corner:.4f}\n")
    if aucs:
        for k, v in aucs.items():
            if not math.isnan(v):
                lines.append(f"- {k} (0-1): {v:.4f}\n")
                lines.append(f"- {k} (%): {v * 100.0:.2f}\n")

    note_path.write_text("".join(lines), encoding="utf-8")


def write_metrics_json(metrics_path: Path, results: Sequence[PairResult]) -> None:
    ensure_dir(metrics_path.parent)
    hpatches_results = [r for r in results if r.dataset.lower() == "hpatches"]
    corner_vals = [r.corner_error for r in results if not math.isnan(r.corner_error)]
    metrics = {
        "num_pairs": len(results),
        "num_success": int(sum(r.success for r in results)),
        "success_rate": float(sum(r.success for r in results) / len(results)) if results else float("nan"),
        "avg_matches": float(np.mean([r.num_matches for r in results])) if results else float("nan"),
        "avg_inliers": float(np.mean([r.num_inliers for r in results])) if results else float("nan"),
        "avg_runtime_ms": float(np.mean([r.runtime_ms for r in results])) if results else float("nan"),
        "mean_corner_error": float(np.mean(corner_vals)) if corner_vals else float("nan"),
        "median_corner_error": float(np.median(corner_vals)) if corner_vals else float("nan"),
    }
    if hpatches_results:
        aucs = error_auc([r.corner_error for r in hpatches_results], thresholds=[3, 5, 10])
        metrics.update(aucs)
        for k, v in aucs.items():
            if not math.isnan(v):
                metrics[k + "_pct"] = v * 100.0

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------
# Pair loading
# -----------------------------
def load_pairs_csv(path: str) -> List[PairRecord]:
    pairs: List[PairRecord] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"pair_id", "dataset", "scene", "img0_path", "img1_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"pairs_csv missing columns: {sorted(missing)}")
        for row in reader:
            pairs.append(
                PairRecord(
                    pair_id=row["pair_id"],
                    dataset=row["dataset"],
                    scene=row["scene"],
                    img0_path=row["img0_path"],
                    img1_path=row["img1_path"],
                )
            )
    return pairs


# -----------------------------
# Core pipeline
# -----------------------------
def process_pair(
    pair: PairRecord,
    detector: cv2.Feature2D,
    matcher: cv2.DescriptorMatcher,
    args: argparse.Namespace,
    vis_raw_dir: Path,
    vis_inlier_dir: Path,
    reg_dir: Path,
    json_dir: Path,
) -> PairResult:
    img0_color = read_image_color(pair.img0_path)
    img1_color = read_image_color(pair.img1_path)
    orig0_shape = img0_color.shape[:2]
    orig1_shape = img1_color.shape[:2]

    img0_color = resize_keep_long_edge(
        img0_color,
        long_edge=args.long_edge,
        round_to=args.round_to,
        no_enlarge=not args.allow_enlarge,
    )
    img1_color = resize_keep_long_edge(
        img1_color,
        long_edge=args.long_edge,
        round_to=args.round_to,
        no_enlarge=not args.allow_enlarge,
    )

    img0_gray = to_gray(img0_color)
    img1_gray = to_gray(img1_color)
    resized0_shape = img0_color.shape[:2]
    resized1_shape = img1_color.shape[:2]

    H_gt_resized: Optional[np.ndarray] = None
    if pair.dataset.lower() == "hpatches":
        H_gt_orig = load_hpatches_pair_homography(pair.img0_path, pair.img1_path)
        if H_gt_orig is not None:
            H_gt_resized = adapt_homography_to_resized_images(
                H_gt_orig, orig0_shape, resized0_shape, orig1_shape, resized1_shape
            )

    t0 = time.perf_counter()

    kpts0, desc0 = detect_and_compute(detector, img0_gray)
    kpts1, desc1 = detect_and_compute(detector, img1_gray)

    ratio = args.orb_ratio if args.method.lower() == "orb" else args.sift_ratio

    if desc0 is None or desc1 is None or len(kpts0) == 0 or len(kpts1) == 0:
        H = None
        matches: List[cv2.DMatch] = []
        inlier_matches: List[cv2.DMatch] = []
        num_inliers = 0
        inlier_ratio = 0.0
        median_err = float("nan")
        mean_err = float("nan")
        corner_err = float("nan")
    else:
        matches = ratio_test_knn(matcher, desc0, desc1, ratio=ratio)
        matches = sorted(matches, key=lambda m: m.distance)
        if len(matches) > args.max_matches:
            matches = matches[: args.max_matches]

        if len(matches) < 4:
            H = None
            inlier_matches = []
            num_inliers = 0
            inlier_ratio = 0.0
            median_err = float("nan")
            mean_err = float("nan")
            corner_err = float("nan")
        else:
            pts0, pts1 = keypoints_to_points(kpts0, kpts1, matches)
            H, mask = estimate_homography(
                pts0,
                pts1,
                ransac_reproj_threshold=args.ransac_reproj_threshold,
                max_iters=args.ransac_max_iters,
                confidence=args.ransac_confidence,
            )
            if H is None or mask is None:
                inlier_matches = []
                num_inliers = 0
                inlier_ratio = 0.0
                median_err = float("nan")
                mean_err = float("nan")
                corner_err = float("nan")
            else:
                inlier_mask = mask.reshape(-1).astype(bool)
                inlier_matches = [m for m, keep in zip(matches, inlier_mask) if keep]
                num_inliers = int(np.sum(inlier_mask))
                inlier_ratio = float(num_inliers / len(matches)) if matches else 0.0
                median_err, mean_err = compute_reprojection_errors(H, pts0, pts1, mask)
                corner_err = compute_corner_error(H, H_gt_resized, resized0_shape) if H_gt_resized is not None else float("nan")

    runtime_ms = (time.perf_counter() - t0) * 1000.0
    success = success_rule(pair.dataset, H, num_inliers, median_err)

    raw_vis_path = vis_raw_dir / f"{pair.pair_id}_raw.jpg"
    inlier_vis_path = vis_inlier_dir / f"{pair.pair_id}_inlier.jpg"
    reg_vis_path = reg_dir / f"{pair.pair_id}_reg.jpg"
    json_path = json_dir / f"{pair.pair_id}.json"

    save_match_vis(
        img0_color,
        kpts0,
        img1_color,
        kpts1,
        matches,
        raw_vis_path,
        title=f"{pair.pair_id} | raw matches={len(matches)}",
    )
    save_match_vis(
        img0_color,
        kpts0,
        img1_color,
        kpts1,
        inlier_matches,
        inlier_vis_path,
        title=f"{pair.pair_id} | inliers={num_inliers}",
    )

    reg_vis = overlay_registration(img0_color, img1_color, H)
    reg_vis = put_title(reg_vis, f"{pair.pair_id} | success={success}")
    save_image(reg_vis_path, reg_vis)

    result = PairResult(
        pair_id=pair.pair_id,
        dataset=pair.dataset,
        scene=pair.scene,
        method=args.method.lower(),
        img0_name=Path(pair.img0_path).name,
        img1_name=Path(pair.img1_path).name,
        num_keypoints0=len(kpts0),
        num_keypoints1=len(kpts1),
        num_matches=len(matches),
        num_inliers=num_inliers,
        inlier_ratio=inlier_ratio,
        median_reproj_error=median_err,
        mean_reproj_error=mean_err,
        corner_error=corner_err,
        success=success,
        runtime_ms=runtime_ms,
        H_found=int(H is not None),
        H_gt_found=int(H_gt_resized is not None),
    )

    json_payload: Dict[str, object] = asdict(result)
    json_payload.update(
        {
            "img0_path": pair.img0_path,
            "img1_path": pair.img1_path,
            "homography": H.tolist() if H is not None else None,
            "homography_gt_resized": H_gt_resized.tolist() if H_gt_resized is not None else None,
        }
    )
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return result


# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ORB/SIFT baseline matching experiments.")
    parser.add_argument("--pairs_csv", type=str, required=True, help="CSV with pair_id,dataset,scene,img0_path,img1_path")
    parser.add_argument("--method", type=str, choices=["orb", "sift"], required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--long_edge", type=int, default=640)
    parser.add_argument("--round_to", type=int, default=8)
    parser.add_argument("--allow_enlarge", action="store_true", help="Allow enlarging images smaller than target long edge")

    parser.add_argument("--max_keypoints", type=int, default=2000)
    parser.add_argument("--max_matches", type=int, default=1000)
    parser.add_argument("--orb_ratio", type=float, default=0.90)
    parser.add_argument("--sift_ratio", type=float, default=0.75)

    parser.add_argument("--ransac_reproj_threshold", type=float, default=3.0)
    parser.add_argument("--ransac_max_iters", type=int, default=2000)
    parser.add_argument("--ransac_confidence", type=float, default=0.995)

    parser.add_argument("--warmup", type=int, default=0, help="Run first N pairs as warm-up and do not include them in summary")
    parser.add_argument("--limit", type=int, default=0, help="If > 0, only run first N pairs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pairs = load_pairs_csv(args.pairs_csv)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        raise ValueError("No pairs loaded from pairs_csv.")

    output_dir = Path(args.output_dir)
    vis_raw_dir = output_dir / "pair_vis" / "raw_matches"
    vis_inlier_dir = output_dir / "pair_vis" / "inlier_matches"
    reg_dir = output_dir / "pair_vis" / "registration"
    json_dir = output_dir / "pair_json"
    for d in [vis_raw_dir, vis_inlier_dir, reg_dir, json_dir]:
        ensure_dir(d)

    detector = create_detector(args.method, args.max_keypoints)
    matcher = create_matcher(args.method)

    results: List[PairResult] = []
    warmup_n = max(0, min(args.warmup, len(pairs)))

    if warmup_n > 0:
        print(f"[Warm-up] Running first {warmup_n} pairs without recording summary...")
        for pair in pairs[:warmup_n]:
            _ = process_pair(pair, detector, matcher, args, vis_raw_dir, vis_inlier_dir, reg_dir, json_dir)

    formal_pairs = pairs[warmup_n:]
    print(f"[Run] method={args.method}, total_formal_pairs={len(formal_pairs)}")
    for idx, pair in enumerate(formal_pairs, start=1):
        result = process_pair(pair, detector, matcher, args, vis_raw_dir, vis_inlier_dir, reg_dir, json_dir)
        results.append(result)
        print(
            f"[{idx}/{len(formal_pairs)}] {pair.pair_id} | "
            f"matches={result.num_matches} inliers={result.num_inliers} "
            f"median_err={result.median_reproj_error:.3f} "
            f"corner_err={result.corner_error:.3f} success={result.success} "
            f"time={result.runtime_ms:.2f}ms"
        )

    summary_path = output_dir / "summary.csv"
    note_path = output_dir / "run_note.md"
    write_summary_csv(summary_path, results)
    write_run_note(note_path, args, results)

    print(f"\n[Done] Summary saved to: {summary_path}")
    print(f"[Done] Run note saved to: {note_path}")
    print(f"[Done] Visualizations saved under: {output_dir / 'pair_vis'}")


if __name__ == "__main__":
    main()
