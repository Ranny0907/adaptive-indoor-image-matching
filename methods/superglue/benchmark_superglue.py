#!/usr/bin/env python3
"""Benchmark SuperPoint + SuperGlue on HPatches or InLoc image pairs.

This script follows the experiment rules defined in `分工.md`:
1. grayscale input only
2. keep aspect ratio, resize longest side to 640 without upsampling
3. floor resized width/height to multiples of 8
4. keep at most 1000 final matches before RANSAC, sorted by confidence
5. fixed homography RANSAC settings
6. runtime excludes image loading / resize and model loading
7. SuperGlue warm-up before timing

The script is designed for remote execution environments such as AutoDL.
It does not download datasets automatically. See `AUTODL_实验说明.md`.
"""

from __future__ import annotations

import argparse
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
    success: int
    runtime_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SuperPoint + SuperGlue on HPatches or InLoc.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["hpatches", "inloc"],
        required=True,
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Dataset root path on AutoDL.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to store CSV, summary JSON and visualizations.",
    )
    parser.add_argument(
        "--inloc_pairs",
        type=Path,
        default=None,
        help="Pair file for InLoc. Required when --dataset inloc.",
    )
    parser.add_argument(
        "--hpatches_split",
        choices=["all", "i", "v"],
        default="all",
        help="Use all HPatches sequences, illumination-only, or viewpoint-only.",
    )
    parser.add_argument(
        "--superglue_weights",
        choices=["auto", "indoor", "outdoor"],
        default="auto",
        help="SuperGlue weight selection. 'auto' uses outdoor for HPatches and indoor for InLoc.",
    )
    parser.add_argument("--max_keypoints", type=int, default=1024)
    parser.add_argument("--keypoint_threshold", type=float, default=0.005)
    parser.add_argument("--nms_radius", type=int, default=4)
    parser.add_argument("--sinkhorn_iterations", type=int, default=20)
    parser.add_argument("--match_threshold", type=float, default=0.20)
    parser.add_argument("--max_final_matches", type=int, default=MAX_FINAL_MATCHES)
    parser.add_argument(
        "--warmup_pairs",
        type=int,
        default=5,
        help="Number of pairs used for warm-up before formal timing.",
    )
    parser.add_argument(
        "--save_viz",
        action="store_true",
        help="Save match visualizations and registration visualizations.",
    )
    parser.add_argument(
        "--viz_max_pairs",
        type=int,
        default=20,
        help="Maximum number of pairs to visualize. Use -1 for all pairs.",
    )
    parser.add_argument(
        "--show_keypoints",
        action="store_true",
        help="Overlay all detected keypoints in the match visualization.",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force inference on CPU.",
    )
    return parser.parse_args()


def choose_weights(dataset: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "outdoor" if dataset == "hpatches" else "indoor"


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


def resolve_existing_image_stem(stem: Path) -> Path:
    for suffix in [".ppm", ".png", ".jpg", ".jpeg", ".bmp"]:
        candidate = stem.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find image for stem: {stem}")


def discover_hpatches_pairs(data_root: Path, split: str) -> List[PairItem]:
    pairs: List[PairItem] = []
    prefixes = {"all": ("i_", "v_"), "i": ("i_",), "v": ("v_",)}[split]

    for seq_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if not seq_dir.name.startswith(prefixes):
            continue
        image0 = resolve_existing_image_stem(seq_dir / "1")
        for target_idx in range(2, 7):
            image1 = resolve_existing_image_stem(seq_dir / str(target_idx))
            pair_id = f"{seq_dir.name}_1_{target_idx}"
            pairs.append(
                PairItem(
                    pair_id=pair_id,
                    dataset="hpatches",
                    image0=image0,
                    image1=image1,
                )
            )
    if not pairs:
        raise RuntimeError(f"No HPatches pairs found under {data_root}")
    return pairs


def parse_inloc_pairs_file(pair_file: Path, data_root: Path) -> List[PairItem]:
    if not pair_file.exists():
        raise FileNotFoundError(f"InLoc pair file not found: {pair_file}")

    pairs: List[PairItem] = []
    if pair_file.suffix.lower() == ".csv":
        with pair_file.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"pair_id", "image0", "image1"}
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise ValueError("CSV pair file must contain headers: pair_id,image0,image1")
            for row in reader:
                image0 = resolve_relative_or_absolute_path(row["image0"], data_root)
                image1 = resolve_relative_or_absolute_path(row["image1"], data_root)
                pairs.append(PairItem(row["pair_id"], "inloc", image0, image1))
    else:
        with pair_file.open("r", encoding="utf-8-sig") as f:
            for line_idx, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "," in line:
                    parts = [part.strip() for part in line.split(",")]
                else:
                    parts = line.split()
                if len(parts) < 3:
                    raise ValueError(
                        f"Invalid line {line_idx} in {pair_file}: expected pair_id image0 image1"
                    )
                pair_id, image0_str, image1_str = parts[:3]
                image0 = resolve_relative_or_absolute_path(image0_str, data_root)
                image1 = resolve_relative_or_absolute_path(image1_str, data_root)
                pairs.append(PairItem(pair_id, "inloc", image0, image1))

    if not pairs:
        raise RuntimeError(f"No InLoc pairs loaded from {pair_file}")
    return pairs


def resolve_relative_or_absolute_path(path_str: str, data_root: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = data_root / path
    if not path.exists():
        raise FileNotFoundError(f"Referenced image not found: {path}")
    return path


def build_pairs(args: argparse.Namespace) -> List[PairItem]:
    if args.dataset == "hpatches":
        return discover_hpatches_pairs(args.data_root, args.hpatches_split)
    if args.inloc_pairs is None:
        raise ValueError("--inloc_pairs is required when --dataset inloc")
    return parse_inloc_pairs_file(args.inloc_pairs, args.data_root)


def infer_pair(
    matching: Matching,
    device: str,
    image0: np.ndarray,
    image1: np.ndarray,
    dataset: str,
    max_final_matches: int,
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

    homography = None
    inlier_mask = np.zeros((len(mkpts0),), dtype=bool)
    median_reproj_error = float("nan")
    mean_reproj_error = float("nan")

    if len(mkpts0) >= 4:
        homography, mask = cv2.findHomography(
            mkpts0,
            mkpts1,
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
            maxIters=RANSAC_MAX_ITERS,
            confidence=RANSAC_CONFIDENCE,
        )
        if homography is not None and mask is not None:
            inlier_mask = mask.ravel().astype(bool)
            if np.any(inlier_mask):
                projected = cv2.perspectiveTransform(
                    mkpts0[inlier_mask].reshape(-1, 1, 2).astype(np.float32),
                    homography,
                ).reshape(-1, 2)
                reproj_errors = np.linalg.norm(projected - mkpts1[inlier_mask], axis=1)
                median_reproj_error = float(np.median(reproj_errors))
                mean_reproj_error = float(np.mean(reproj_errors))
        else:
            homography = None

    runtime_ms = (time.perf_counter() - timer_start) * 1000.0
    num_matches = int(len(mkpts0))
    num_inliers = int(inlier_mask.sum())
    inlier_ratio = float(num_inliers / num_matches) if num_matches > 0 else 0.0
    success = int(is_registration_success(dataset, homography, num_inliers, median_reproj_error))

    result = PairResult(
        pair_id="",
        dataset=dataset,
        method="SuperPoint+SuperGlue",
        num_matches=num_matches,
        num_inliers=num_inliers,
        inlier_ratio=inlier_ratio,
        median_reproj_error=median_reproj_error,
        mean_reproj_error=mean_reproj_error,
        success=success,
        runtime_ms=runtime_ms,
    )
    aux = {
        "kpts0": kpts0,
        "kpts1": kpts1,
        "mkpts0": mkpts0,
        "mkpts1": mkpts1,
        "mconf": mconf,
        "inlier_mask": inlier_mask,
        "homography": homography,
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
        "success": result.success,
        "runtime_ms": result.runtime_ms,
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
        "success",
        "runtime_ms",
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
        "success_rate": safe_mean(r.success for r in results),
        "avg_runtime_ms": safe_mean(r.runtime_ms for r in results),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def print_summary(summary: Dict[str, object]) -> None:
    print("\nSummary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def maybe_visualize(
    pair_idx: int,
    args: argparse.Namespace,
    image0: np.ndarray,
    image1: np.ndarray,
    aux: Dict[str, np.ndarray],
    result: PairResult,
    viz_dir: Path,
) -> None:
    if not args.save_viz:
        return
    if args.viz_max_pairs >= 0 and pair_idx >= args.viz_max_pairs:
        return
    match_path = viz_dir / f"{result.pair_id}_matches.png"
    reg_path = viz_dir / f"{result.pair_id}_registration.png"
    save_match_visualization(match_path, image0, image1, aux, result, args.show_keypoints)
    save_registration_visualization(reg_path, image0, image1, aux, result)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = args.output_dir / "viz"
    if args.save_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(args)
    weights = choose_weights(args.dataset, args.superglue_weights)
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"

    print(f"Dataset: {args.dataset}")
    print(f"Pairs: {len(pairs)}")
    print(f"Device: {device}")
    print(f"SuperGlue weights: {weights}")

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
    matching = Matching(config).eval().to(device)

    warmup(pairs, matching, device, args.warmup_pairs, args.max_final_matches)

    results: List[PairResult] = []
    for idx, pair in enumerate(pairs, start=1):
        image0 = load_and_prepare_image(pair.image0)
        image1 = load_and_prepare_image(pair.image1)

        result, aux = infer_pair(
            matching,
            device,
            image0,
            image1,
            pair.dataset,
            args.max_final_matches,
        )
        result.pair_id = pair.pair_id
        results.append(result)

        maybe_visualize(idx - 1, args, image0, image1, aux, result, viz_dir)

        print(
            f"[{idx}/{len(pairs)}] {pair.pair_id} | "
            f"matches={result.num_matches} inliers={result.num_inliers} "
            f"success={result.success} runtime={result.runtime_ms:.2f} ms"
        )

    csv_path = args.output_dir / f"{args.dataset}_superglue_results.csv"
    summary_path = args.output_dir / f"{args.dataset}_superglue_summary.json"
    write_results_csv(csv_path, results)
    summary = summarize_results(args.dataset, results, weights, summary_path)
    print_summary(summary)


if __name__ == "__main__":
    main()
