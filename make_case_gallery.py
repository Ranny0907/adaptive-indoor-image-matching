#!/usr/bin/env python3
"""Generate case-gallery visualizations for presentation slides."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
SUPERGLUE_DIR = ROOT / "methods" / "superglue"
sys.path.insert(0, str(SUPERGLUE_DIR))

from adaptive_cascade_inloc import (  # noqa: E402
    INLOC_MAX_MEDIAN_ERROR,
    INLOC_MIN_INLIERS,
    MAX_FINAL_MATCHES,
    compute_reprojection_errors,
    resolve_path,
)
from superglue_benchmark_lib import (  # noqa: E402
    create_matching,
    frame_to_tensor,
    infer_pair,
    load_and_prepare_image,
    save_match_visualization,
)

try:
    import kornia.feature as KF
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Kornia is required to generate LoFTR case images.") from exc


STRATEGY_DIR = ROOT / "results" / "adaptive_cascade_results" / "inloc_sg20_loftr40_cached"
OUTPUT_DIR = ROOT / "results" / "case_gallery"


def draw_loftr_matches(
    image0: np.ndarray,
    image1: np.ndarray,
    points0: np.ndarray,
    points1: np.ndarray,
    inlier_mask: np.ndarray,
    output_path: Path,
    title_lines: list[str],
) -> None:
    left = cv2.cvtColor(image0, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] :] = right

    offset = np.array([left.shape[1], 0], dtype=np.float32)
    keep = np.linspace(0, len(points0) - 1, min(len(points0), 140)).astype(int) if len(points0) else []
    for idx in keep:
        p0 = tuple(np.round(points0[idx]).astype(int))
        p1 = tuple(np.round(points1[idx] + offset).astype(int))
        color = (35, 170, 80) if inlier_mask[idx] else (55, 80, 220)
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, color, -1, cv2.LINE_AA)

    pad = 92
    out = np.full((canvas.shape[0] + pad, canvas.shape[1], 3), 248, dtype=np.uint8)
    out[pad:, :] = canvas
    y = 28
    for line in title_lines:
        cv2.putText(out, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 28, 38), 2, cv2.LINE_AA)
        y += 28
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out)


def build_contact_sheet(image_paths: list[Path], output_path: Path, title: str) -> None:
    thumbs = []
    thumb_w, thumb_h = 520, 300
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        scale = min(thumb_w / image.shape[1], thumb_h / image.shape[0])
        resized = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        tile = np.full((thumb_h, thumb_w, 3), 245, dtype=np.uint8)
        y = (thumb_h - resized.shape[0]) // 2
        x = (thumb_w - resized.shape[1]) // 2
        tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        thumbs.append(tile)
    if not thumbs:
        return

    rows = []
    for i in range(0, len(thumbs), 4):
        row = thumbs[i : i + 4]
        while len(row) < 4:
            row.append(np.full((thumb_h, thumb_w, 3), 245, dtype=np.uint8))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    header = np.full((70, grid.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, title, (24, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 28, 38), 3, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack([header, grid]))


def select_case_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    query_df = pd.read_csv(STRATEGY_DIR / "adaptive_query_results.csv")
    attempts = pd.read_csv(STRATEGY_DIR / "adaptive_attempts.csv")

    sg_queries = query_df[(query_df["success"] == 1) & (query_df["final_stage"] == "SuperGlue")].copy()
    sg_queries = sg_queries.sort_values(["final_rank", "total_runtime_ms", "query_img"]).head(8)

    loftr_queries = query_df[(query_df["success"] == 1) & (query_df["final_stage"] == "LoFTR")].copy()
    loftr_queries = loftr_queries.sort_values(["final_rank", "total_runtime_ms", "query_img"]).head(8)

    sg_rows = []
    for _, row in sg_queries.iterrows():
        selected = attempts[
            (attempts["query_img"] == row["query_img"])
            & (attempts["stage"] == "SuperGlue")
            & (attempts["rank"] == int(row["final_rank"]))
        ]
        if not selected.empty:
            sg_rows.append(selected.iloc[0])

    loftr_rows = []
    for _, row in loftr_queries.iterrows():
        selected = attempts[
            (attempts["query_img"] == row["query_img"])
            & (attempts["stage"] == "LoFTR")
            & (attempts["rank"] == int(row["final_rank"]))
        ]
        if not selected.empty:
            loftr_rows.append(selected.iloc[0])

    return pd.DataFrame(sg_rows), pd.DataFrame(loftr_rows)


def generate_superglue_cases(rows: pd.DataFrame, device: str) -> list[Path]:
    args = SimpleNamespace(
        nms_radius=4,
        keypoint_threshold=0.005,
        max_keypoints=1024,
        sinkhorn_iterations=20,
        match_threshold=0.20,
    )
    model = create_matching(device, "indoor", args)
    out_paths = []
    for idx, row in enumerate(rows.itertuples(index=False), start=1):
        image0 = load_and_prepare_image(resolve_path(str(row.image0_path)))
        image1 = load_and_prepare_image(resolve_path(str(row.image1_path)))
        result, aux = infer_pair(model, device, image0, image1, "inloc", MAX_FINAL_MATCHES)
        result.pair_id = str(row.pair_id)
        out_path = OUTPUT_DIR / "superglue_success" / f"{idx:02d}_{str(row.query_img).replace('.', '_')}_rank{int(row.rank):02d}_superglue.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_match_visualization(out_path, image0, image1, aux, result, show_keypoints=False)
        out_paths.append(out_path)
    return out_paths


def generate_loftr_cases(rows: pd.DataFrame, device: str) -> list[Path]:
    model = KF.LoFTR(pretrained="indoor").to(device).eval()
    out_paths = []
    for idx, row in enumerate(rows.itertuples(index=False), start=1):
        image0 = load_and_prepare_image(resolve_path(str(row.image0_path)))
        image1 = load_and_prepare_image(resolve_path(str(row.image1_path)))
        start = time.perf_counter()
        with torch.inference_mode():
            output = model({"image0": frame_to_tensor(image0, device), "image1": frame_to_tensor(image1, device)})
        points0 = output["keypoints0"].detach().cpu().numpy()
        points1 = output["keypoints1"].detach().cpu().numpy()
        confidence = output["confidence"].detach().cpu().numpy()
        order = np.argsort(-confidence)
        points0 = points0[order][:MAX_FINAL_MATCHES]
        points1 = points1[order][:MAX_FINAL_MATCHES]

        homography = None
        mask = None
        if len(points0) >= 4:
            homography, mask = cv2.findHomography(
                points0.reshape(-1, 1, 2),
                points1.reshape(-1, 1, 2),
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=2000,
                confidence=0.995,
            )
        inlier_mask = np.zeros((len(points0),), dtype=bool)
        median_error = float("nan")
        if homography is not None and mask is not None:
            inlier_mask = mask.reshape(-1).astype(bool)
            if inlier_mask.any():
                errors = compute_reprojection_errors(homography, points0[inlier_mask], points1[inlier_mask])
                median_error = float(np.median(errors))
        runtime_ms = (time.perf_counter() - start) * 1000.0
        success = int(
            homography is not None
            and int(inlier_mask.sum()) >= INLOC_MIN_INLIERS
            and not math.isnan(median_error)
            and median_error <= INLOC_MAX_MEDIAN_ERROR
        )
        title = [
            "LoFTR rescue after SuperGlue failure",
            f"rank={int(row.rank)} matches={len(points0)} inliers={int(inlier_mask.sum())} success={success}",
            f"median reproj error={'NaN' if math.isnan(median_error) else f'{median_error:.3f}px'} runtime={runtime_ms:.1f}ms",
        ]
        out_path = OUTPUT_DIR / "loftr_rescue" / f"{idx:02d}_{str(row.query_img).replace('.', '_')}_rank{int(row.rank):02d}_loftr.png"
        draw_loftr_matches(image0, image1, points0, points1, inlier_mask, out_path, title)
        out_paths.append(out_path)
    return out_paths


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sg_rows, loftr_rows = select_case_rows()
    if len(sg_rows) < 6 or len(loftr_rows) < 6:
        raise RuntimeError(f"Not enough cases selected: SuperGlue={len(sg_rows)}, LoFTR={len(loftr_rows)}")

    sg_paths = generate_superglue_cases(sg_rows, device)
    loftr_paths = generate_loftr_cases(loftr_rows, device)
    build_contact_sheet(sg_paths, OUTPUT_DIR / "superglue_success_contact_sheet.png", "SuperGlue direct success cases")
    build_contact_sheet(loftr_paths, OUTPUT_DIR / "loftr_rescue_contact_sheet.png", "LoFTR rescue success cases")

    print(f"Generated {len(sg_paths)} SuperGlue cases")
    print(f"Generated {len(loftr_paths)} LoFTR rescue cases")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
