import os
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import kornia.feature as KF

PAIRS_CSV = r"E:\CV\LoFTR\InLoc-Loftr\pairs_duc1_top201.csv"
OUTPUT_DIR = r"E:\CV\LoFTR\InLoc-Loftr"

SAVE_VIS_PER_METHOD = 100

# Evaluation settings
MAX_MATCHES = 1000

RANSAC_THRESH = 3.0
RANSAC_MAX_ITERS = 2000
RANSAC_CONFIDENCE = 0.995

INLOC_MIN_INLIERS = 15
INLOC_MAX_MEDIAN_ERROR = 5.0

WARMUP_PAIRS = 5

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_gray(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"读取失败: {path}")
    return img

def gray_to_tensor(img, device):
    t = torch.from_numpy(img).float() / 255.0
    return t.unsqueeze(0).unsqueeze(0).to(device)

def pts_to_keypoints(pts):
    return [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in pts]

def make_dmatches(n):
    return [cv2.DMatch(i, i, 0.0) for i in range(n)]

def resize_img(img, max_size=640):
    h, w = img.shape
    scale = max_size / max(h, w)

    if scale < 1:
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale))
        )
    return img

def compute_reprojection_errors(H, pts0, pts1):
    pts0_cv = pts0.reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(
        pts0_cv,
        H
    ).reshape(-1, 2)

    errors = np.linalg.norm(
        projected - pts1,
        axis=1
    )

    return errors

def run_loftr(img1, img2, loftr, device):
    t0 = time.perf_counter()

    t1_img = gray_to_tensor(img1, device)
    t2_img = gray_to_tensor(img2, device)

    with torch.no_grad():
        batch = {
            "image0": t1_img,
            "image1": t2_img
        }

        out = loftr(batch)

    result = out if isinstance(out, dict) else batch

    pts0 = result["keypoints0"].cpu().numpy()
    pts1 = result["keypoints1"].cpu().numpy()
    conf = result["confidence"].cpu().numpy()

    total_time_ms = (time.perf_counter() - t0) * 1000

    if len(pts0) == 0:
        return {
            "method": "LoFTR",
            "kp1": 0,
            "kp2": 0,
            "raw_matches": 0,
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "median_reproj_error": np.inf,
            "success": 0,
            "time_ms": total_time_ms,
            "vis_matches": [],
            "H_found": 0,
        }

    idx = np.argsort(-conf)

    pts0 = pts0[idx]
    pts1 = pts1[idx]

    raw_matches = len(pts0)

    if len(pts0) > MAX_MATCHES:
        pts0 = pts0[:MAX_MATCHES]
        pts1 = pts1[:MAX_MATCHES]

    num_matches = len(pts0)

    inliers = 0
    inlier_ratio = 0.0
    median_error = np.inf
    H_found = 0
    success = 0

    vis_indices = list(range(num_matches))

    if num_matches >= 4:

        pts0_cv = pts0.reshape(-1, 1, 2)
        pts1_cv = pts1.reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            pts0_cv,
            pts1_cv,
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_THRESH,
            maxIters=RANSAC_MAX_ITERS,
            confidence=RANSAC_CONFIDENCE
        )

        if H is not None and mask is not None:

            H_found = 1

            mask = mask.ravel().astype(bool)

            inliers = int(mask.sum())

            inlier_ratio = inliers / num_matches

            vis_indices = np.where(mask)[0]

            errors = compute_reprojection_errors(
                H,
                pts0,
                pts1
            )

            inlier_errors = errors[mask]

            if len(inlier_errors) > 0:
                median_error = float(
                    np.median(inlier_errors)
                )

            if (
                inliers >= INLOC_MIN_INLIERS and
                median_error <= INLOC_MAX_MEDIAN_ERROR
            ):
                success = 1

    kp1 = pts_to_keypoints(pts0)
    kp2 = pts_to_keypoints(pts1)

    matches = make_dmatches(len(vis_indices))

    return {
        "method": "LoFTR",
        "kp1": len(kp1),
        "kp2": len(kp2),
        "raw_matches": raw_matches,
        "good_matches": num_matches,
        "inliers": inliers,
        "inlier_ratio": float(inlier_ratio),
        "median_reproj_error": median_error,
        "success": success,
        "time_ms": float(total_time_ms),
        "vis_matches": matches,
        "kp1_obj": kp1,
        "kp2_obj": kp2,
        "H_found": H_found,
    }


def draw_and_save(img1, img2, kp1, kp2, matches, save_path, max_draw=100):

    if kp1 is None or kp2 is None:
        return

    draw_matches = matches[:max_draw]

    vis = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        draw_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(save_path, vis)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    loftr = KF.LoFTR(
        pretrained="indoor"
    ).to(device).eval()

    results_dir = os.path.join(
        OUTPUT_DIR,
        "resultsduc61"
    )

    vis_dir = os.path.join(
        OUTPUT_DIR,
        "vis_loftrduc61"
    )

    ensure_dir(results_dir)
    ensure_dir(vis_dir)

    df = pd.read_csv(PAIRS_CSV)

    all_rows = []
    saved_vis = 0

    for idx, row in tqdm(
        df.iterrows(),
        total=len(df)
    ):

        q = row["query_img"]
        d = row["db_img"]

        rank = int(row["rank"])
        score = float(row["score"])

        is_warmup = idx < WARMUP_PAIRS

        try:

            img1 = resize_img(
                load_gray(q),
                640
            )

            img2 = resize_img(
                load_gray(d),
                640
            )

        except Exception as e:

            if not is_warmup:

                all_rows.append({
                    "pair_index": idx,
                    "query_img": q,
                    "db_img": d,
                    "rank": rank,
                    "retrieval_score": score,
                    "method": "LoFTR",
                    "kp1": 0,
                    "kp2": 0,
                    "raw_matches": 0,
                    "good_matches": 0,
                    "inliers": 0,
                    "inlier_ratio": 0.0,
                    "median_reproj_error": np.inf,
                    "success": 0,
                    "time_ms": 0.0,
                    "H_found": 0,
                    "error": str(e),
                })

            continue

        res = run_loftr(
            img1,
            img2,
            loftr,
            device
        )

        if not is_warmup:

            all_rows.append({
                "pair_index": idx,
                "query_img": q,
                "db_img": d,
                "rank": rank,
                "retrieval_score": score,
                "method": "LoFTR",
                "kp1": res["kp1"],
                "kp2": res["kp2"],
                "raw_matches": res["raw_matches"],
                "good_matches": res["good_matches"],
                "inliers": res["inliers"],
                "inlier_ratio": res["inlier_ratio"],
                "median_reproj_error": res["median_reproj_error"],
                "success": res["success"],
                "time_ms": res["time_ms"],
                "H_found": res["H_found"],
                "error": "",
            })

        if (
            not is_warmup and
            saved_vis < SAVE_VIS_PER_METHOD and
            res["good_matches"] > 0
        ):

            qname = Path(q).stem
            dname = Path(d).stem

            save_path = os.path.join(
                vis_dir,
                f"{idx:05d}_r{rank}_{qname}__{dname}.jpg"
            )

            draw_and_save(
                img1,
                img2,
                res["kp1_obj"],
                res["kp2_obj"],
                res["vis_matches"],
                save_path
            )

            saved_vis += 1

    results_df = pd.DataFrame(all_rows)

    results_csv = os.path.join(
        results_dir,
        "loftr_resultsduc61.csv"
    )

    results_df.to_csv(
        results_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\\nSaved results to: {results_csv}")

    summary = (
        results_df.groupby("method")[
            [
                "good_matches",
                "inliers",
                "inlier_ratio",
                "median_reproj_error",
                "success",
                "time_ms"
            ]
        ]
        .mean()
        .round(3)
    )

    print("\\n===== Mean Summary =====")
    print(summary)

    success_rate = (
        results_df["success"].mean() * 100.0
    )

    print(
        f"\\nInLoc Success Rate = "
        f"{success_rate:.2f}%"
    )


if __name__ == "__main__":
    main()
