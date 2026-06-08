import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ========= 这里改路径 =========
PAIRS_CSV = r"D:\xidian\study\homework\CV\InLoc_exp\pairs_duc1_top20.csv"
OUTPUT_DIR = r"D:\xidian\study\homework\CV\InLoc_exp"
TOPK_PER_QUERY = 20
SAVE_VIS_PER_METHOD = 100    # 每种方法最多保存多少张可视化图，避免磁盘太大
# ============================


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_gray(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"读取失败: {path}")
    return img


def run_orb(img1, img2):
    t0 = time.perf_counter()

    orb = cv2.ORB_create(
        nfeatures=2000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        patchSize=31,
        fastThreshold=20
    )

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    detect_time_ms = (time.perf_counter() - t0) * 1000

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return {
            "method": "ORB",
            "kp1": len(kp1) if kp1 is not None else 0,
            "kp2": len(kp2) if kp2 is not None else 0,
            "raw_matches": 0,
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "time_ms": detect_time_ms,
            "vis_matches": [],
            "H_found": 0,
        }

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    t1 = time.perf_counter()
    knn_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m_n in knn_matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)

    match_time_ms = (time.perf_counter() - t1) * 1000
    total_time_ms = detect_time_ms + match_time_ms

    inliers = 0
    H_found = 0
    inlier_ratio = 0.0
    vis_matches = good

    if len(good) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None and mask is not None:
            H_found = 1
            inliers = int(mask.sum())
            inlier_ratio = inliers / max(len(good), 1)
            vis_matches = [m for m, keep in zip(good, mask.ravel().tolist()) if keep]

    return {
        "method": "ORB",
        "kp1": len(kp1),
        "kp2": len(kp2),
        "raw_matches": len(knn_matches),
        "good_matches": len(good),
        "inliers": inliers,
        "inlier_ratio": float(inlier_ratio),
        "time_ms": float(total_time_ms),
        "vis_matches": vis_matches,
        "kp1_obj": kp1,
        "kp2_obj": kp2,
        "H_found": H_found,
    }


def run_sift(img1, img2):
    t0 = time.perf_counter()

    sift = cv2.SIFT_create(
        nfeatures=2000,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6
    )

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    detect_time_ms = (time.perf_counter() - t0) * 1000

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return {
            "method": "SIFT",
            "kp1": len(kp1) if kp1 is not None else 0,
            "kp2": len(kp2) if kp2 is not None else 0,
            "raw_matches": 0,
            "good_matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "time_ms": detect_time_ms,
            "vis_matches": [],
            "H_found": 0,
        }

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    t1 = time.perf_counter()
    knn_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m_n in knn_matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)

    match_time_ms = (time.perf_counter() - t1) * 1000
    total_time_ms = detect_time_ms + match_time_ms

    inliers = 0
    H_found = 0
    inlier_ratio = 0.0
    vis_matches = good

    if len(good) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is not None and mask is not None:
            H_found = 1
            inliers = int(mask.sum())
            inlier_ratio = inliers / max(len(good), 1)
            vis_matches = [m for m, keep in zip(good, mask.ravel().tolist()) if keep]

    return {
        "method": "SIFT",
        "kp1": len(kp1),
        "kp2": len(kp2),
        "raw_matches": len(knn_matches),
        "good_matches": len(good),
        "inliers": inliers,
        "inlier_ratio": float(inlier_ratio),
        "time_ms": float(total_time_ms),
        "vis_matches": vis_matches,
        "kp1_obj": kp1,
        "kp2_obj": kp2,
        "H_found": H_found,
    }


def draw_and_save(img1, img2, kp1, kp2, matches, save_path, max_draw=100):
    if kp1 is None or kp2 is None:
        return
    draw_matches = matches[:max_draw]
    vis = cv2.drawMatches(
        img1, kp1, img2, kp2, draw_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(save_path, vis)


def main():
    results_dir = os.path.join(OUTPUT_DIR, "results")
    vis_dir = os.path.join(OUTPUT_DIR, "vis")
    ensure_dir(results_dir)
    ensure_dir(vis_dir)
    ensure_dir(os.path.join(vis_dir, "orb"))
    ensure_dir(os.path.join(vis_dir, "sift"))

    df = pd.read_csv(PAIRS_CSV)
    print(df.head(3)[["query_img", "db_img"]])

    # 保险一点：每个 query 只保留前 topk
    df = df.sort_values(["query_img", "rank"]).groupby("query_img").head(TOPK_PER_QUERY).reset_index(drop=True)

    all_rows = []
    saved_orb = 0
    saved_sift = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Matching all pairs"):
        query_img = row["query_img"]
        db_img = row["db_img"]
        rank = int(row["rank"])
        score = float(row["score"])

        try:
            img1 = load_gray(query_img)
            img2 = load_gray(db_img)
        except Exception as e:
            all_rows.append({
                "pair_index": idx,
                "query_img": query_img,
                "db_img": db_img,
                "rank": rank,
                "retrieval_score": score,
                "method": "ORB",
                "kp1": 0, "kp2": 0,
                "raw_matches": 0, "good_matches": 0, "inliers": 0,
                "inlier_ratio": 0.0, "time_ms": 0.0,
                "H_found": 0,
                "error": str(e),
            })
            all_rows.append({
                "pair_index": idx,
                "query_img": query_img,
                "db_img": db_img,
                "rank": rank,
                "retrieval_score": score,
                "method": "SIFT",
                "kp1": 0, "kp2": 0,
                "raw_matches": 0, "good_matches": 0, "inliers": 0,
                "inlier_ratio": 0.0, "time_ms": 0.0,
                "H_found": 0,
                "error": str(e),
            })
            continue

        # ===== ORB =====
        orb_res = run_orb(img1, img2)
        all_rows.append({
            "pair_index": idx,
            "query_img": query_img,
            "db_img": db_img,
            "rank": rank,
            "retrieval_score": score,
            "method": "ORB",
            "kp1": orb_res["kp1"],
            "kp2": orb_res["kp2"],
            "raw_matches": orb_res["raw_matches"],
            "good_matches": orb_res["good_matches"],
            "inliers": orb_res["inliers"],
            "inlier_ratio": orb_res["inlier_ratio"],
            "time_ms": orb_res["time_ms"],
            "H_found": orb_res["H_found"],
            "error": "",
        })

        if saved_orb < SAVE_VIS_PER_METHOD and orb_res["good_matches"] > 0:
            qname = Path(query_img).stem
            dname = Path(db_img).stem
            save_path = os.path.join(vis_dir, "orb", f"{idx:05d}_r{rank}_{qname}__{dname}.jpg")
            draw_and_save(img1, img2, orb_res.get("kp1_obj"), orb_res.get("kp2_obj"), orb_res["vis_matches"], save_path)
            saved_orb += 1

        # ===== SIFT =====
        sift_res = run_sift(img1, img2)
        all_rows.append({
            "pair_index": idx,
            "query_img": query_img,
            "db_img": db_img,
            "rank": rank,
            "retrieval_score": score,
            "method": "SIFT",
            "kp1": sift_res["kp1"],
            "kp2": sift_res["kp2"],
            "raw_matches": sift_res["raw_matches"],
            "good_matches": sift_res["good_matches"],
            "inliers": sift_res["inliers"],
            "inlier_ratio": sift_res["inlier_ratio"],
            "time_ms": sift_res["time_ms"],
            "H_found": sift_res["H_found"],
            "error": "",
        })

        if saved_sift < SAVE_VIS_PER_METHOD and sift_res["good_matches"] > 0:
            qname = Path(query_img).stem
            dname = Path(db_img).stem
            save_path = os.path.join(vis_dir, "sift", f"{idx:05d}_r{rank}_{qname}__{dname}.jpg")
            draw_and_save(img1, img2, sift_res.get("kp1_obj"), sift_res.get("kp2_obj"), sift_res["vis_matches"], save_path)
            saved_sift += 1

    results_df = pd.DataFrame(all_rows)
    results_csv = os.path.join(results_dir, f"orb_sift_results_top{TOPK_PER_QUERY}.csv")
    results_df.to_csv(results_csv, index=False, encoding="utf-8-sig")

    print(f"\nSaved results to: {results_csv}")

    summary = (
        results_df.groupby("method")[["good_matches", "inliers", "inlier_ratio", "time_ms"]]
        .mean()
        .round(3)
    )
    print("\n===== Mean Summary =====")
    print(summary)


if __name__ == "__main__":
    main()