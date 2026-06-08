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
import torch
import kornia
import kornia.feature as KF

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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_image_color(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img
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
    h, w = src_shape
    corners = np.array([
        [0.0, 0.0],
        [w - 1.0, 0.0],
        [w - 1.0, h - 1.0],
        [0.0, h - 1.0],
    ], dtype=np.float32).reshape(-1, 1, 2)
    try:
        est = cv2.perspectiveTransform(corners, H_est)
        gt  = cv2.perspectiveTransform(corners, H_gt)
        errs = np.linalg.norm((est - gt).reshape(-1, 2), axis=1)
        return float(np.mean(errs))
    except Exception:
        return float("nan")


def error_auc(errors: Sequence[float], thresholds: Sequence[float]) -> Dict[str, float]:
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
        out[f"auc@{int(t)}"] = float(np.trapz(y, x) / float(t))
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
        pts0, pts1,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
        maxIters=max_iters,
        confidence=confidence,
    )
    return H, mask


def success_rule(dataset: str, H: Optional[np.ndarray], num_inliers: int, median_err: float) -> int:
    dataset_low = dataset.lower()
    if H is None or math.isnan(median_err):
        return 0
    if dataset_low == "hpatches":
        n_min, tau = 10, 3.0
    else:
        n_min, tau = 15, 5.0
    return int(num_inliers >= n_min and median_err <= tau)


def overlay_registration(img0: np.ndarray, img1: np.ndarray, H: Optional[np.ndarray]) -> np.ndarray:
    h1, w1 = img1.shape[:2]
    if H is None:
        warped = np.zeros_like(img1)
    else:
        warped = cv2.warpPerspective(img0, H, (w1, h1))
    overlay = cv2.addWeighted(img1, 0.5, warped, 0.5, 0)
    return np.concatenate([img1, warped, overlay], axis=1)


def put_title(img: np.ndarray, title: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return out



def pts_to_keypoints(pts: np.ndarray) -> List[cv2.KeyPoint]:
    return [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in pts]


def make_dmatch_list(n: int) -> List[cv2.DMatch]:
    return [cv2.DMatch(i, i, 0.0) for i in range(n)]


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
        img0, list(kpts0),
        img1, list(kpts1),
        list(matches), None,
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
        "success", "runtime_ms", "H_found", "H_gt_found",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))


def write_run_note(note_path: Path, args: argparse.Namespace, results: Sequence[PairResult]) -> None:
    ensure_dir(note_path.parent)
    num_pairs  = len(results)
    num_success = sum(r.success for r in results)
    avg_runtime = float(np.mean([r.runtime_ms for r in results])) if results else float("nan")
    avg_matches = float(np.mean([r.num_matches for r in results])) if results else float("nan")
    avg_inliers = float(np.mean([r.num_inliers for r in results])) if results else float("nan")

    corner_vals   = [r.corner_error for r in results if not math.isnan(r.corner_error)]
    mean_corner   = float(np.mean(corner_vals))   if corner_vals else float("nan")
    median_corner = float(np.median(corner_vals)) if corner_vals else float("nan")

    hpatches_results = [r for r in results if r.dataset.lower() == "hpatches"]
    aucs = error_auc([r.corner_error for r in hpatches_results], thresholds=[3, 5, 10]) if hpatches_results else {}

    lines = [
        "# 实验记录\n",
        f"- 方法: loftr (weights={args.weights})\n",
        f"- 输入 pairs_csv: {args.pairs_csv}\n",
        f"- 输出目录: {args.output_dir}\n",
        f"- 图像长边: {args.long_edge}\n",
        f"- 不放大: {not args.allow_enlarge}\n",
        f"- LoFTR confidence 阈值: {args.loftr_conf_threshold}\n",
        f"- 最大匹配数 (RANSAC前): {args.max_matches}\n",
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
        "num_pairs":          len(results),
        "num_success":        int(sum(r.success for r in results)),
        "success_rate":       float(sum(r.success for r in results) / len(results)) if results else float("nan"),
        "avg_matches":        float(np.mean([r.num_matches for r in results])) if results else float("nan"),
        "avg_inliers":        float(np.mean([r.num_inliers for r in results])) if results else float("nan"),
        "avg_runtime_ms":     float(np.mean([r.runtime_ms for r in results])) if results else float("nan"),
        "mean_corner_error":  float(np.mean(corner_vals))   if corner_vals else float("nan"),
        "median_corner_error":float(np.median(corner_vals)) if corner_vals else float("nan"),
    }
    if hpatches_results:
        aucs = error_auc([r.corner_error for r in hpatches_results], thresholds=[3, 5, 10])
        metrics.update(aucs)
        for k, v in aucs.items():
            if not math.isnan(v):
                metrics[k + "_pct"] = v * 100.0

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def load_pairs_csv(path: str) -> List[PairRecord]:
    pairs: List[PairRecord] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"pair_id", "dataset", "scene", "img0_path", "img1_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"pairs_csv missing columns: {sorted(missing)}")
        for row in reader:
            pairs.append(PairRecord(
                pair_id=row["pair_id"],
                dataset=row["dataset"],
                scene=row["scene"],
                img0_path=row["img0_path"],
                img1_path=row["img1_path"],
            ))
    return pairs



def gray_to_loftr_tensor(gray: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert HxW uint8 numpy gray image to (1,1,H,W) float32 tensor in [0,1]."""
    t = torch.from_numpy(gray).float() / 255.0          # (H, W)
    return t.unsqueeze(0).unsqueeze(0).to(device)        # (1, 1, H, W)


def loftr_match(
    loftr: KF.LoFTR,
    gray0: np.ndarray,
    gray1: np.ndarray,
    device: torch.device,
    conf_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  
    t0 = gray_to_loftr_tensor(gray0, device)
    t1 = gray_to_loftr_tensor(gray1, device)

    with torch.no_grad():
        batch = {"image0": t0, "image1": t1}
        out = loftr(batch)   # 必须接返回值

    if out is None:
        out = batch

    pts0 = out["keypoints0"].cpu().numpy()
    pts1 = out["keypoints1"].cpu().numpy()
    conf = out["confidence"].cpu().numpy()
    # optional confidence filter
    if conf_threshold > 0.0:
        keep = conf >= conf_threshold
        pts0, pts1, conf = pts0[keep], pts1[keep], conf[keep]

    return pts0, pts1, conf




def process_pair(
    pair: PairRecord,
    loftr: KF.LoFTR,
    device: torch.device,
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
        img0_color, long_edge=args.long_edge, round_to=args.round_to,
        no_enlarge=not args.allow_enlarge,
    )
    img1_color = resize_keep_long_edge(
        img1_color, long_edge=args.long_edge, round_to=args.round_to,
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
                H_gt_orig, orig0_shape, resized0_shape, orig1_shape, resized1_shape,
            )

    t_start = time.perf_counter()

    pts0_np, pts1_np, conf_np = loftr_match(
        loftr, img0_gray, img1_gray, device,
        conf_threshold=args.loftr_conf_threshold,
    )

    num_kpts0 = len(pts0_np)
    num_kpts1 = len(pts1_np)

    if len(conf_np) > 0:
        order = np.argsort(-conf_np)
        pts0_np = pts0_np[order]
        pts1_np = pts1_np[order]
        conf_np = conf_np[order]

    if len(pts0_np) > args.max_matches:
        pts0_np = pts0_np[: args.max_matches]
        pts1_np = pts1_np[: args.max_matches]
        conf_np = conf_np[: args.max_matches]

    num_matches = len(pts0_np)

    if num_matches < 4:
        H = None
        inlier_mask_bool = np.zeros(num_matches, dtype=bool)
        num_inliers = 0
        inlier_ratio = 0.0
        median_err = float("nan")
        mean_err   = float("nan")
        corner_err = float("nan")
    else:
        pts0_cv = pts0_np.reshape(-1, 1, 2).astype(np.float32)
        pts1_cv = pts1_np.reshape(-1, 1, 2).astype(np.float32)

        H, mask = estimate_homography(
            pts0_cv, pts1_cv,
            ransac_reproj_threshold=args.ransac_reproj_threshold,
            max_iters=args.ransac_max_iters,
            confidence=args.ransac_confidence,
        )
        if H is None or mask is None:
            inlier_mask_bool = np.zeros(num_matches, dtype=bool)
            num_inliers = 0
            inlier_ratio = 0.0
            median_err   = float("nan")
            mean_err     = float("nan")
            corner_err   = float("nan")
        else:
            inlier_mask_bool = mask.reshape(-1).astype(bool)
            num_inliers  = int(np.sum(inlier_mask_bool))
            inlier_ratio = float(num_inliers / num_matches)
            median_err, mean_err = compute_reprojection_errors(H, pts0_cv, pts1_cv, mask)
            corner_err = (
                compute_corner_error(H, H_gt_resized, resized0_shape)
                if H_gt_resized is not None else float("nan")
            )

    runtime_ms = (time.perf_counter() - t_start) * 1000.0
    success = success_rule(pair.dataset, H, num_inliers, median_err)


    kpts0_all = pts_to_keypoints(pts0_np)
    kpts1_all = pts_to_keypoints(pts1_np)
    matches_all: List[cv2.DMatch] = make_dmatch_list(num_matches)

    inlier_idx = np.where(inlier_mask_bool)[0] if num_matches >= 4 else np.array([], dtype=int)
    kpts0_inlier = pts_to_keypoints(pts0_np[inlier_idx]) if len(inlier_idx) else []
    kpts1_inlier = pts_to_keypoints(pts1_np[inlier_idx]) if len(inlier_idx) else []
    matches_inlier: List[cv2.DMatch] = make_dmatch_list(len(inlier_idx))

    raw_vis_path    = vis_raw_dir    / f"{pair.pair_id}_raw.jpg"
    inlier_vis_path = vis_inlier_dir / f"{pair.pair_id}_inlier.jpg"
    reg_vis_path    = reg_dir        / f"{pair.pair_id}_reg.jpg"
    json_path       = json_dir       / f"{pair.pair_id}.json"

    save_match_vis(
        img0_color, kpts0_all, img1_color, kpts1_all, matches_all,
        raw_vis_path,
        title=f"{pair.pair_id} | raw matches={num_matches}",
    )
    save_match_vis(
        img0_color, kpts0_inlier, img1_color, kpts1_inlier, matches_inlier,
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
        method=f"loftr_{args.weights}",
        img0_name=Path(pair.img0_path).name,
        img1_name=Path(pair.img1_path).name,
        num_keypoints0=num_kpts0,
        num_keypoints1=num_kpts1,
        num_matches=num_matches,
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
    json_payload.update({
        "img0_path": pair.img0_path,
        "img1_path": pair.img1_path,
        "homography": H.tolist() if H is not None else None,
        "homography_gt_resized": H_gt_resized.tolist() if H_gt_resized is not None else None,
    })
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return result



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LoFTR baseline matching experiments.")
    parser.add_argument("--pairs_csv",   type=str, required=True,
                        help="CSV with pair_id,dataset,scene,img0_path,img1_path")
    parser.add_argument("--output_dir",  type=str, required=True)

    parser.add_argument("--weights", type=str, default="outdoor", choices=["outdoor", "indoor"],
                        help="LoFTR pretrained weights to use (default: outdoor)")
    parser.add_argument("--loftr_conf_threshold", type=float, default=0.0,
                        help="Minimum LoFTR match confidence to keep (0 = keep all)")

    parser.add_argument("--long_edge",   type=int,   default=640)
    parser.add_argument("--round_to",    type=int,   default=8)
    parser.add_argument("--allow_enlarge", action="store_true",
                        help="Allow enlarging images smaller than target long edge")
    parser.add_argument("--max_matches", type=int,   default=1000,
                        help="Keep top-N matches (by confidence) before RANSAC")

    parser.add_argument("--max_keypoints", type=int, default=2000,
                        help="(Unused for LoFTR, kept for CLI parity)")
    parser.add_argument("--orb_ratio",   type=float, default=0.90,
                        help="(Unused for LoFTR, kept for CLI parity)")
    parser.add_argument("--sift_ratio",  type=float, default=0.75,
                        help="(Unused for LoFTR, kept for CLI parity)")

    parser.add_argument("--ransac_reproj_threshold", type=float, default=3.0)
    parser.add_argument("--ransac_max_iters",        type=int,   default=2000)
    parser.add_argument("--ransac_confidence",       type=float, default=0.995)

    parser.add_argument("--device", type=str, default="",
                        help="PyTorch device string, e.g. 'cuda', 'cpu'. "
                             "Auto-detected if empty.")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Run first N pairs as warm-up and do not include them in summary")
    parser.add_argument("--limit",  type=int, default=0,
                        help="If > 0, only run first N pairs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pairs = load_pairs_csv(args.pairs_csv)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        raise ValueError("No pairs loaded from pairs_csv.")


    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    print(f"[Info] Loading LoFTR ({args.weights} weights)…")
    loftr_model = KF.LoFTR(pretrained=args.weights).to(device).eval()
    print("[Info] LoFTR loaded.")

    output_dir    = Path(args.output_dir)
    vis_raw_dir   = output_dir / "pair_vis" / "raw_matches"
    vis_inlier_dir= output_dir / "pair_vis" / "inlier_matches"
    reg_dir       = output_dir / "pair_vis" / "registration"
    json_dir      = output_dir / "pair_json"
    for d in [vis_raw_dir, vis_inlier_dir, reg_dir, json_dir]:
        ensure_dir(d)

    results: List[PairResult] = []
    warmup_n = max(0, min(args.warmup, len(pairs)))

    if warmup_n > 0:
        print(f"[Warm-up] Running first {warmup_n} pairs without recording summary…")
        for pair in pairs[:warmup_n]:
            _ = process_pair(
                pair, loftr_model, device, args,
                vis_raw_dir, vis_inlier_dir, reg_dir, json_dir,
            )

    formal_pairs = pairs[warmup_n:]
    print(f"[Run] method=loftr_{args.weights}, total_formal_pairs={len(formal_pairs)}")
    for idx, pair in enumerate(formal_pairs, start=1):
        result = process_pair(
            pair, loftr_model, device, args,
            vis_raw_dir, vis_inlier_dir, reg_dir, json_dir,
        )
        results.append(result)
        print(
            f"[{idx}/{len(formal_pairs)}] {pair.pair_id} | "
            f"matches={result.num_matches} inliers={result.num_inliers} "
            f"median_err={result.median_reproj_error:.3f} "
            f"corner_err={result.corner_error:.3f} success={result.success} "
            f"time={result.runtime_ms:.2f}ms"
        )

    summary_path  = output_dir / "summary.csv"
    note_path     = output_dir / "run_note.md"
    metrics_path  = output_dir / "metrics.json"
    write_summary_csv(summary_path, results)
    write_run_note(note_path, args, results)
    write_metrics_json(metrics_path, results)

    print(f"\n[Done] Summary saved to:        {summary_path}")
    print(f"[Done] Run note saved to:        {note_path}")
    print(f"[Done] Metrics JSON saved to:    {metrics_path}")
    print(f"[Done] Visualisations saved under: {output_dir / 'pair_vis'}")


if __name__ == "__main__":
    main()
