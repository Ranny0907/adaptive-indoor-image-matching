#!/usr/bin/env python3
"""Run SuperPoint + SuperGlue benchmark on HPatches."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from superglue_benchmark_lib import (
    MAX_FINAL_MATCHES,
    PairItem,
    choose_weights,
    create_matching,
    infer_pair,
    load_and_prepare_image_with_info,
    load_hpatches_homography,
    maybe_visualize,
    print_summary,
    resolve_existing_image_stem,
    scale_homography_to_resized_images,
    summarize_results,
    warmup,
    write_results_csv,
)


DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/hpatches_data/hpatches-sequences-release")
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/outputs/hpatches")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SuperPoint + SuperGlue on HPatches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hpatches_split", choices=["all", "i", "v"], default="all")
    parser.add_argument("--superglue_weights", choices=["auto", "outdoor", "indoor"], default="auto")
    parser.add_argument("--max_keypoints", type=int, default=1024)
    parser.add_argument("--keypoint_threshold", type=float, default=0.005)
    parser.add_argument("--nms_radius", type=int, default=4)
    parser.add_argument("--sinkhorn_iterations", type=int, default=20)
    parser.add_argument("--match_threshold", type=float, default=0.20)
    parser.add_argument("--max_final_matches", type=int, default=MAX_FINAL_MATCHES)
    parser.add_argument("--warmup_pairs", type=int, default=5)
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--viz_max_pairs", type=int, default=20)
    parser.add_argument("--show_keypoints", action="store_true")
    parser.add_argument("--force_cpu", action="store_true")
    return parser.parse_args()


def discover_hpatches_pairs(data_root: Path, split: str) -> List[PairItem]:
    pairs: List[PairItem] = []
    prefixes = {"all": ("i_", "v_"), "i": ("i_",), "v": ("v_",)}[split]

    for seq_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if not seq_dir.name.startswith(prefixes):
            continue
        image0 = resolve_existing_image_stem(seq_dir / "1")
        for target_idx in range(2, 7):
            image1 = resolve_existing_image_stem(seq_dir / str(target_idx))
            gt_homography_path = seq_dir / f"H_1_{target_idx}"
            if not gt_homography_path.exists():
                raise FileNotFoundError(f"Missing GT homography: {gt_homography_path}")
            pair_id = f"{seq_dir.name}_1_{target_idx}"
            pairs.append(
                PairItem(
                    pair_id=pair_id,
                    dataset="hpatches",
                    image0=image0,
                    image1=image1,
                    gt_homography_path=gt_homography_path,
                )
            )

    if not pairs:
        raise RuntimeError(f"No HPatches pairs found under {data_root}")
    return pairs


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = args.output_dir / "viz"
    if args.save_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_hpatches_pairs(args.data_root, args.hpatches_split)
    weights = choose_weights("hpatches", args.superglue_weights)
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"

    print(f"Dataset: hpatches")
    print(f"Data root: {args.data_root}")
    print(f"Pairs: {len(pairs)}")
    print(f"Device: {device}")
    print(f"SuperGlue weights: {weights}")

    matching = create_matching(device, weights, args)
    warmup_count = min(args.warmup_pairs, len(pairs))
    warmup(pairs, matching, device, warmup_count, args.max_final_matches)
    formal_pairs = pairs[warmup_count:]

    results = []
    for idx, pair in enumerate(formal_pairs, start=1):
        image0, original0_hw, resized0_hw = load_and_prepare_image_with_info(pair.image0)
        image1, original1_hw, resized1_hw = load_and_prepare_image_with_info(pair.image1)
        gt_homography = load_hpatches_homography(pair.gt_homography_path)
        gt_homography_resized = scale_homography_to_resized_images(
            gt_homography,
            original0_hw,
            resized0_hw,
            original1_hw,
            resized1_hw,
        )
        result, aux = infer_pair(
            matching,
            device,
            image0,
            image1,
            "hpatches",
            args.max_final_matches,
            eval_homography=gt_homography_resized,
        )
        result.pair_id = pair.pair_id
        results.append(result)

        maybe_visualize(
            idx - 1,
            args.save_viz,
            args.viz_max_pairs,
            args.show_keypoints,
            image0,
            image1,
            aux,
            result,
            viz_dir,
        )

        print(
            f"[{idx}/{len(formal_pairs)}] {pair.pair_id} | "
            f"matches={result.num_matches} inliers={result.num_inliers} "
            f"median_err={result.median_reproj_error:.3f} "
            f"corner_err={result.corner_error:.3f} success={result.success} "
            f"runtime={result.runtime_ms:.2f} ms"
        )

    csv_path = args.output_dir / "hpatches_superglue_results.csv"
    summary_path = args.output_dir / "hpatches_superglue_summary.json"
    write_results_csv(csv_path, results)
    summary = summarize_results("hpatches", results, weights, summary_path)
    print_summary(summary)


if __name__ == "__main__":
    main()
