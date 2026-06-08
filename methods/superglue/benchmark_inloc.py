#!/usr/bin/env python3
"""Run SuperPoint + SuperGlue benchmark on InLoc."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import torch

from superglue_benchmark_lib import (
    MAX_FINAL_MATCHES,
    PairItem,
    choose_weights,
    create_matching,
    infer_pair,
    load_and_prepare_image,
    maybe_visualize,
    print_summary,
    resolve_relative_or_absolute_path,
    summarize_results,
    warmup,
    write_results_csv,
)


DEFAULT_QUERY_ROOT = Path("/root/autodl-tmp/Inloc/iphone7")
DEFAULT_DATABASE_ROOT = Path("/root/autodl-tmp/Inloc/cutouts_imageonly")
DEFAULT_OUTPUT_DIR = Path("/root/autodl-tmp/outputs/inloc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SuperPoint + SuperGlue on InLoc.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query_root", type=Path, default=DEFAULT_QUERY_ROOT)
    parser.add_argument("--database_root", type=Path, default=DEFAULT_DATABASE_ROOT)
    parser.add_argument("--pairs_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--superglue_weights", choices=["auto", "indoor", "outdoor"], default="auto")
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


def parse_inloc_pairs_file(pair_file: Path, query_root: Path, database_root: Path) -> List[PairItem]:
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
                image0 = resolve_relative_or_absolute_path(row["image0"], query_root, database_root)
                image1 = resolve_relative_or_absolute_path(row["image1"], query_root, database_root)
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
                if len(parts) == 2:
                    pair_id = f"pair_{line_idx:06d}"
                    image0_str, image1_str = parts
                elif len(parts) >= 3:
                    pair_id, image0_str, image1_str = parts[:3]
                else:
                    raise ValueError(
                        f"Invalid line {line_idx} in {pair_file}: expected either 'image0 image1' or 'pair_id image0 image1'"
                    )
                image0 = resolve_relative_or_absolute_path(image0_str, query_root, database_root)
                image1 = resolve_relative_or_absolute_path(image1_str, query_root, database_root)
                pairs.append(PairItem(pair_id, "inloc", image0, image1))

    if not pairs:
        raise RuntimeError(f"No InLoc pairs loaded from {pair_file}")
    return pairs


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = args.output_dir / "viz"
    if args.save_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    pairs = parse_inloc_pairs_file(args.pairs_file, args.query_root, args.database_root)
    weights = choose_weights("inloc", args.superglue_weights)
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"

    print(f"Dataset: inloc")
    print(f"Query root: {args.query_root}")
    print(f"Database root: {args.database_root}")
    print(f"Pairs file: {args.pairs_file}")
    print(f"Pairs: {len(pairs)}")
    print(f"Device: {device}")
    print(f"SuperGlue weights: {weights}")

    matching = create_matching(device, weights, args)
    warmup_count = min(args.warmup_pairs, len(pairs))
    warmup(pairs, matching, device, warmup_count, args.max_final_matches)
    formal_pairs = pairs[warmup_count:]

    results = []
    for idx, pair in enumerate(formal_pairs, start=1):
        image0 = load_and_prepare_image(pair.image0)
        image1 = load_and_prepare_image(pair.image1)
        result, aux = infer_pair(matching, device, image0, image1, "inloc", args.max_final_matches)
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
            f"success={result.success} runtime={result.runtime_ms:.2f} ms"
        )

    csv_path = args.output_dir / "inloc_superglue_results.csv"
    summary_path = args.output_dir / "inloc_superglue_summary.json"
    write_results_csv(csv_path, results)
    summary = summarize_results("inloc", results, weights, summary_path)
    print_summary(summary)


if __name__ == "__main__":
    main()
