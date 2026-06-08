#!/usr/bin/env python3
"""Streamlit demo for the adaptive InLoc image matching system."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import kornia.feature as KF

ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
METHODS_DIR = ROOT / "methods"
RESULTS_DIR = ROOT / "results"
SUPERGLUE_DIR = METHODS_DIR / "superglue"
if str(SUPERGLUE_DIR) not in sys.path:
    sys.path.insert(0, str(SUPERGLUE_DIR))

from adaptive_cascade_inloc import (  # noqa: E402
    INLOC_MAX_MEDIAN_ERROR,
    INLOC_MIN_INLIERS,
    InLocPair,
    compute_reprojection_errors,
    extract_superpoint_features,
    group_by_query,
    infer_pair_from_superpoint_features,
    load_pairs,
    precompute_database_features,
    resolve_path,
    run_loftr_pair,
)
from superglue_benchmark_lib import (  # noqa: E402
    MAX_FINAL_MATCHES,
    create_matching,
    infer_pair,
    load_and_prepare_image,
    save_match_visualization,
    frame_to_tensor,
)


RESULT_ROOT = RESULTS_DIR / "adaptive_cascade_results"
SUMMARY_CSV = RESULT_ROOT / "adaptive_strategy_summary_with_policy.csv"
SCENE_CSV = RESULT_ROOT / "adaptive_strategy_scene_split_with_cache.csv"
PAIRS_CSV = DATASETS_DIR / "baseline" / "standardized_20260525" / "pairs" / "pairs_inloc_netvlad40.csv"
DEMO_OUTPUT_DIR = RESULTS_DIR / "adaptive_demo_outputs"

STRATEGIES: Dict[str, str] = {
    "SuperGlue only + cache": "inloc_sg40_only_cached",
    "LoFTR only": "inloc_loftr40_only_real",
    "SG Top5 -> LoFTR + cache": "inloc_sg5_loftr40_cached",
    "SG Top10 -> LoFTR + cache": "inloc_sg10_loftr40_cached",
    "SG Top20 -> LoFTR + cache": "inloc_sg20_loftr40_cached",
    "SG Top40 -> LoFTR + cache": "inloc_sg40_loftr40_cached",
    "Confidence dispatch + cache": "inloc_confidence_dispatch_cached",
    "Confidence then fixed + cache": "inloc_confidence_then_fixed_cached",
    "Confidence then fixed strict + cache": "inloc_confidence_then_fixed_strict_cached",
}


st.set_page_config(
    page_title="Adaptive InLoc Matching Demo",
    page_icon="CV",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 14px;
    }
    div[data-testid="stMetric"] label {color: #475569;}
    .step-row {
        display: grid;
        grid-template-columns: repeat(5, minmax(120px, 1fr));
        gap: 8px;
        margin: 8px 0 18px 0;
    }
    .step {
        border: 1px solid #dbe3ef;
        border-radius: 8px;
        padding: 10px 12px;
        background: #ffffff;
        min-height: 74px;
    }
    .step b {display: block; color: #0f172a; margin-bottom: 4px;}
    .step span {color: #64748b; font-size: 0.88rem;}
    .tag {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 999px;
        background: #eef2ff;
        color: #3730a3;
        font-size: 0.78rem;
        margin-left: 6px;
    }
    .note {
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def ms(value: float) -> str:
    return f"{value:.2f} ms"


def stage_label(stage: str) -> str:
    if stage == "SuperGlue":
        return "SuperGlue 直接成功"
    if stage == "LoFTR":
        return "LoFTR 困难样本补救"
    return "未通过几何验证"


@st.cache_data(show_spinner=False)
def read_summary() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_CSV)
    df["success_rate_percent"] = df["success_rate"] * 100
    df["loftr_trigger_rate_percent"] = df["loftr_trigger_rate"] * 100
    return df


@st.cache_data(show_spinner=False)
def read_scene_summary() -> pd.DataFrame:
    df = pd.read_csv(SCENE_CSV)
    df["success_rate_percent"] = df["success_rate"] * 100
    df["loftr_trigger_rate_percent"] = df["loftr_trigger_rate"] * 100
    return df


@st.cache_data(show_spinner=False)
def read_strategy_outputs(strategy_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    directory = RESULT_ROOT / STRATEGIES[strategy_name]
    query_df = pd.read_csv(directory / "adaptive_query_results.csv")
    attempt_df = pd.read_csv(directory / "adaptive_attempts.csv")
    return query_df, attempt_df


@st.cache_data(show_spinner=False)
def read_formal_pairs() -> List[Tuple[str, List[InLocPair]]]:
    pairs = load_pairs(PAIRS_CSV)
    formal_pairs = pairs[5:]
    return group_by_query(formal_pairs, None, None)


@st.cache_resource(show_spinner=False)
def load_superglue_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = SimpleNamespace(
        nms_radius=4,
        keypoint_threshold=0.005,
        max_keypoints=1024,
        sinkhorn_iterations=20,
        match_threshold=0.20,
    )
    model = create_matching(device, "indoor", args)
    return model, device


@st.cache_resource(show_spinner=False)
def load_loftr_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KF.LoFTR(pretrained="indoor").to(device).eval()
    return model, device


def choose_attempt_for_display(query_row: pd.Series, attempts: pd.DataFrame) -> pd.Series:
    if int(query_row["success"]) == 1:
        final_stage = str(query_row["final_stage"])
        final_rank = int(float(query_row["final_rank"]))
        selected = attempts[(attempts["stage"] == final_stage) & (attempts["rank"] == final_rank)]
        if not selected.empty:
            return selected.iloc[0]
    best = attempts.sort_values(["success", "num_inliers", "num_matches"], ascending=False)
    return best.iloc[0]


def draw_loftr_matches(
    image0: np.ndarray,
    image1: np.ndarray,
    points0: np.ndarray,
    points1: np.ndarray,
    inlier_mask: np.ndarray,
    output_path: Path,
    title_lines: List[str],
) -> None:
    left = cv2.cvtColor(image0, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] :] = right

    offset = np.array([left.shape[1], 0], dtype=np.float32)
    if len(points0) > 140:
        keep = np.linspace(0, len(points0) - 1, 140).astype(int)
    else:
        keep = np.arange(len(points0))

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


def generate_loftr_visualization(row: pd.Series, output_path: Path) -> Tuple[Path, Dict[str, float]]:
    loftr, device = load_loftr_model()
    image0 = load_and_prepare_image(resolve_path(str(row["image0_path"])))
    image1 = load_and_prepare_image(resolve_path(str(row["image1_path"])))

    timer_start = time.perf_counter()
    with torch.inference_mode():
        output = loftr(
            {
                "image0": frame_to_tensor(image0, device),
                "image1": frame_to_tensor(image1, device),
            }
        )
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
    runtime_ms = (time.perf_counter() - timer_start) * 1000.0
    success = int(
        homography is not None
        and int(inlier_mask.sum()) >= INLOC_MIN_INLIERS
        and not math.isnan(median_error)
        and median_error <= INLOC_MAX_MEDIAN_ERROR
    )
    title = [
        "LoFTR geometric verification",
        f"rank={int(row['rank'])} matches={len(points0)} inliers={int(inlier_mask.sum())} success={success}",
        f"median reprojection error={'NaN' if math.isnan(median_error) else f'{median_error:.3f}px'}",
    ]
    draw_loftr_matches(image0, image1, points0, points1, inlier_mask, output_path, title)
    return output_path, {
        "runtime_ms": runtime_ms,
        "matches": float(len(points0)),
        "inliers": float(inlier_mask.sum()),
        "success": float(success),
    }


def generate_superglue_visualization(row: pd.Series, output_path: Path) -> Path:
    matching, device = load_superglue_model()
    image0 = load_and_prepare_image(resolve_path(str(row["image0_path"])))
    image1 = load_and_prepare_image(resolve_path(str(row["image1_path"])))
    result, aux = infer_pair(matching, device, image0, image1, "inloc", MAX_FINAL_MATCHES)
    result.pair_id = str(row["pair_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_match_visualization(output_path, image0, image1, aux, result, show_keypoints=False)
    return output_path


def generate_match_visualization(row: pd.Series) -> Optional[Path]:
    safe_query = str(row["query_img"]).replace(".", "_")
    safe_stage = str(row["stage"]).lower()
    output_path = DEMO_OUTPUT_DIR / f"{safe_query}_rank{int(row['rank']):02d}_{safe_stage}.png"
    if str(row["stage"]) == "SuperGlue":
        return generate_superglue_visualization(row, output_path)
    if str(row["stage"]) == "LoFTR":
        path, _ = generate_loftr_visualization(row, output_path)
        return path
    return None


def run_live_query(
    query_name: str,
    sg_topk: int,
    loftr_topk: int,
    use_cache: bool,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    grouped = dict(read_formal_pairs())
    candidates = grouped[query_name]
    scene = candidates[0].scene

    superglue, sg_device = load_superglue_model() if sg_topk > 0 else (None, "")
    loftr, loftr_device = load_loftr_model() if loftr_topk > 0 else (None, "")

    db_feature_cache = {}
    if use_cache and superglue is not None:
        db_feature_cache, _ = precompute_database_features(superglue, sg_device, [(query_name, candidates)], sg_topk)

    image0_cache: Dict[Path, np.ndarray] = {}

    def load_pair_images(pair: InLocPair) -> Tuple[np.ndarray, np.ndarray]:
        if pair.image0 not in image0_cache:
            image0_cache[pair.image0] = load_and_prepare_image(pair.image0)
        return image0_cache[pair.image0], load_and_prepare_image(pair.image1)

    attempts: List[Dict[str, object]] = []
    success = 0
    final_stage = "failed"
    final_rank: Optional[int] = None
    sg_runtime_ms = 0.0
    loftr_runtime_ms = 0.0
    sg_attempts = 0
    loftr_attempts = 0
    query_features = None

    if superglue is not None:
        if use_cache:
            image0_cache[candidates[0].image0] = load_and_prepare_image(candidates[0].image0)
            query_features, query_feature_ms = extract_superpoint_features(superglue, sg_device, image0_cache[candidates[0].image0])
            sg_runtime_ms += query_feature_ms
        for pair in [p for p in candidates if p.rank <= sg_topk]:
            if use_cache:
                assert query_features is not None
                result, _ = infer_pair_from_superpoint_features(
                    superglue,
                    sg_device,
                    query_features,
                    db_feature_cache[pair.image1],
                    "inloc",
                    MAX_FINAL_MATCHES,
                )
            else:
                image0, image1 = load_pair_images(pair)
                result, _ = infer_pair(superglue, sg_device, image0, image1, "inloc", MAX_FINAL_MATCHES)
            sg_attempts += 1
            sg_runtime_ms += result.runtime_ms
            attempts.append(
                {
                    "stage": "SuperGlue",
                    "rank": pair.rank,
                    "pair_id": pair.pair_id,
                    "image0_path": str(pair.image0),
                    "image1_path": str(pair.image1),
                    "num_matches": result.num_matches,
                    "num_inliers": result.num_inliers,
                    "median_reproj_error": result.median_reproj_error,
                    "success": result.success,
                    "runtime_ms": result.runtime_ms,
                }
            )
            if result.success:
                success = 1
                final_stage = "SuperGlue"
                final_rank = pair.rank
                break

    loftr_triggered = int(success == 0 and loftr_topk > 0)
    if success == 0 and loftr is not None:
        for pair in [p for p in candidates if p.rank <= loftr_topk]:
            image0, image1 = load_pair_images(pair)
            result = run_loftr_pair(loftr, loftr_device, image0, image1, MAX_FINAL_MATCHES)
            loftr_attempts += 1
            loftr_runtime_ms += result.runtime_ms
            attempts.append(
                {
                    "stage": "LoFTR",
                    "rank": pair.rank,
                    "pair_id": pair.pair_id,
                    "image0_path": str(pair.image0),
                    "image1_path": str(pair.image1),
                    "num_matches": result.num_matches,
                    "num_inliers": result.num_inliers,
                    "median_reproj_error": result.median_reproj_error,
                    "success": result.success,
                    "runtime_ms": result.runtime_ms,
                }
            )
            if result.success:
                success = 1
                final_stage = "LoFTR"
                final_rank = pair.rank
                break

    query_result = {
        "query_img": query_name,
        "scene": scene,
        "success": success,
        "final_stage": final_stage,
        "final_rank": final_rank,
        "loftr_triggered": loftr_triggered,
        "sg_attempts": sg_attempts,
        "loftr_attempts": loftr_attempts,
        "sg_runtime_ms": sg_runtime_ms,
        "loftr_runtime_ms": loftr_runtime_ms,
        "total_runtime_ms": sg_runtime_ms + loftr_runtime_ms,
    }
    return query_result, pd.DataFrame(attempts)


def render_metrics(row: pd.Series) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("成功率", pct(float(row["success_rate"])))
    c2.metric("平均在线耗时", ms(float(row["avg_total_runtime_ms"])))
    c3.metric("LoFTR 调用率", pct(float(row["loftr_trigger_rate"])))
    c4.metric("成功 query", f"{int(row['success_count'])}/{int(row['num_queries'])}")


def render_system_flow() -> None:
    st.markdown(
        """
        <div class="step-row">
          <div class="step"><b>1. 查询图像</b><span>输入一张室内照片，模拟在线重定位请求。</span></div>
          <div class="step"><b>2. NetVLAD 候选</b><span>使用已给定 Top-K 候选图，进入几何验证阶段。</span></div>
          <div class="step"><b>3. SuperGlue 快速验证</b><span>优先使用稀疏匹配处理多数简单样本。</span></div>
          <div class="step"><b>4. LoFTR 按需补救</b><span>SuperGlue 失败后触发稠密匹配处理困难样本。</span></div>
          <div class="step"><b>5. 输出定位结果</b><span>根据单应性、内点数和重投影误差判断成功。</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    summary = read_summary()
    scene_summary = read_scene_summary()

    st.title("室内视觉定位自适应图像匹配 Demo")
    st.caption("SuperGlue 快速验证 + LoFTR 困难样本补救 + SuperPoint 数据库特征缓存")
    render_system_flow()

    default_strategy = "SG Top20 -> LoFTR + cache"
    strategy_name = st.sidebar.selectbox(
        "策略",
        list(STRATEGIES.keys()),
        index=list(STRATEGIES.keys()).index(default_strategy),
    )
    selected_summary = summary[summary["strategy"] == strategy_name].iloc[0]
    render_metrics(selected_summary)

    tab_overview, tab_query, tab_live, tab_engineering, tab_files = st.tabs(
        ["策略总览", "单张查询回放", "现场运行", "工程接口", "实验文件"]
    )

    with tab_overview:
        st.subheader("运行模式说明")
        mode_display = pd.DataFrame(
            [
                {
                    "模式": "快速模式",
                    "对应策略": "Confidence dispatch + cache",
                    "成功率": "94.38%",
                    "平均在线耗时": "267.97 ms",
                    "适合场景": "需要更低延迟，允许少量成功率下降",
                },
                {
                    "模式": "均衡模式",
                    "对应策略": "SG Top20 -> LoFTR + cache",
                    "成功率": "97.19%",
                    "平均在线耗时": "374.57 ms",
                    "适合场景": "成功率、耗时和 LoFTR 调用率之间最均衡，建议主推",
                },
                {
                    "模式": "高精度模式",
                    "对应策略": "Confidence then fixed strict + cache",
                    "成功率": "97.47%",
                    "平均在线耗时": "406.04 ms",
                    "适合场景": "优先追求最高成功率，允许略高耗时",
                },
            ]
        )
        st.dataframe(mode_display, use_container_width=True, hide_index=True)
        st.markdown(
            '<p class="note">这三个模式不是重新造三个算法，而是把同一个自适应匹配系统按工程需求配置成低延迟、均衡和高精度三种运行档位。</p>',
            unsafe_allow_html=True,
        )

        st.subheader("策略对比")
        display = summary[
            [
                "strategy",
                "success_rate",
                "avg_total_runtime_ms",
                "loftr_trigger_rate",
                "avg_sg_attempts",
                "avg_loftr_attempts",
            ]
        ].copy()
        display["success_rate"] = display["success_rate"].map(pct)
        display["avg_total_runtime_ms"] = display["avg_total_runtime_ms"].map(ms)
        display["loftr_trigger_rate"] = display["loftr_trigger_rate"].map(pct)
        st.dataframe(display, use_container_width=True, hide_index=True)

        left, right = st.columns(2)
        with left:
            fig = px.scatter(
                summary,
                x="avg_total_runtime_ms",
                y="success_rate_percent",
                color="strategy",
                size="loftr_trigger_rate_percent",
                labels={
                    "avg_total_runtime_ms": "平均在线耗时(ms)",
                    "success_rate_percent": "成功率(%)",
                    "strategy": "策略",
                    "loftr_trigger_rate_percent": "LoFTR调用率(%)",
                },
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)
        with right:
            subset = scene_summary[scene_summary["strategy"].isin(["SuperGlue only + cache", "SG Top20 -> LoFTR + cache", "SG Top40 -> LoFTR + cache"])]
            fig = px.bar(
                subset,
                x="scene",
                y="success_rate_percent",
                color="strategy",
                barmode="group",
                labels={"scene": "场景", "success_rate_percent": "成功率(%)", "strategy": "策略"},
                height=380,
                range_y=[80, 100],
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<p class="note">推荐主策略是 SG Top20 -> LoFTR + cache：它在成功率、耗时和 LoFTR 调用率之间最均衡。</p>',
            unsafe_allow_html=True,
        )

    with tab_query:
        st.subheader("单张查询回放")
        query_df, attempt_df = read_strategy_outputs(strategy_name)
        case_filter = st.radio(
            "案例类型",
            ["LoFTR 补救成功", "SuperGlue 直接成功", "最终失败", "全部"],
            horizontal=True,
        )
        filtered = query_df
        if case_filter == "LoFTR 补救成功":
            filtered = query_df[(query_df["success"] == 1) & (query_df["final_stage"] == "LoFTR")]
        elif case_filter == "SuperGlue 直接成功":
            filtered = query_df[(query_df["success"] == 1) & (query_df["final_stage"] == "SuperGlue")]
        elif case_filter == "最终失败":
            filtered = query_df[query_df["success"] == 0]
        if filtered.empty:
            st.warning("当前策略下没有这种案例。")
        else:
            query_name = st.selectbox("选择 query", filtered["query_img"].tolist())
            query_row = filtered[filtered["query_img"] == query_name].iloc[0]
            attempts = attempt_df[attempt_df["query_img"] == query_name].copy()
            display_attempt = choose_attempt_for_display(query_row, attempts)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("最终状态", stage_label(str(query_row["final_stage"])))
            c2.metric("最终 rank", "-" if pd.isna(query_row["final_rank"]) else int(float(query_row["final_rank"])))
            c3.metric("总耗时", ms(float(query_row["total_runtime_ms"])))
            c4.metric("LoFTR 是否触发", "是" if int(query_row["loftr_triggered"]) else "否")

            img_left, img_right = st.columns(2)
            display_image0 = resolve_path(str(display_attempt["image0_path"]))
            display_image1 = resolve_path(str(display_attempt["image1_path"]))
            img_left.image(str(display_image0), caption=f"Query: {query_name}", use_column_width=True)
            img_right.image(
                str(display_image1),
                caption=f"Candidate rank {int(display_attempt['rank'])}: {display_image1.name}",
                use_column_width=True,
            )

            table = attempts[
                [
                    "stage",
                    "rank",
                    "num_matches",
                    "num_inliers",
                    "inlier_ratio",
                    "median_reproj_error",
                    "success",
                    "runtime_ms",
                ]
            ].copy()
            table["runtime_ms"] = table["runtime_ms"].map(ms)
            st.dataframe(table, use_container_width=True, hide_index=True)

            if st.button("生成匹配可视化", type="primary"):
                with st.spinner("正在调用真实模型生成匹配图..."):
                    output_path = generate_match_visualization(display_attempt)
                if output_path is not None and output_path.exists():
                    st.image(str(output_path), caption=str(output_path), use_column_width=True)

    with tab_live:
        st.subheader("现场运行单个 query")
        st.markdown(
            '<p class="note">这里会真实加载模型并运行一个 query。首次运行会慢一些，之后模型会被缓存。</p>',
            unsafe_allow_html=True,
        )
        grouped = read_formal_pairs()
        query_options = [name for name, _ in grouped]
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        live_query = c1.selectbox("Query", query_options, index=query_options.index("IMG_0994.JPG") if "IMG_0994.JPG" in query_options else 0)
        live_sg_topk = c2.selectbox("SuperGlue Top-K", [5, 10, 20, 40], index=2)
        live_loftr_topk = c3.selectbox("LoFTR Top-K", [0, 5, 10, 20, 40], index=4)
        live_cache = c4.checkbox("特征缓存", value=True)

        if st.button("运行当前 query", type="primary"):
            with st.spinner("正在运行 SuperGlue -> LoFTR 自适应流程..."):
                live_result, live_attempts = run_live_query(live_query, live_sg_topk, live_loftr_topk, live_cache)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("结果", stage_label(str(live_result["final_stage"])))
            c2.metric("最终 rank", "-" if live_result["final_rank"] is None else live_result["final_rank"])
            c3.metric("总耗时", ms(float(live_result["total_runtime_ms"])))
            c4.metric("LoFTR 触发", "是" if live_result["loftr_triggered"] else "否")
            st.dataframe(live_attempts, use_container_width=True, hide_index=True)

    with tab_engineering:
        st.subheader("离线建库 + 在线查询接口")
        st.markdown(
            '<p class="note">这一页对应实际工程流程：数据库图像离线缓存特征，在线阶段输入 query，输出 JSON 结果给后续定位或建图模块。</p>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("离线模块", "SuperPoint 特征缓存")
        c2.metric("在线模块", "自适应几何验证")
        c3.metric("输出格式", "JSON + CSV")

        st.markdown("**1. 离线建库**")
        st.code(
            "python offline_build_inloc_feature_cache.py --query_name IMG_0994.JPG --sg_topk 20 --output_dir results\\adaptive_feature_cache\\demo_img0994_top20",
            language="powershell",
        )
        st.markdown("**2. 在线查询**")
        st.code(
            "python online_localize_inloc_query.py --query_name IMG_0994.JPG --sg_topk 20 --loftr_topk 40 --feature_cache_dir results\\adaptive_feature_cache\\demo_img0994_top20 --output_dir results\\adaptive_online_outputs\\demo_img0994",
            language="powershell",
        )
        st.markdown("**3. 结果可以被后续模块读取**")
        st.code(
            """{
  "success": 1,
  "final_stage": "LoFTR",
  "final_rank": 1,
  "pose_estimation_ready": true,
  "accepted_candidate": {
    "candidate_image": "...",
    "num_inliers": 24,
    "median_reproj_error": 0.843
  }
}""",
            language="json",
        )

    with tab_files:
        st.subheader("输出文件")
        st.write("Demo 文件：", str(ROOT / "adaptive_inloc_demo_app.py"))
        st.write("离线建库脚本：", str(ROOT / "offline_build_inloc_feature_cache.py"))
        st.write("在线查询脚本：", str(ROOT / "online_localize_inloc_query.py"))
        st.write("策略汇总：", str(SUMMARY_CSV))
        st.write("场景拆分：", str(SCENE_CSV))
        st.write("可视化输出目录：", str(DEMO_OUTPUT_DIR))
        st.code(
            "streamlit run adaptive_inloc_demo_app.py --server.port 8501",
            language="powershell",
        )


if __name__ == "__main__":
    main()
