# Adaptive Indoor Image Matching

面向室内视觉重定位与三维建图前端的自适应稀疏-稠密图像匹配系统。

本项目比较了 ORB、SIFT、SuperPoint+SuperGlue 和 LoFTR，并在 InLoc 室内定位场景中设计了一个自适应匹配系统：简单样本优先使用 SuperGlue 快速几何验证，困难样本再触发 LoFTR 补救，同时引入数据库 SuperPoint 特征缓存，支持离线建库、在线查询和 JSON 结果输出。

完整项目说明见：[docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)。

## Highlights

- 多方法对比：ORB、SIFT、SuperGlue、LoFTR。
- 多数据集实验：HPatches、InLoc。
- 自适应策略：SuperGlue 快速验证，LoFTR 困难样本补救。
- 工程优化：数据库 SuperPoint 特征离线缓存。
- 工程接口：离线建库、在线 query、JSON 输出。
- Streamlit Demo：策略总览、案例回放、现场运行、工程接口展示。

## Main Results

### Recommended Modes on InLoc

| 模式 | 策略 | 成功率 | 平均在线耗时 | 说明 |
|---|---|---:|---:|---|
| 快速模式 | Confidence dispatch + cache | 94.38% | 267.97 ms | 更低延迟，允许少量成功率下降 |
| 均衡模式 | SG Top20 -> LoFTR + cache | 97.19% | 374.57 ms | 推荐主推方案 |
| 高精度模式 | Confidence then fixed strict + cache | 97.47% | 406.04 ms | 优先追求最高成功率 |

### Demo Cases

| 类型 | 示例 |
|---|---|
| SuperGlue 直接成功 | `results/adaptive_demo_outputs/IMG_0964_JPG_rank01_superglue.png` |
| LoFTR 补救成功 | `results/adaptive_demo_outputs/IMG_0994_JPG_rank01_loftr.png` |
| 最终失败案例 | `results/adaptive_demo_outputs/IMG_1053_JPG_rank03_loftr.png` |

## Run

### 1. Build a Demo Feature Cache

```powershell
.\build_demo_feature_cache.ps1
```

### 2. Run Online Query

```powershell
.\run_online_query_demo.ps1
```

The output JSON contains the accepted candidate, final stage, rank, inliers, reprojection error, and runtime.

### 3. Launch Streamlit Demo

```powershell
.\run_adaptive_demo.ps1
```

Then open:

```text
http://127.0.0.1:8501
```

## Repository Structure

```text
datasets/                         # datasets, pair CSVs, ORB/SIFT baseline material
methods/                          # SuperGlue / LoFTR method code and checkpoints
results/                          # experiment results, raw result tables, demo images
docs/PROJECT_GUIDE.md             # merged formal project guide
materials/                        # local-only old course materials and references, ignored by Git

adaptive_cascade_inloc.py          # full InLoc adaptive strategy evaluation
adaptive_feature_cache.py          # persistent SuperPoint feature cache utilities
offline_build_inloc_feature_cache.py
online_localize_inloc_query.py
adaptive_inloc_demo_app.py         # Streamlit demo
build_demo_feature_cache.ps1
run_online_query_demo.ps1
run_adaptive_demo.ps1
```

## Notes

- This project is an image matching and geometric verification frontend, not a complete 3D reconstruction pipeline.
- The system is designed for online single-image relocalization, not strict 30 FPS video-level realtime.
- Large datasets, checkpoints and generated caches are excluded from Git by `.gitignore`.
- `results/adaptive_feature_cache/` and `results/adaptive_online_outputs/` are generated folders. Rebuild them with the provided PowerShell scripts when needed.
