# SuperGlue 实验说明（AutoDL）

现在实验代码已经拆成两份：

- `benchmark_hpatches.py`
- `benchmark_inloc.py`

两份脚本互相独立，结果目录也独立。

## 1. 你当前的 AutoDL 路径

### HPatches

你当前提供的路径是：

```text
/root/autodl-tmp/hpatches_data/hpatches-sequences-release
```

这份路径已经写成 `benchmark_hpatches.py` 的默认值。

### InLoc

你当前提供的路径是：

```text
/root/autodl-tmp/Inloc/iphone7
/root/autodl-tmp/Inloc/cutouts_imageonly
```

这两个路径已经分别写成 `benchmark_inloc.py` 的默认值：

- `--query_root=/root/autodl-tmp/Inloc/iphone7`
- `--database_root=/root/autodl-tmp/Inloc/cutouts_imageonly`

## 2. 两份脚本分别做什么

### `benchmark_hpatches.py`

只负责 HPatches。

- 自动扫描 `i_*` 和 `v_*` 序列
- 自动构造 `1->2`、`1->3`、`1->4`、`1->5`、`1->6`
- 默认输出到：

```text
/root/autodl-tmp/outputs/hpatches
```

### `benchmark_inloc.py`

只负责 InLoc。

- 使用你给定的 `pairs_file`
- 从 `iphone7` 读 query 图
- 从 `cutouts_imageonly` 读 database 图
- 默认输出到：

```text
/root/autodl-tmp/outputs/inloc
```

## 3. 实验规则

两份脚本都按你的 `分工.md` 执行：

- 灰度图
- 长边缩放到 `640`
- 原图长边小于 `640` 不放大
- resize 后宽高向下取整到 `8` 的倍数
- `match_threshold = 0.20`
- RANSAC 前最多保留 `1000` 对匹配
- 用匹配置信度从高到低截断
- `cv2.findHomography(..., cv2.RANSAC, 3.0, maxIters=2000, confidence=0.995)`
- warm-up 5 对
- 单对耗时不包含模型加载，也不包含读图和 resize

输出字段统一为：

```csv
pair_id,dataset,method,num_matches,num_inliers,inlier_ratio,median_reproj_error,mean_reproj_error,success,runtime_ms
```

## 4. HPatches 如何使用

进入仓库目录后直接运行：

```bash
python benchmark_hpatches.py --save_viz
```

因为默认路径已经写好了，所以这条命令就会读取：

```text
/root/autodl-tmp/hpatches_data/hpatches-sequences-release
```

结果默认保存到：

```text
/root/autodl-tmp/outputs/hpatches
```

### 常用变体

只跑 viewpoint：

```bash
python benchmark_hpatches.py --hpatches_split v --save_viz
```

只跑 illumination：

```bash
python benchmark_hpatches.py --hpatches_split i --save_viz
```

不保存图，只保留数值结果：

```bash
python benchmark_hpatches.py
```

## 5. InLoc 如何使用

InLoc 还需要你准备一个配对文件，例如：

```text
/root/autodl-tmp/inloc_pairs.csv
```

运行方式：

```bash
python benchmark_inloc.py \
  --pairs_file /root/autodl-tmp/Inloc/cutouts_imageonly/pairs-query-netvlad20.txt \
  --save_viz
```

默认会读取：

- query 根目录：`/root/autodl-tmp/Inloc/iphone7`
- database 根目录：`/root/autodl-tmp/Inloc/cutouts_imageonly`

结果默认保存到：

```text
/root/autodl-tmp/outputs/inloc
```

## 6. InLoc 配对文件格式

支持 `csv` 或 `txt`。

### 方式 A：CSV

```csv
pair_id,image0,image1
pair_0001,IMG_0001.jpg,DUC1/001/cutout_01.jpg
pair_0002,IMG_0002.jpg,DUC1/002/cutout_03.jpg
```

这里：

- `image0` 会优先在 `--query_root` 下查找
- `image1` 会优先在 `--database_root` 下查找

也支持直接写绝对路径。

如果你想写得更完整，也可以：

```csv
pair_id,image0,image1
pair_0001,/root/autodl-tmp/Inloc/iphone7/IMG_0001.jpg,/root/autodl-tmp/Inloc/cutouts_imageonly/DUC1/001/cutout_01.jpg
```

### 方式 B：TXT

```text
pair_0001 IMG_0001.jpg DUC1/001/cutout_01.jpg
pair_0002 IMG_0002.jpg DUC1/002/cutout_03.jpg
```

### 方式 C：你现在这份两列绝对路径 TXT

脚本现在也支持每行只有两列路径，例如：

```text
/root/autodl-tmp/Inloc/iphone7/IMG_0764.JPG /root/autodl-tmp/Inloc/cutouts_imageonly/DUC1/007/DUC_cutout_007_150_-30.jpg
```

这种格式下：

- 不需要你手动写 `pair_id`
- 脚本会自动生成：
  - `pair_000001`
  - `pair_000002`
  - `pair_000003`
  - ...

## 7. 输出结果分别在哪里

### HPatches 输出

```text
/root/autodl-tmp/outputs/hpatches/
  hpatches_superglue_results.csv
  hpatches_superglue_summary.json
  viz/
```

### InLoc 输出

```text
/root/autodl-tmp/outputs/inloc/
  inloc_superglue_results.csv
  inloc_superglue_summary.json
  viz/
```

这样两个实验从代码、输入参数、结果目录上都已经完全分开。

## 8. 建议你的实际执行顺序

### 第一步：先确认依赖

```bash
pip install -r requirements.txt
```

### 第二步：先跑 HPatches

```bash
python benchmark_hpatches.py --save_viz
```

### 第三步：准备 InLoc 配对文件

例如：

```text
/root/autodl-tmp/inloc_pairs.csv
```

### 第四步：再跑 InLoc

```bash
python benchmark_inloc.py \
  --pairs_file /root/autodl-tmp/Inloc/cutouts_imageonly/pairs-query-netvlad20.txt \
  --save_viz
```

## 9. 你现在最该注意的一点

HPatches 可以直接跑，因为它本身就是标准同序列配准评测结构。

InLoc 不能“直接全量自动跑”的原因不是代码问题，而是官方数据本身不是像 HPatches 那样天然给出统一的平面配准对。  
所以 InLoc 这部分你必须先定一份你要测的 `query-cutout` 配对列表，然后脚本才能批量算指标。
