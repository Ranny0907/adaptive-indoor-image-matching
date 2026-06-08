$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

python online_localize_inloc_query.py `
  --query_name IMG_0994.JPG `
  --sg_topk 20 `
  --loftr_topk 40 `
  --feature_cache_dir results\adaptive_feature_cache\demo_img0994_top20 `
  --output_dir results\adaptive_online_outputs\demo_img0994
