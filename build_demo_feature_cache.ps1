$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

python offline_build_inloc_feature_cache.py `
  --query_name IMG_0994.JPG `
  --sg_topk 20 `
  --output_dir results\adaptive_feature_cache\demo_img0994_top20 `
  --print_every 5
