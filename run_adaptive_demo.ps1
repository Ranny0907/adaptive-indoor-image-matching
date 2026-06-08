$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

python -m streamlit run adaptive_inloc_demo_app.py `
  --server.port 8501 `
  --server.headless true `
  --browser.gatherUsageStats false `
  --server.fileWatcherType none
