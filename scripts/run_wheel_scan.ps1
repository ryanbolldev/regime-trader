# Wrapper for the scheduled in-hours wheel scan (Task "RegimeTrader-WheelScan").
# Runs scripts/wheel_scan_once.py during market hours and captures output to a
# dated file. cmd byte-redirects Python's UTF-8 stdout straight to the file.
Set-Location -Path "c:\Users\bollr\regime_trader"
$env:PYTHONIOENCODING = "utf-8"
if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }
$out = "logs\wheel_scan_$(Get-Date -Format 'yyyy-MM-dd').txt"
cmd /c "python scripts\wheel_scan_once.py > `"$out`" 2>&1"
