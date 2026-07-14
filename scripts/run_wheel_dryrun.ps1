# Wrapper for the scheduled wheel-executor dry-run (Task "RegimeTrader-WheelDryrun").
# Runs scripts/wheel_dryrun.py during market hours and writes the captured output
# to a dated file under logs/ for review. cmd byte-redirects Python's UTF-8 stdout
# straight to the file (piping through PowerShell mojibakes the glyphs).
Set-Location -Path "c:\Users\bollr\regime_trader"
$env:PYTHONIOENCODING = "utf-8"
if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }
$out = "logs\wheel_dryrun_$(Get-Date -Format 'yyyy-MM-dd').txt"
cmd /c "python scripts\wheel_dryrun.py > `"$out`" 2>&1"
