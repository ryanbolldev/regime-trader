# Wrapper for the scheduled market-hours parity check.
# Runs scripts/tastytrade_parity_check.py and writes the diagnostic to a dated
# file under logs/ for later review. Invoked by the Windows Task Scheduler job
# "RegimeTrader-ParityCheck".
Set-Location -Path "c:\Users\bollr\regime_trader"
$env:PYTHONIOENCODING = "utf-8"
if (-not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }
$out = "logs\parity_check_$(Get-Date -Format 'yyyy-MM-dd').txt"
# Let cmd byte-redirect Python's UTF-8 stdout straight to the file. Piping through
# PowerShell re-decodes it via the console's OEM codepage and mojibakes the glyphs.
cmd /c "python scripts\tastytrade_parity_check.py MSTR 21 45 > `"$out`" 2>&1"
