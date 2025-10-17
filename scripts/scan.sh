#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python - <<'PY'
from src.analysis.analyze_chain import analyze_amzn
rows = analyze_amzn((7,21), max_rows=40)
print(f"{'Expiry':<10} {'DTE':>3} {'Strike':>8} {'Mid':>6} {'Delta':>6} {'IV':>7} {'Prem%':>7} {'Ann%':>7} {'P(call)':>8}")
for r in rows:
    ivs = "" if r.iv is None else f"{r.iv*100:0.1f}%"
    delt = "" if r.delta is None else f"{r.delta:0.2f}"
    pcal = "" if r.prob_called is None else f"{r.prob_called*100:0.0f}%"
    print(f"{r.expiry:<10} {r.dte:>3} {r.strike:>8.2f} {r.mid:>6.2f} {delt:>6} {ivs:>7} {r.premium_perc*100:>6.2f}% {r.ann_yield_simple*100:>6.1f}% {pcal:>8}")
PY
