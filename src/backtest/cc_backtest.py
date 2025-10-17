import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple
from statistics import pstdev
from ib_insync import IB, Stock

from src.utils.ib_client import connect_ib

@dataclass
class Trade:
    date: dt.date
    spot: float
    strike: float
    premium: float
    expired_itm: bool
    pnl: float
    dte: int

def hist_closes(ib: IB, days=365*2) -> List[Tuple[dt.date,float]]:
    stk = Stock("AMZN", "SMART", "USD")
    ib.qualifyContracts(stk)
    bars = ib.reqHistoricalData(
        stk, endDateTime="", durationStr=f"{days} D",
        barSizeSetting="1 day", whatToShow="TRADES",
        useRTH=True, formatDate=1
    )
    return [(b.date.date(), b.close) for b in bars]

def backtest_weekly_covered_call(days=365*2, otm_pct=0.08):
    ib = connect_ib()
    closes = hist_closes(ib, days)
    trades: List[Trade] = []
    for i, (d, px) in enumerate(closes):
        if d.weekday() < 4:  # only act on Fridays
            continue
        strike = round(px * (1 + otm_pct), 2)
        # estimate premium from last 21 daily returns
        N = min(22, i)
        rets = [(closes[j][1]/closes[j-1][1]-1) for j in range(i-N+1, i) if j > 0]
        sigma_day = pstdev(rets) if len(rets) >= 10 else 0.02
        sigma_week = sigma_day * (5**0.5)
        prem = max(0.01, round(px * sigma_week * 0.40, 2))
        if i+5 >= len(closes): break
        exp_date, exp_px = closes[i+5]
        itm = (exp_px > strike)
        pnl = prem if not itm else (prem - max(0.0, exp_px - strike))
        trades.append(Trade(d, px, strike, prem, itm, pnl, 5))
    total = sum(t.pnl for t in trades)
    print(f"Trades={len(trades)} Total PnL={total:.2f} Avg/Week={total/max(1,len(trades)):.2f}")
    return trades, total

if __name__ == "__main__":
    backtest_weekly_covered_call()
