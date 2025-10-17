#analyze_chain.py
# --- make sure a loop exists (Streamlit worker thread has none by default) ---
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import os
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

from ib_insync import IB, util, Stock, Option, Contract, Ticker
from datetime import datetime, date

def _connect_with_retry():
    ib = IB()
    base = int(os.getenv("IB_CLIENT_ID", "101"))
    for i in range(5):
        try:
            # short timeout so we fail fast if taken
            ib.connect(HOST, PORT, clientId=base + i, readonly=True, timeout=3)
            return ib
        except Exception as e:
            # 326 -> client id in use. Try next id.
            if "client id is already in use" in str(e).lower():
                continue
            raise
    raise RuntimeError("Could not obtain a free IB clientId after retries.")

def _parse_ib_expiry(raw) -> date:
    s = str(raw).strip()
    if ' ' in s:  # drop time if present
        s = s.split(' ')[0]
    if '-' in s:  # 'YYYY-MM-DD'
        return datetime.strptime(s, '%Y-%m-%d').date()
    if len(s) == 8 and s.isdigit():  # 'YYYYMMDD'
        return datetime.strptime(s, '%Y%m%d').date()
    raise ValueError(f'Unrecognized expiry format: {raw!r}')

util.patchAsyncio()  # safe to call more than once, makes ib_insync & notebooks/Streamlit play nice

HOST = os.getenv("IB_HOST", "127.0.0.1")
PORT = int(os.getenv("IB_PORT", "4002"))
CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "101"))
DRY_RUN = os.getenv("DRY_RUN", "1") in ("1", "true", "True", "YES", "yes")

@dataclass
class Row:
    expiry: str
    dte: int
    strike: float
    bid: float
    ask: float
    mid: float
    delta: Optional[float]
    iv: Optional[float]
    premium_perc: Optional[float]
    ann_yield_simple: Optional[float]
    prob_called: Optional[float]
    breakeven: Optional[float]

# --- helpers ------------------------------------------------------------------

def _mid(bid: float, ask: float) -> float:
    if not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0 or ask <= 0:
        return float("nan")
    return round((bid + ask) / 2, 2)

# Convert IB’s “NNN D NNN H” to integer DTE when needed; we’ll mostly compute ourselves
def _dte_from_expiry(today, expiry_str) -> int:
    return (expiry_str - today).days  # when expiry is datetime.date

# --- main async routine -------------------------------------------------------

def _parse_ib_expiry(raw) -> date:
    s = str(raw).strip()
    if ' ' in s:  # drop time if present
        s = s.split(' ')[0]
    if '-' in s:  # 'YYYY-MM-DD'
        return datetime.strptime(s, '%Y-%m-%d').date()
    if len(s) == 8 and s.isdigit():  # 'YYYYMMDD'
        return datetime.strptime(s, '%Y%m%d').date()
    raise ValueError(f'Unrecognized expiry format: {raw!r}')

util.patchAsyncio()  # safe to call more than once, makes ib_insync & notebooks/Streamlit play nice

HOST = os.getenv("IB_HOST", "127.0.0.1")
PORT = int(os.getenv("IB_PORT", "4002"))
CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "101"))
DRY_RUN = os.getenv("DRY_RUN", "1") in ("1", "true", "True", "YES", "yes")

@dataclass
class Row:
    expiry: str
    dte: int
    strike: float
    bid: float
    ask: float
    mid: float
    delta: Optional[float]
    iv: Optional[float]
    premium_perc: Optional[float]
    ann_yield_simple: Optional[float]
    prob_called: Optional[float]
    breakeven: Optional[float]

# --- helpers ------------------------------------------------------------------

def _mid(bid: float, ask: float) -> float:
    if not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0 or ask <= 0:
        return float("nan")
    return round((bid + ask) / 2, 2)

# Convert IB’s “NNN D NNN H” to integer DTE when needed; we’ll mostly compute ourselves
async def _get_delayed_spot(ib: IB, amzn: Contract):
    ib.reqMarketDataType(3)  # delayed streaming
    t = ib.reqMktData(amzn, '', snapshot=False)
    for _ in range(12):      # ~6s
        await asyncio.sleep(0.5)
        for px in (t.marketPrice(), t.last, t.close):
            if px is not None and math.isfinite(px):
                return float(px)
    # fallback: delayed historical
    bars = await ib.reqHistoricalDataAsync(
        amzn, endDateTime='', durationStr='1 D', barSizeSetting='1 min',
        whatToShow='TRADES', useRTH=False, formatDate=1
    )
    return float(bars[-1].close) if bars else None

# --- main async routine -------------------------------------------------------

async def _analyze_amzn_async(dte_range: Tuple[int, int], max_rows: int = 60) -> List[Row]:
    min_dte, max_dte = dte_range
    ib = _connect_with_retry()

    try:
        # Use delayed quotes if you don't have live subs
        ib.reqMarketDataType(4)  # 1=Live, 2=Frozen, 3=Delayed, 4=Delayed-Frozen

        # Use NASDAQ for the stock (delayed fills faster); keep options on SMART
        amzn = Stock('AMZN', 'NASDAQ', 'USD')
        [amzn] = await ib.qualifyContractsAsync(amzn)

        last = await _get_delayed_spot(ib, amzn)
        if last is None:
            return []

        # ---- expiries: nearest 1 within DTE window ----
        chain = await ib.reqSecDefOptParamsAsync(amzn.symbol, '', amzn.secType, amzn.conId)
        if not chain:
            return []
        cmeta = next((p for p in chain if p.exchange == 'SMART'), chain[0])

        today = date.today()
        exp_filtered = []
        for e in cmeta.expirations:
            try:
                ed = _parse_ib_expiry(e)
            except Exception:
                continue
            dte = (ed - today).days
            if min_dte <= dte <= max_dte:
                exp_filtered.append((e, ed, dte))
        exp_filtered.sort(key=lambda x: x[1])
        exp_filtered = exp_filtered[:1]   # <— **only the nearest one**
        if not exp_filtered:
            return []

        # ---- strikes: ±3 around ATM ----
        strikes_sorted = sorted(s for s in cmeta.strikes if math.isfinite(s))
        atm_idx = min(range(len(strikes_sorted)), key=lambda i: abs(strikes_sorted[i] - last))
        lo, hi = max(0, atm_idx-3), min(len(strikes_sorted), atm_idx+4)
        strikes_focus = strikes_sorted[lo:hi]
        if not strikes_focus:
            return []
        strikes_sorted = sorted(s for s in cmeta.strikes if math.isfinite(s))
        # find ATM index
        atm_idx = min(range(len(strikes_sorted)), key=lambda i: abs(strikes_sorted[i] - last))
        # slice ±5 around ATM (bounds-safe)
        lo = max(0, atm_idx - 5)
        hi = min(len(strikes_sorted), atm_idx + 6)
        strikes_focus = strikes_sorted[lo:hi]

        # ---- Build & qualify contracts (calls only) ----
        to_qualify = []
        for e, _, _ in exp_filtered:
            for k in strikes_focus:
                to_qualify.append(Option('AMZN', e, k, 'C',
                                         exchange='CBOE', currency='USD',
                                         tradingClass='AMZN', multiplier='100'))
        qualified = await ib.qualifyContractsAsync(*to_qualify)
        if not qualified:
            return []

        ib.reqMarketDataType(3)  # delayed streaming
        for con in qualified:
            ib.reqMktData(con, '', snapshot=False)

        # wait up to ~6–8s to fill some ticks
        for _ in range(16):
            await asyncio.sleep(0.5)

        tickers = [ib.ticker(con) for con in qualified if ib.ticker(con)]



        rows: list[Row] = []
        for t in tickers:
            con = t.contract
            if con.right != 'C':
                continue

            # prefer bid/ask mid, fallback to last if needed
            bid = t.bid if (t.bid is not None and math.isfinite(t.bid) and t.bid > 0) else None
            ask = t.ask if (t.ask is not None and math.isfinite(t.ask) and t.ask > 0) else None

            mid = None
            if bid is not None and ask is not None:
                mid = round((bid + ask) / 2, 2)
            else:
                lp = t.last if (t.last is not None and math.isfinite(t.last) and t.last > 0) else None
                if lp is not None:
                    mid = float(lp)

            g = getattr(t, 'modelGreeks', None)
            delta = float(getattr(g, 'delta', None)) if g and getattr(g, 'delta', None) is not None else None
            iv = float(getattr(g, 'impliedVol', None)) if g and getattr(g, 'impliedVol', None) is not None else None

            # expiry → date (handles both 20251018 and 2025-10-18)
            exp_raw = con.lastTradeDateOrContractMonth
            exp_dt = _parse_ib_expiry(exp_raw)
            dte = (exp_dt - today).days

            premium_perc = ann = prob_called = breakeven = None
            if mid is not None and mid > 0 and last and dte >= 0:
                premium_perc = round(100.0 * mid / last, 2)
                if dte > 0:
                    ann = (premium_perc / 100.0) * (365.0 / max(dte, 1))
                breakeven = round(last - mid, 2)
                if delta is not None:
                    prob_called = max(0.0, min(1.0, delta))

            rows.append(Row(
                expiry=exp_dt.isoformat(),
                dte=dte,
                strike=float(con.strike),
                bid=bid if bid is not None else float('nan'),
                ask=ask if ask is not None else float('nan'),
                mid=mid if mid is not None else float('nan'),
                delta=delta,
                iv=iv,
                premium_perc=premium_perc,
                ann_yield_simple=ann,
                prob_called=prob_called,
                breakeven=breakeven
            ))

        rows = [r for r in rows if r.ann_yield_simple is not None]
        rows.sort(key=lambda r: (-r.ann_yield_simple, r.dte))
        return rows[:max_rows]

    finally:
        if ib.isConnected():
            ib.disconnect()

def analyze_amzn(dte_range: Tuple[int, int], max_rows: int = 60) -> List[Row]:
    # Run the coroutine on the patched asyncio loop
    return util.run(_analyze_amzn_async(dte_range, max_rows=max_rows))