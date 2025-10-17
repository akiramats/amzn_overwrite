#!/usr/bin/env python3
# amzn_cc_metrics_4x_sync_v3.py (hardened for delayed data & junk quotes)

import os, math, random
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore")
from ib_insync import IB, Stock, Option, util
from zoneinfo import ZoneInfo
from datetime import time as dtime

# >>> ADD: stability caches & helpers
from collections import defaultdict

# Per-expiry calibration cache
# CALIB[expiry] = {'atm_iv': float|None, 'haircut': float|None, 'drift': float|None,
#                  'iv_samples': int, 'drift_date': 'YYYY-MM-DD'}
CALIB = {}

# Remember last chosen hedge strike per expiry
LAST_KH = {}

# Per-(expiry,strike) smoothed effective IV used in distribution
# ROW_IV[(expiry, strike)] = float
ROW_IV = {}

# Stability knobs
ATM_BAND = float(os.getenv('ATM_BAND', '0.02'))           # ±2% moneyness for ATM collection
ATM_MIN_SAMPLES = int(os.getenv('ATM_MIN_SAMPLES', '6'))  # need this many IV samples to update ATM
HAIRCUT_MAX_STEP = float(os.getenv('HAIRCUT_MAX_STEP', '0.05'))  # cap haircut change/run (±5 vol-pts on ratio)
KHYST_THRESH = float(os.getenv('KHYST_THRESH', '1.00'))   # % absolute improvement required to switch K_h
ROW_IV_MAX_STEP = float(os.getenv('ROW_IV_MAX_STEP', '0.03'))    # cap per-run IV_eff change (±3 vol-pts)
IV_EMA_ALPHA = float(os.getenv('IV_EMA_ALPHA', '0.30'))   # EMA weight for new ATM/row IV
DRIFT_RTH_STICKY = True                                   # freeze drift intra-day

def _ema(prev, new, alpha):
    if prev is None or not isinstance(prev, (int, float)):
        return new
    return (1.0 - alpha) * prev + alpha * new

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def _rate_limit(prev, new, max_step):
    if prev is None:
        return new
    delta = new - prev
    if delta > max_step:  return prev + max_step
    if delta < -max_step: return prev - max_step
    return new

SYM = 'AMZN'
EXCHANGE = 'SMART'
CURRENCY = 'USD'
PRIMARY = 'NASDAQ'

STRIKES_AROUND = 15
WAIT_SEC = float(os.getenv('WAIT_SEC', '10.0'))   # longer for delayed
GENERIC_TICKS = '106'     # modelGreeks
MAX_EXPIRIES = 4
DELTA_MAX = 0.80

# --- Tunables ---
FEE_PER_SHARE = float(os.getenv('FEE_PER_SHARE', '0.02'))
RISK_FREE = float(os.getenv('RISK_FREE_PCT', '5.0'))       # annual %
MARGIN_RATE = float(os.getenv('MARGIN_RATE', '0.33'))      # % of notional
DIST_POINTS = int(os.getenv('DIST_POINTS', '121'))         # quantile grid size
PHYS_DRIFT = float(os.getenv('PHYS_DRIFT_PCT', '0.05'))     # annual drift (%)
IV_HAIRCUT = float(os.getenv('IV_HAIRCUT', '0.80'))        # realized ≈ haircut * IV
FORCE_MKT_TYPE = int(os.getenv('IB_MKT_TYPE', '4'))        # 1=live, 4=delayed

ACCOUNT_ID = os.getenv('IB_ACCOUNT', 'U19289409')

# ---------------- helpers ----------------
import random as _rnd

def _mc_weighted_ar_check(spot, k_o, const_cash, dte, drift_pct, sigma_ann, n_paths=5000):
    """
    Quick MC to cross-check mean annualized return from the overwrite payoff model.
    Returns (weightedAR_pct, diff_vs_quadrature_pct_or_None)
    Caller must pass the quadrature WeightedAR% for the diff (or None to skip diff).
    """
    if sigma_ann <= 0 or dte <= 0:
        return None
    T = dte / 365.0
    mu = (drift_pct/100.0 - 0.5*(sigma_ann**2)) * T
    sdt = sigma_ann * (T ** 0.5)

    pnl_sum = 0.0
    for _ in range(n_paths):
        z = _rnd.gauss(0.0, 1.0)
        ST = spot * math.exp(mu + sdt*z)
        pnl = const_cash - max(ST - k_o, 0.0)
        pnl_sum += pnl
    mean_pnl = pnl_sum / n_paths
    return (mean_pnl / spot) * 100.0 * (365.0 / dte)


def safe(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v

def is_bad_quote(x):
    # IB uses -1 for no data; treat <=0 as unusable for option quotes
    return (x is None) or (isinstance(x, float) and (math.isnan(x) or x <= 0.0 or x == -1.0))

def mid_from_ticker(tk):
    """Return robust (mid, bid, ask) with sanitization and crossed-fix."""
    if not tk:
        return None, None, None
    bid = safe(getattr(tk, 'bid', None))
    ask = safe(getattr(tk, 'ask', None))
    if is_bad_quote(bid): bid = None
    if is_bad_quote(ask): ask = None
    # fix crossed/locked markets if both exist
    if bid is not None and ask is not None:
        if ask < bid:
            # swap or widen minimally
            mid = (bid + ask)/2.0
            if mid <= 0:
                ask = bid + 0.01
        mid = (bid + ask)/2.0
    elif bid is not None:
        mid = bid
    elif ask is not None:
        mid = ask
    else:
        mid = None
    return mid, bid, ask

def _snap_bidask(ib: IB, contract, mkt_type: int):
    """
    Snapshot a single contract with the chosen market data type:
      2 = frozen real-time (needs real-time OPRA)
      4 = delayed-frozen (needs delayed OPRA to API)
    Returns (bid, ask) where unusable values are returned as None.
    """
    try:
        ib.reqMarketDataType(mkt_type)
        t = ib.reqMktData(contract, genericTickList='', snapshot=True, regulatorySnapshot=False)
        ib.sleep(2.0)
        b = safe(getattr(t, 'bid', None))
        a = safe(getattr(t, 'ask', None))
        b = None if is_bad_quote(b) else b
        a = None if is_bad_quote(a) else a
        return b, a
    except Exception:
        return None, None

def mid_from_ticker_or_frozen(ib: IB, tk, contract):
    """
    Prefer streaming (what you already request). If bid/ask are unusable,
    fall back to frozen (2) then delayed-frozen (4) snapshots.
    """
    mid, bid, ask = mid_from_ticker(tk)
    if (bid is not None) or (ask is not None):
        return mid, bid, ask

    # Try frozen real-time (2), then delayed-frozen (4)
    for mkt_type in (2, 4):
        b, a = _snap_bidask(ib, contract, mkt_type)
        if (b is not None) or (a is not None):
            if b is not None and a is not None:
                m = (b + a) / 2.0
            else:
                m = b if b is not None else a
            return m, b, a

    # Nothing better found; return the original streaming values
    return mid, bid, ask

def nearest_strikes_from_set(available, spot, n_each_side):
    s = sorted({float(x) for x in available if x and x > 0})
    if not s:
        return []
    idx = min(range(len(s)), key=lambda i: abs(s[i] - spot))
    lo = max(0, idx - n_each_side)
    hi = min(len(s), idx + n_each_side + 1)
    return s[lo:hi]

def _is_us_equity_rth_now() -> bool:
    """
    Returns True during regular trading hours (US/Eastern, Mon–Fri, 09:30–16:00).
    Does not account for NYSE holidays (treated as closed).
    """
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:  # Sat/Sun
        return False
    t = now_et.time()
    return (t >= dtime(9, 30)) and (t <= dtime(16, 0))

def fetch_spot_with_fallback(ib: IB, stk: Stock, poll_live=5.0, poll_delayed=6.0):
    """
    During RTH: return a live-ish spot (marketPrice/last/close) like before.
    Outside RTH: return the last daily close (so it matches option quotes that are from the close).
    """
    rth_now = _is_us_equity_rth_now()

    # Always set the requested market data type first
    ib.reqMarketDataType(FORCE_MKT_TYPE)
    ib.reqMktData(stk, '', False, False)
    ib.sleep(0.5)
    tk = ib.ticker(stk)

    if not rth_now:
        # Prefer last *official* daily close outside RTH
        v_close = safe(getattr(tk, 'close', None)) if tk else None
        if v_close is not None and v_close > 0:
            return float(v_close)

        # Fallback: fetch recent daily bars and use the most recent bar's close
        try:
            bars = ib.reqHistoricalData(
                stk, endDateTime='', durationStr='3 D', barSizeSetting='1 day',
                whatToShow='TRADES', useRTH=True, formatDate=1, keepUpToDate=False
            )
            closes = [b.close for b in bars if safe(getattr(b, 'close', None))]
            if closes:
                return float(closes[-1])
        except Exception:
            pass
        # Absolute last fallback if everything fails: use whatever marketPrice/last we can read
        if tk:
            v_fallback = next((x for x in (tk.marketPrice(), tk.last, tk.close) if safe(x)), None)
            if v_fallback is not None:
                return float(v_fallback)
        return None

    # RTH path: same logic as before (prefer real-time-ish)
    steps = max(1, int((poll_live if FORCE_MKT_TYPE == 1 else poll_delayed) / 0.25))
    for _ in range(steps):
        ib.sleep(0.25)
        tk = ib.ticker(stk)
        if not tk:
            continue
        v = next((x for x in (tk.marketPrice(), tk.last, tk.close) if safe(x)), None)
        if v is not None:
            return float(v)

    # Final fallback: recent daily close (should rarely trigger during RTH)
    try:
        bars = ib.reqHistoricalData(
            stk, endDateTime='', durationStr='2 D', barSizeSetting='1 day',
            whatToShow='TRADES', useRTH=True, formatDate=1, keepUpToDate=False
        )
        if bars and safe(bars[-1].close):
            return float(bars[-1].close)
    except Exception:
        pass
    return None

def pick_nearest_expiries(chains, n=4):
    today = datetime.now(timezone.utc).date()
    rec = next((r for r in chains if getattr(r, 'exchange', None) == 'SMART'), chains[0])
    exps = sorted({datetime.strptime(e, '%Y%m%d').date() for e in rec.expirations})
    exps = [e for e in exps if e >= today] or [exps[-1]]
    return [e.strftime('%Y%m%d') for e in exps[:n]]

def per_expiry_available_strikes_and_trading_class(ib: IB, expiry: str):
    probe = Option(symbol=SYM, lastTradeDateOrContractMonth=expiry,
                   right='C', exchange=EXCHANGE, currency=CURRENCY)
    details = ib.reqContractDetails(probe)
    if not details:
        return set(), None
    strikes = {float(d.contract.strike) for d in details if d.contract.strike}
    tclass = details[0].contract.tradingClass or SYM
    return strikes, tclass


# ===== Realized Vol & ATM IV calibration helpers =====

def realized_vol_annualized(ib: IB, stk: Stock, lookback_days: int) -> float:
    """
    Annualized realized volatility from daily log returns over ~lookback_days.
    Uses regular trading hours daily bars.
    Returns a float (e.g., 0.35 for 35%).
    """
    lookback_days = max(10, min(60, int(lookback_days)))
    try:
        bars = ib.reqHistoricalData(
            stk, endDateTime='', durationStr=f'{lookback_days+5} D',
            barSizeSetting='1 day', whatToShow='TRADES', useRTH=True,
            formatDate=1, keepUpToDate=False
        )
        closes = [b.close for b in bars if safe(getattr(b, 'close', None))]
        if len(closes) < 12:
            return None
        import math as _m
        rets = []
        for i in range(1, len(closes)):
            if closes[i-1] and closes[i-1] > 0 and closes[i] > 0:
                rets.append(_m.log(closes[i]/closes[i-1]))
        if len(rets) < 10:
            return None
        stdev = (sum((x - (sum(rets)/len(rets)))**2 for x in rets) / (len(rets)-1))**0.5
        rv_annual = stdev * (252 ** 0.5)
        return float(rv_annual)
    except Exception:
        return None

def annualized_momentum(ib: IB, stk: Stock, lookback_days: int = 60) -> float:
    """
    Annualized drift proxy from past lookback_days total return.
    Returns percent (e.g., 8.0 for +8% annualized).
    """
    try:
        bars = ib.reqHistoricalData(
            stk, endDateTime='', durationStr=f'{lookback_days+5} D',
            barSizeSetting='1 day', whatToShow='TRADES', useRTH=True,
            formatDate=1, keepUpToDate=False
        )
        closes = [b.close for b in bars if safe(getattr(b, 'close', None))]
        if len(closes) < 5:
            return 0.0
        start, end = closes[0], closes[-1]
        if start <= 0:
            return 0.0
        total_ret = (end / start) - 1.0
        ann = (1.0 + total_ret) ** (252.0 / max(1.0, float(lookback_days))) - 1.0
        return float(ann * 100.0)
    except Exception:
        return 0.0

def _collect_near_atm_ivs(ticks_map, spot: float, band: float = None):
    """
    Collect call IVs near ATM. Escalates search band if not enough samples.
    Returns (median_iv, iv_list_used).
    """
    if not spot or spot <= 0:
        return None, []
    bands = [ATM_BAND or 0.02, 0.05, 0.08]
    for b in bands:
        ivs = []
        for (right, k), tk in ticks_map.items():
            if right != 'C' or not k or k <= 0:
                continue
            if abs(k/spot - 1.0) <= b:
                g = getattr(tk, 'modelGreeks', None)
                iv = safe(getattr(g, 'impliedVol', None))
                if iv and iv > 0:
                    ivs.append(float(iv))
        if len(ivs) >= ATM_MIN_SAMPLES:
            ivs.sort()
            mid = ivs[len(ivs)//2]
            return float(mid), ivs
    # If we reach here, either no data or fewer than ATM_MIN_SAMPLES
    # Return whatever we found at the widest band (could be empty)
    ivs.sort() if 'ivs' in locals() else None
    if ivs:
        mid = ivs[len(ivs)//2]
        return float(mid), ivs
    return None, []

def dynamic_calibration_for_expiry(ib: IB, stk: Stock, expiry_dte: int,
                                   ticks_map, spot: float, risk_free_pct: float,
                                   expiry_key: str = None):
    expiry_key = expiry_key or "UNKNOWN"
    """
    Smoothed per-expiry calibration:
      - ATM_IV: median within ±ATM_BAND, EMA-smoothed, update only if ≥ ATM_MIN_SAMPLES
      - Haircut: rv/atm_iv, clamped [0.60, 1.05], rate-limited by HAIRCUT_MAX_STEP
      - Drift: 0.8*RF + 0.2*60d momentum, clamped [-10, 20], sticky for RTH day
    """
    # Fetch prior state
    st = CALIB.get(expiry_key, {'atm_iv': None, 'haircut': None, 'drift': None,
                                'iv_samples': 0, 'drift_date': None})

    # Realized vol over ~DTE days (clamped to [10, 60])
    window = max(10, min(60, int(expiry_dte)))
    rv = realized_vol_annualized(ib, stk, window)  # e.g., 0.35
    # Near-ATM IV
    atm_now, used = _collect_near_atm_ivs(ticks_map, spot, band=ATM_BAND)

    # Smooth ATM IV if we have enough samples; else keep previous
    if atm_now is not None and len(used) >= ATM_MIN_SAMPLES:
        atm_iv = _ema(st['atm_iv'], atm_now, IV_EMA_ALPHA) if st['atm_iv'] is not None else atm_now
        iv_samples = len(used)
    else:
        atm_iv = st['atm_iv']  # stick
        iv_samples = st['iv_samples']  # carry forward

    # Haircut raw from rv/atm, with guards
    if rv and atm_iv and atm_iv > 1e-6:
        hc_raw = _clamp(rv / atm_iv, 0.60, 1.05)
    else:
        hc_raw = st['haircut'] if st['haircut'] is not None else 0.90

    # Rate-limit haircut change
    haircut = _rate_limit(st['haircut'], hc_raw, HAIRCUT_MAX_STEP)

    # Drift: compute once per session day if sticky; otherwise recompute
    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    drift = st['drift']
    if (not DRIFT_RTH_STICKY) or (st['drift'] is None) or (st['drift_date'] != today_str):
        mom_pct = annualized_momentum(ib, stk, 60)  # %
        drift_new = 0.8 * risk_free_pct + 0.2 * mom_pct
        drift = _clamp(drift_new, -10.0, 20.0)
        drift_date = today_str
    else:
        drift_date = st['drift_date']

    # Save back
    CALIB[expiry_key] = {
        'atm_iv': atm_iv,
        'haircut': haircut,
        'drift': drift,
        'iv_samples': iv_samples,
        'drift_date': drift_date
    }

    diag = {
        'RV_ann%': None if rv is None else round(rv*100.0, 2),
        'ATM_IV%': None if atm_iv is None else round(atm_iv*100.0, 2),
        'Haircut': None if haircut is None else round(haircut, 3),
        'Drift%': None if drift is None else round(drift, 2),
        'IV_samples': int(iv_samples or 0)
    }
    return drift, haircut, diag




# --------- BS delta (for proxy when greeks missing) ----------
def _norm_cdf(x):
    return 0.5*(1.0 + math.erf(x / math.sqrt(2.0)))

def call_delta_bs(S, K, T, sigma, r=0.0, q=0.0):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return None
    d1 = (math.log(S/K) + (r - q + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
    return math.exp(-q*T) * _norm_cdf(d1)

    # ===== ProbITM via N(d2) helpers =====
def _bs_d1_d2(S, K, T, sigma, r=0.0, q=0.0):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return None, None
    volT = sigma * math.sqrt(T)
    m = math.log(S/K) + (r - q + 0.5*sigma*sigma)*T
    d1 = m / volT
    d2 = d1 - volT
    return d1, d2

def _prob_itm_call_nd2(S, K, T, sigma, r=0.0, q=0.0):
    """Risk-neutral probability P(ST > K) = N(d2)."""
    d1, d2 = _bs_d1_d2(S, K, T, sigma, r, q)
    return None if d2 is None else _norm_cdf(d2)

def _prob_itm_from_delta(delta_as_Nd1, T, sigma):
    """
    If we only know call delta ≈ N(d1), invert to d1 then d2=d1-σ√T, then Prob=N(d2).
    """
    if delta_as_Nd1 is None or sigma is None or T <= 0 or sigma <= 0:
        return None
    # reuse your _norm_ppf inversion
    d1 = _norm_ppf(max(1e-8, min(1-1e-8, delta_as_Nd1)))
    d2 = d1 - sigma*math.sqrt(T)
    return _norm_cdf(d2)

# ---------- Distribution helpers ----------
def _norm_ppf(p):
    a = 0.147
    x = 2*p - 1.0
    sgn = 1.0 if x >= 0 else -1.0
    ln = math.log(1.0 - x*x)
    erfinv = sgn * math.sqrt( math.sqrt( (2/(math.pi*a) + ln/2.0)**2 - ln/a ) - (2/(math.pi*a) + ln/2.0) )
    return math.sqrt(2.0) * erfinv

def _terminal_prices_logN(S0, T, drift_annual_pct, sigma_annual, n_points):
    n = max(31, int(n_points))
    if n % 2 == 0:
        n += 1
    qs = [(i+1)/(n+1) for i in range(n)]
    z = [_norm_ppf(q) for q in qs]
    mu = (drift_annual_pct/100.0 - 0.5*sigma_annual*sigma_annual) * T
    sdt = sigma_annual * math.sqrt(T)
    ST = [ S0 * math.exp(mu + sdt*zi) for zi in z ]
    w  = [1.0/len(ST)] * len(ST)
    return ST, w

def _ann_from_per_share(x, S0, dte):
    return (x / S0) * 100.0 * (365.0 / max(dte,1))

    # ===== Empirical horizon returns (optional; uses yfinance if present) =====
def _trading_days_from_dte(dte):
    # ~252 trading days per year
    return max(1, int(round(dte * 252.0 / 365.0)))

def _get_empirical_multipliers(symbol: str, horizon_td: int, start="2010-01-01", end=None, max_rows=100000):
    """
    Returns a FLAT list[float] of forward price multipliers over `horizon_td` trading days:
      M_t = Close(t+horizon) / Close(t)
    If yfinance is missing/unavailable, returns None (script continues normally).
    """
    try:
        import pandas as pd  # local import to avoid hard dependency if not installed
        import yfinance as yf
        df = yf.download(symbol, start=start, end=end, progress=False)[["Close"]].dropna()
        px = df["Close"]                      # Series
        fwd = px.shift(-horizon_td)           # Series
        mult = (fwd / px).dropna()            # Series of floats (but be defensive)

        if len(mult) > max_rows:
            mult = mult.iloc[-max_rows:]

        # DEFENSIVE FLATTEN: always return python floats
        vals = mult.to_numpy()
        # vals could be shape (N, 1) — flatten safely:
        if getattr(vals, "ndim", 1) > 1:
            vals = vals.reshape(-1)
        return [float(x) for x in vals.tolist()]
    except Exception:
        return None

# >>> ADD: sticky hedge chooser
def _choose_sticky_kh(expiry: str, tmp_rows):
    """
    Pick best strike by AnnOnSpot%, but only switch away from prior K_h
    if improvement ≥ KHYST_THRESH (absolute percentage points).
    """
    if not tmp_rows:
        return None, None

    # Find best in this run
    best_idx = max(range(len(tmp_rows)), key=lambda i: tmp_rows[i]['AnnOnSpot%'])
    best_k   = tmp_rows[best_idx]['Strike']
    best_val = tmp_rows[best_idx]['AnnOnSpot%']

    prev_k = LAST_KH.get(expiry)
    if prev_k is None:
        LAST_KH[expiry] = best_k
        return best_k, best_val

    # Find previous strike's AnnOnSpot% in current tmp_rows (if present)
    prev_row = next((r for r in tmp_rows if r['Strike'] == prev_k), None)
    if prev_row is None:
        LAST_KH[expiry] = best_k
        return best_k, best_val

    prev_val = prev_row['AnnOnSpot%']
    improvement = best_val - prev_val

    if improvement >= KHYST_THRESH:
        LAST_KH[expiry] = best_k
        return best_k, best_val
    else:
        return prev_k, prev_val


# ---------- Core per-expiry block ----------
def one_expiry_block(ib: IB, stk_ref: Stock, spot: float, expiry: str):
    avail_strikes, trading_class = per_expiry_available_strikes_and_trading_class(ib, expiry)
    if not avail_strikes:
        return [], None, None, None

    strikes_sel = nearest_strikes_from_set(avail_strikes, spot, STRIKES_AROUND)
    if not strikes_sel:
        return [], None, None, None

    # Build contracts
    contracts = []
    for k in strikes_sel:
        kf = float(k)
        contracts.append(Option(symbol=SYM, lastTradeDateOrContractMonth=expiry,
                                strike=kf, right='C', exchange=EXCHANGE,
                                currency=CURRENCY, tradingClass=trading_class))
        contracts.append(Option(symbol=SYM, lastTradeDateOrContractMonth=expiry,
                                strike=kf, right='P', exchange=EXCHANGE,
                                currency=CURRENCY, tradingClass=trading_class))

    contracts = ib.qualifyContracts(*contracts)
    if not contracts:
        return [], None, None, None

    # Subscribe
    ib.reqMarketDataType(FORCE_MKT_TYPE)
    for c in contracts:
        ib.reqMktData(c, GENERIC_TICKS, False, False)
    ib.sleep(WAIT_SEC)

    ticks = {(c.right, round(float(c.strike), 4)): ib.ticker(c) for c in contracts}
    contract_by_key = {(c.right, round(float(c.strike), 4)): c for c in contracts}
    today = datetime.now(timezone.utc).date()
    dte = (datetime.strptime(expiry, '%Y%m%d').date() - today).days or 1
    T = dte / 365.0

        # >>> ADD THIS CALIBRATION BLOCK (must come before tmp_rows and any dyn_diag use)
    dyn_drift_pct, dyn_iv_haircut, dyn_diag = dynamic_calibration_for_expiry(
        ib=ib, stk=stk_ref, expiry_dte=dte, ticks_map=ticks, spot=spot,
        risk_free_pct=RISK_FREE, expiry_key=expiry
    )
    dyn_diag = dyn_diag or {}   # defensive

    tmp_rows = []
    for k in strikes_sel:
        kf = round(float(k), 4)
        ct = ticks.get(('C', kf))
        pt = ticks.get(('P', kf))
        if not ct or not pt:
            continue

        c_mid, c_bid, c_ask = mid_from_ticker_or_frozen(ib, ct, contract_by_key.get(('C', kf)))
        p_mid, p_bid, p_ask = mid_from_ticker_or_frozen(ib, pt, contract_by_key.get(('P', kf)))
        if c_mid is None or p_mid is None:
            continue  # skip junk rows entirely

        # greeks (may be None after-hours)
        g = getattr(ct, 'modelGreeks', None)
        delta = safe(getattr(g, 'delta', None))
        iv = safe(getattr(g, 'impliedVol', None))

        # Δ proxy if needed
        delta_proxy = None
        if delta is None and iv is not None and T > 0:
            delta_proxy = call_delta_bs(spot, kf, T, iv, r=RISK_FREE/100.0)

        premium_pct = (c_mid / spot) * 100.0
        ann_yield = premium_pct * (365.0 / dte)
        # Risk-neutral ProbITM via N(d2) if IV available; else derive from delta≈N(d1)
        prob_itm_nd2 = None
        if iv is not None and T > 0:
            prob_itm_nd2 = _prob_itm_call_nd2(spot, kf, T, iv, r=RISK_FREE/100.0, q=0.0)
        elif delta is not None and T > 0:
            # try row IV; else dyn ATM IV as sigma for conversion
            sigma_for_delta = iv if iv is not None else ((dyn_diag.get('ATM_IV%') or 0)/100.0 or None)
            if sigma_for_delta:
                prob_itm_nd2 = _prob_itm_from_delta(abs(delta), T, sigma_for_delta)

        prob_itm = (prob_itm_nd2 * 100.0) if prob_itm_nd2 is not None else 0.0
        breakeven = spot - c_mid


        # conversion carry: prefer bid( call ) vs ask( put ); fall back to mids if one side missing
        c_bid_eff = c_bid if c_bid is not None else c_mid
        p_ask_eff = p_ask if p_ask is not None else p_mid
        if c_bid_eff is None or p_ask_eff is None:
            continue  # still not good enough
        lock_pnl_sh = (kf - spot) + (c_bid_eff - p_ask_eff)
        ann_on_spot = _ann_from_per_share(lock_pnl_sh, spot, dte)

        # Δ filter only if we actually have Δ (real or proxy)
        delta_for_filter = delta if delta is not None else delta_proxy
        if (delta_for_filter is not None) and (abs(delta_for_filter) > DELTA_MAX):
            continue

        tmp_rows.append({
            'Expiry': expiry,
            'DTE': dte,
            'Strike': float(kf),
            'Bid': None if c_bid is None else round(c_bid, 3),
            'Ask': None if c_ask is None else round(c_ask, 3),
            'Mid': round(c_mid, 3),
            'Delta': None if delta_for_filter is None else round(delta_for_filter, 3),
            'IV': None if iv is None else round(iv, 4),
            'Premium%': round(premium_pct, 2),
            'AnnYield%': round(ann_yield, 1),
            'AnnOnSpot%': round(ann_on_spot, 2),
            'ProbITM%': round(prob_itm, 1),
            'Breakeven': round(breakeven, 2),
            'LockPnL/sh': round(lock_pnl_sh, 4),
        })

    if not tmp_rows:
        return [], None, None, None

    # --- ATM backfill from rows if calibration didn't find any ---
    if dyn_diag.get('ATM_IV%') is None:
        iv_near = [r['IV'] for r in tmp_rows
                   if r.get('IV') is not None and abs(r['Strike']/spot - 1.0) <= max(ATM_BAND, 0.05)]
        if iv_near:
            iv_near.sort()
            atm_proxy = iv_near[len(iv_near)//2]
            # update calibration cache and diag
            key = expiry  # same key you passed to dynamic_calibration_for_expiry
            st = CALIB.get(key, {'atm_iv': None, 'haircut': None, 'drift': None,
                                 'iv_samples': 0, 'drift_date': None})
            atm_iv_new = _ema(st['atm_iv'], atm_proxy, IV_EMA_ALPHA) if st['atm_iv'] else atm_proxy
            CALIB[key] = {**st, 'atm_iv': atm_iv_new, 'iv_samples': max(st.get('iv_samples', 0), len(iv_near))}
            dyn_diag['ATM_IV%'] = round(atm_iv_new * 100.0, 2)

    # Best hedge by max AnnOnSpot%
    best_idx = max(range(len(tmp_rows)), key=lambda i: tmp_rows[i]['AnnOnSpot%'])

    # Best hedge with hysteresis
    best_kh, best_ann_on_spot = _choose_sticky_kh(expiry, tmp_rows)

    # Hedge quotes (prefer firm bid/ask; fall back to mid)
    kh_key = round(float(best_kh), 4)
    ct_h = ticks.get(('C', kh_key))
    pt_h = ticks.get(('P', kh_key))
    c_mid_h, c_bid_h, _ = mid_from_ticker(ct_h) if ct_h else (None, None, None)
    p_mid_h, _, p_ask_h = mid_from_ticker(pt_h) if pt_h else (None, None, None)
    # Effective legs: prefer firm side, else mid
    c_bid_h_eff = c_bid_h if c_bid_h is not None else c_mid_h
    p_ask_h_eff = p_ask_h if p_ask_h is not None else p_mid_h

    rows = []
    for r in tmp_rows:
        k_o = r['Strike']
        iv_row = r['IV'] if r['IV'] is not None else 0.40

        c_mid_o = r['Mid']
        c_bid_o = r['Bid']  # may be None

        # Use *bid* when we SELL the overwrite call; fall back to mid if bid missing
        premium_sell = c_bid_o if c_bid_o is not None else c_mid_o

        if (c_bid_h_eff is not None) and (p_ask_h_eff is not None):
            # cash from synthetic short (sell call @ bid, buy put @ ask) + overwrite premium
            const_cash = (best_kh - spot) + (c_bid_h_eff - p_ask_h_eff) + premium_sell - FEE_PER_SHARE
        else:
            # if hedge quotes missing, only the overwrite premium contributes
            const_cash = premium_sell - FEE_PER_SHARE

        # Expiry breakeven computed strictly from expiry PnL model:
        # pnl(ST) = const_cash - max(ST - K_o, 0)
        # If const_cash <= 0, there is no ST where pnl >= 0 ⇒ breakeven is undefined
        all_in_breakeven = (k_o + const_cash) if const_cash > 0 else None

        # right above this line you already have: dyn_drift_pct, dyn_iv_haircut, dyn_diag = dynamic_calibration_for_expiry(...)
        atm_iv_frac = ((dyn_diag or {}).get('ATM_IV%') or 0) / 100.0
        iv_base = r['IV'] if r['IV'] is not None else (atm_iv_frac if atm_iv_frac > 0 else 0.40)
        iv_row_eff = max(1e-6, iv_base * dyn_iv_haircut)

        # Clamp realized vol into a sane band around your per-expiry RV (prevents jumpy WeightedAR%)
        rv_ann = (dyn_diag.get('RV_ann%') or 0.0) / 100.0
        if rv_ann > 0:
            floor = 0.80 * rv_ann       
            iv_row_eff = max(iv_row_eff, floor)

        drift_for_dist = RISK_FREE 
        ST, W = _terminal_prices_logN(spot, T, drift_for_dist, iv_row_eff, DIST_POINTS)


        # ----- Optional empirical sanity checks -----
        emp_prob = None
        emp_cvar5_ann = None
        h_td = _trading_days_from_dte(r['DTE'])
        emp_mults = _get_empirical_multipliers(SYM, h_td, start="2010-01-01", end=None)
        if emp_mults and len(emp_mults) >= 200:
            thresh = k_o / spot
            emp_prob = sum(1 for m in emp_mults if float(m) >= thresh) / float(len(emp_mults)) * 100.0
            emp_pnls = [const_cash - max(spot*float(m) - k_o, 0.0) for m in emp_mults]
            emp_pnls.sort()
            tail_n = max(1, int(round(0.05 * len(emp_pnls))))
            tail_avg = sum(emp_pnls[:tail_n]) / tail_n
            emp_cvar5_ann = _ann_from_per_share(tail_avg, spot, r['DTE'])

        # Overwrite path
        ow_path = [-max(sT - k_o, 0.0) for sT in ST]
        pnl = [const_cash + x for x in ow_path]

        mean_pnl = sum(w * x for w, x in zip(W, pnl))
        var_pnl = sum(w * (x - mean_pnl)**2 for w, x in zip(W, pnl))
        std_pnl = math.sqrt(max(var_pnl, 0.0))

        weighted_ar = _ann_from_per_share(mean_pnl, spot, r['DTE'])
        std_ann = _ann_from_per_share(std_pnl, spot, r['DTE'])
        sharpe_w = (weighted_ar - RISK_FREE) / std_ann if std_ann > 1e-12 else None

        # Build row FIRST so we can safely write to r2[…]
        r2 = dict(r)
        r2['All-in AR%']     = round((r['AnnYield%'] + best_ann_on_spot), 2)
        r2['AllInBreakeven'] = None if const_cash <= 0 else round(k_o + const_cash, 2)
        r2['WeightedAR%']    = round(weighted_ar, 2)
        r2['SharpeW']        = None if sharpe_w is None else round(sharpe_w, 2)

        # Sortino
        downside = [((x - mean_pnl) if x < mean_pnl else 0.0) for x in pnl]
        var_down = sum(w * d**2 for w, d in zip(W, downside))
        std_down = math.sqrt(max(var_down, 0.0))
        sortino = (weighted_ar - RISK_FREE) / _ann_from_per_share(std_down, spot, r['DTE']) if std_down > 1e-12 else None
        r2['Sortino'] = None if sortino is None else round(sortino, 2)

        # CVaR(5%)
        pairs = sorted(zip(pnl, W), key=lambda t: t[0])
        accum = 0.0; cutoff = 0.05; tail_sum = 0.0; tail_w = 0.0
        for x, w in pairs:
            if accum + w <= cutoff:
                tail_sum += x * w; tail_w += w; accum += w
            else:
                remain = max(0.0, cutoff - accum)
                if remain > 0: tail_sum += x * remain; tail_w += remain
                break
        cvar5_ann = _ann_from_per_share(tail_sum / max(tail_w, 1e-12), spot, r['DTE'])
        r2['CVaR(5%)'] = round(cvar5_ann, 2)

        # Stress ±15%
        r2['StressAR-15%'] = round(_ann_from_per_share(const_cash - max(spot*0.85 - k_o, 0.0), spot, r['DTE']), 2)
        r2['StressAR+15%'] = round(_ann_from_per_share(const_cash - max(spot*1.15 - k_o, 0.0), spot, r['DTE']), 2)

        # ROE and per-trade exp. PnL
        all_in_point = r['AnnYield%'] + best_ann_on_spot
        r2['ROE%'] = None if MARGIN_RATE <= 0 else round(all_in_point / MARGIN_RATE, 1)
        r2['ExpPnL%_perTrade'] = round((mean_pnl / spot) * 100.0, 2)

        # Single MC check (risk-neutral drift)
        try:
            mc_ar = _mc_weighted_ar_check(spot, k_o, const_cash, r['DTE'], drift_for_dist, iv_row_eff, n_paths=5000)
            if mc_ar is not None:
                r2['WeightedAR_MC%'] = round(mc_ar, 2)
                r2['AR_MC_diff%']    = round(mc_ar - weighted_ar, 2)
        except Exception:
            pass

        # Attach empirical stats if available (emp_prob / emp_cvar5_ann computed above)
        if emp_prob is not None:     r2['ProbITM_emp%'] = round(emp_prob, 1)
        if emp_cvar5_ann is not None: r2['CVaR_emp5%']  = round(emp_cvar5_ann, 2)

        # ProbITM% blending (uses r['IV'] or ATM_IV if needed)
        prob_nd2_frac_local = None
        if r.get('IV') is not None and T > 0:
            prob_nd2_frac_local = _prob_itm_call_nd2(spot, k_o, T, r['IV'], r=RISK_FREE/100.0, q=0.0)
        elif r.get('Delta') is not None and T > 0:
            sigma_for_delta = r.get('IV') or (((dyn_diag or {}).get('ATM_IV%') or 0) / 100.0)
            if sigma_for_delta:
                prob_nd2_frac_local = _prob_itm_from_delta(abs(r['Delta']), T, sigma_for_delta)

        if (prob_nd2_frac_local is not None) and (emp_prob is not None):
            w_emp = _clamp(0.7 - 0.01*r['DTE'], 0.3, 0.7)
            r2['ProbITM%'] = round(w_emp*emp_prob + (1.0 - w_emp)*(prob_nd2_frac_local*100.0), 1)
        else:
            r2['ProbITM%'] = round(r.get('ProbITM%', 0.0), 1)

        rows.append(r2)

    return rows, best_kh, best_ann_on_spot, dyn_diag

def main():
    host = os.getenv("IB_HOST", "127.0.0.1")
    port = int(os.getenv("IB_PORT", "4001"))
    cid_env = os.getenv("IB_CLIENT_ID", "")
    client_id = int(cid_env) if cid_env.strip() else random.randint(2000, 9999)

    ib = IB()
    try:
        print(f"Connecting to IB @ {host}:{port} clientId={client_id} …")
        ib.connect(host, port, clientId=client_id, timeout=10)

        # quiet some account noise (pending accounts)
        def _quiet_errors(reqId, code, msg, contract):
            if code in (10275,):
                return
            print(f"Error {code}, reqId {reqId}: {msg}")
        ib.errorEvent += _quiet_errors

        # Lock to intended account
        accounts = ib.managedAccounts()
        if ACCOUNT_ID not in accounts:
            raise RuntimeError(f"Connected, but account {ACCOUNT_ID} not in managedAccounts: {accounts}")
        print(f"Using account: {ACCOUNT_ID}")

        # Underlying
        stk = Stock(SYM, EXCHANGE, CURRENCY, primaryExchange=PRIMARY)
        stk = ib.qualifyContracts(stk)[0]

        # Spot
        spot = fetch_spot_with_fallback(ib, stk)
        if spot is None:
            raise RuntimeError("Failed to fetch AMZN spot. Check market data entitlements.")

        # Expiries
        chains = ib.reqSecDefOptParams(SYM, '', 'STK', stk.conId)
        if not chains:
            raise RuntimeError("reqSecDefOptParams returned no data.")
        expiries = pick_nearest_expiries(chains, n=MAX_EXPIRIES)

        print(f"\n📈 AMZN spot: {spot:.2f} USD")
        for expiry in expiries:
            rows, best_kh, best_ann_on_spot, dyn_diag = one_expiry_block(ib, stk, spot, expiry)
            print(f"\n=== Expiry {expiry} | Δ≤{DELTA_MAX:.2f} | ~{STRIKES_AROUND*2+1} strikes around spot ===")
            if rows:
                print(f"Best hedge Kₕ = {best_kh} | AnnOnSpot% = {best_ann_on_spot:.2f}%")
                print(f"All-in AR% = AnnYield%(Kₒ) + {best_ann_on_spot:.2f}%\n")

                # small diagnostics (will be identical for all rows in this expiry)
                print("Calib:",
                      f"RV≈{dyn_diag['RV_ann%']}%",
                      f"ATM_IV≈{dyn_diag['ATM_IV%']}%",
                      f"haircut={dyn_diag['Haircut']}",
                      f"drift={dyn_diag['Drift%']}%",
                      f"IV_samples={dyn_diag['IV_samples']}\n")

                df = util.df(rows).sort_values(
                    ['WeightedAR%','All-in AR%','AnnYield%','Strike'],
                    ascending=[False, False, False, True]
                )
                cols = [
                    'Strike','Bid','Ask','Mid','Delta','IV',
                    'Premium%','AnnYield%','AnnOnSpot%','All-in AR%',
                    'WeightedAR%','WeightedAR_MC%','AR_MC_diff%','ExpPnL%_perTrade',
                    'SharpeW','Sortino','CVaR(5%)',
                    'StressAR-15%','StressAR+15%',
                    'AllInBreakeven','ProbITM%','ROE%',
                    'ProbITM_emp%','CVaR_emp5%',
                    'LockPnL/sh','DTE','Expiry'
                ]
                df = df[[c for c in cols if c in df.columns]]
                print(df.to_string(index=False))
            else:
                print("No usable option quotes (likely delayed data not served yet). "
                      "Keep FORCE_MKT_TYPE=4, WAIT_SEC≈10–15, and confirm OPRA shows Real-Time or Delayed in Account Window.")
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

if __name__ == "__main__":
    main()