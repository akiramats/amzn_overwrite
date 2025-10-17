from ib_insync import *
from typing import List
from src.utils.helpers import dte_from_ib_expiry, mid_price

def qualify_underlying(ib: IB, ticker="AMZN"):
    stk = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(stk)
    return stk

def get_underlying_spot(ib: IB, stk: Contract) -> float:
    t = ib.reqMktData(stk, "", False, False)
    ib.sleep(1.0)
    px = t.marketPrice() or mid_price(t.bid, t.ask, t.last)
    ib.cancelMktData(stk)
    return float(px)

def get_option_universe(ib: IB, stk: Contract):
    params = ib.reqSecDefOptParams(stk.symbol, "", stk.secType, stk.conId)
    smart = next(p for p in params if p.exchange == "SMART")
    expirations = sorted(list(smart.expirations))
    strikes = sorted([float(s) for s in smart.strikes if s is not None])
    return expirations, strikes

def expirations_in_window(expirations: List[str], min_dte: int, max_dte: int) -> List[str]:
    return [e for e in expirations if min_dte <= dte_from_ib_expiry(e) <= max_dte]

def strikes_near_spot(strikes: List[float], spot: float, count: int = 60) -> List[float]:
    otm = [k for k in strikes if k >= spot * 0.9]
    lower = max(0, len(otm)-count)
    return otm[lower:][:count]

def option_ticker(ib: IB, symbol: str, expiry: str, strike: float, right="C"):
    opt = Option(symbol, expiry, strike, right, "SMART")
    ib.qualifyContracts(opt)
    t = ib.reqMktData(opt, "100,101,104,106", False, False)  # bid/ask/IV/OI (delayed ok)
    ib.sleep(0.7)
    return opt, t

def approx_target_otm_pct_for_delta(target_delta: float) -> float:
    # rough weekly mapping; refine later
    if target_delta <= 0.16: return 0.10
    if target_delta <= 0.19: return 0.08
    if target_delta <= 0.23: return 0.07
    return 0.06

def choose_by_delta_or_otm(ib: IB, stk: Contract, expiry: str, strikes, target_delta: float, spot: float):
    best = None
    hadGreeks = False
    target_otm = approx_target_otm_pct_for_delta(target_delta)

    for k in strikes:
        opt, tk = option_ticker(ib, stk.symbol, expiry, k, "C")
        g = tk.modelGreeks
        mp = mid_price(tk.bid, tk.ask, tk.last)

        if g and g.delta is not None:
            hadGreeks = True
            delta = abs(g.delta or 0.0)
            if delta <= target_delta and mp > 0.01:
                score = mp / max(0.01, delta)
                payload = dict(opt=opt, mid=mp, strike=k, delta=delta, expiry=expiry, hadGreeks=True, score=score)
                best = max([best, payload], key=lambda x: -1 if x is None else x["score"])
        else:
            otm_pct = (k - spot) / spot
            if otm_pct >= target_otm and mp > 0.01:
                score = (1.0 / (1e-6 + abs(otm_pct - target_otm))) * mp
                payload = dict(opt=opt, mid=mp, strike=k, delta=None, expiry=expiry, hadGreeks=False, score=score)
                best = max([best, payload], key=lambda x: -1 if x is None else x["score"])

        ib.cancelMktData(opt)

    return best

def pick_contract(ib: IB, target_delta: float, min_dte: int, max_dte: int):
    stk = qualify_underlying(ib, "AMZN")
    spot = get_underlying_spot(ib, stk)
    exps, strikes = get_option_universe(ib, stk)
    cands = expirations_in_window(exps, min_dte, max_dte)
    if not cands:
        return None
    strikes_band = strikes_near_spot(strikes, spot, count=60)
    for expiry in sorted(cands, key=lambda e: dte_from_ib_expiry(e)):
        sel = choose_by_delta_or_otm(ib, stk, expiry, strikes_band, target_delta, spot)
        if sel:
            return sel
    return None
