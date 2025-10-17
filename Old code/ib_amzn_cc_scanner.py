#!/usr/bin/env python3
import os
import math
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from ib_insync import IB, Stock, Option, util, Ticker


# ---------------------------- Config ----------------------------
SYMBOL = "AMZN"
CURRENCY = "USD"
EXCHANGE = "SMART"
PRIMARY = "NASDAQ"

# Ladder settings (can override via env)
DEFAULT_N_STRIKES = int(os.getenv("N_STRIKES", "10"))        # number of strikes in ladder (OTM-up)
DEFAULT_STRIKE_STEP = float(os.getenv("STRIKE_STEP", "5"))    # strike increment
DELTA_FILTER_MAX = float(os.getenv("DELTA_MAX", "0.25"))      # shortlist threshold

# Market data type: 1=real-time, 2=frozen, 3=delayed-frozen, 4=frozen (last)
MARKET_DATA_TYPE = int(os.getenv("MD_TYPE", "3"))

# Explicit expiry override (YYYYMMDD) or auto-pick next Friday >= 3 DTE
EXPIRY_OVERRIDE = os.getenv("EXPIRY", "").strip()


# ----------------------- Date / Expiry helpers -----------------------
def next_friday_at_least(days_ahead: int = 3) -> str:
    """Return YYYYMMDD for next Friday that is at least N calendar days ahead."""
    d = datetime.now().date() + timedelta(days=days_ahead)
    # weekday(): Mon=0..Sun=6 -> Friday=4
    days_to_fri = (4 - d.weekday()) % 7
    target = d + timedelta(days=days_to_fri)
    return target.strftime("%Y%m%d")


# ----------------------- Option math helpers -----------------------
def mid_price(t: Optional[Ticker]) -> Optional[float]:
    if not t:
        return None
    b, a = t.bid, t.ask
    if b is not None and a is not None and b > 0 and a > 0:
        return 0.5 * (b + a)
    return None


def marketish_under_px(ut: Optional[Ticker]) -> Optional[float]:
    if not ut:
        return None
    # Try marketPrice (ib_insync computed), then last, then close
    for f in (ut.marketPrice, ):
        try:
            v = f()
            if v and math.isfinite(v):
                return v
        except Exception:
            pass
    for v in (ut.last, ut.close, ut.bid, ut.ask):
        if v is not None and v > 0:
            return v
    return None


# ----------------------- Main async workflow -----------------------
async def main():
    host = os.getenv("IB_HOST", "127.0.0.1")
    port = int(os.getenv("IB_PORT", "4001"))
    client_id = int(os.getenv("IB_CLIENT_ID", "1101"))

    ib = IB()
    try:
        print(f"Connecting to IB @ {host}:{port} clientId={client_id} ...")
        await ib.connectAsync(host, port, clientId=client_id, readonly=False, timeout=10)

        # Use delayed-frozen if off-hours/entitlements absent
        ib.reqMarketDataType(MARKET_DATA_TYPE)

        # Qualify underlying and subscribe
        [stock] = await ib.qualifyContractsAsync(
            Stock(SYMBOL, EXCHANGE, CURRENCY, primaryExchange=PRIMARY)
        )
        ib.reqMktData(stock, "", snapshot=False)

        # Small pause to populate underlying price
        await asyncio.sleep(1.0)
        ut = ib.ticker(stock)
        under_px = marketish_under_px(ut)
        print(f"{SYMBOL} underlying ~ {under_px if under_px else 'n/a'} {CURRENCY}")

        # Determine expiry
        expiry = EXPIRY_OVERRIDE if EXPIRY_OVERRIDE else next_friday_at_least(3)

        # If user gave a strike center via env, use that; else anchor slightly OTM from underlying
        strike_center_env = os.getenv("STRIKE_CENTER", "").strip()
        if strike_center_env:
            try:
                center = float(strike_center_env)
            except ValueError:
                center = None
        else:
            center = None

        # If no underlying price yet, fall back to a reasonable anchor near 215
        if center is None:
            center = (under_px or 215.0)
            # bias towards slightly OTM call ladder (add one step)
            center = math.ceil(center / DEFAULT_STRIKE_STEP) * DEFAULT_STRIKE_STEP + DEFAULT_STRIKE_STEP

        n = max(1, DEFAULT_N_STRIKES)
        step = max(0.5, DEFAULT_STRIKE_STEP)

        strikes = [round(center + i * step, 2) for i in range(n)]

        # Build & qualify option contracts
        opts = [Option(SYMBOL, expiry, k, "C", EXCHANGE, CURRENCY) for k in strikes]
        qualified = await ib.qualifyContractsAsync(*opts)

        # Subscribe to option market data (request 106 model ticks; may not arrive off-hours)
        for c in qualified:
            ib.reqMktData(c, "106", snapshot=False)
        await asyncio.sleep(1.5)  # let NBBO come in

        rows = []
        for c in qualified:
            t = ib.ticker(c)
            b, a = (t.bid, t.ask) if t else (None, None)
            m = mid_price(t)
            g = getattr(t, "modelGreeks", None)
            model_px = getattr(t, "modelPrice", None)
            delta = getattr(g, "delta", None) if g else None
            iv = getattr(g, "impliedVol", None) if g else None

            # Fallback: calculate IV from mid, then greeks from IV (works even off-hours)
            if (iv is None or delta is None) and m and (under_px or marketish_under_px(ib.ticker(stock))):
                uref = under_px or marketish_under_px(ib.ticker(stock))
                try:
                    iv_res = await ib.calculateImpliedVolatility(c, price=m, underPrice=uref)
                    if iv_res and iv_res[0] and iv_res[0].impliedVol and iv_res[0].impliedVol > 0:
                        iv = iv_res[0].impliedVol
                        g_res = await ib.calculateOptionPrice(c, volatility=iv, underPrice=uref)
                        if g_res and g_res[0]:
                            delta = g_res[0].delta
                            gamma = g_res[0].gamma
                            theta = g_res[0].theta
                            vega = g_res[0].vega
                            model_px = g_res[0].optPrice
                        else:
                            gamma = theta = vega = None
                    else:
                        gamma = theta = vega = None
                except Exception:
                    gamma = theta = vega = None
            else:
                gamma = getattr(g, "gamma", None) if g else None
                theta = getattr(g, "theta", None) if g else None
                vega = getattr(g, "vega", None) if g else None

            rows.append({
                "expiry": c.lastTradeDateOrContractMonth,
                "strike": c.strike,
                "bid": b,
                "ask": a,
                "mid": m,
                "modelPx": model_px,
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "vega": vega,
                "iv": iv
            })

        # Sort by strike and print
        rows.sort(key=lambda r: r["strike"])

        def fmt(x, p=4):
            if x is None or (isinstance(x, float) and not math.isfinite(x)):
                return "-"
            if isinstance(x, float):
                if abs(x) >= 100:
                    return f"{x:.2f}"
                return f"{x:.{p}f}"
            return str(x)

        print(f"\nAMZN Calls @ {expiry}  (dataType={MARKET_DATA_TYPE}; Δ filter < {DELTA_FILTER_MAX})")
        print("strike   bid     ask     mid     modelPx delta    gamma     theta     vega     IV")
        print("------  ------  ------  ------  -------  -------  --------  --------  --------  --------")
        for r in rows:
            print(f"{fmt(r['strike'],2):>6}  {fmt(r['bid'],2):>6}  {fmt(r['ask'],2):>6}  {fmt(r['mid'],2):>6}  "
                  f"{fmt(r['modelPx'],4):>7}  {fmt(r['delta'],4):>7}  {fmt(r['gamma'],6):>8}  "
                  f"{fmt(r['theta'],4):>8}  {fmt(r['vega'],4):>8}  {fmt(r['iv'],4):>8}")

        # Shortlist by delta
        shortlist = [r for r in rows if r["delta"] is not None and abs(r["delta"]) <= DELTA_FILTER_MAX]
        print("\nShortlist (Δ ≤ %.2f):" % DELTA_FILTER_MAX)
        if not shortlist:
            print("  None (either Δ not computed yet or all above threshold).")
        else:
            for r in shortlist:
                print(f"  {expiry} AMZN {fmt(r['strike'],2)}C  Δ={fmt(r['delta'],4)}  mid={fmt(r['mid'],2)}  IV={fmt(r['iv'],4)}")

    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    util.startLoop()  # nest_asyncio compatibility when needed
    asyncio.run(main())