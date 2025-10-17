#!/usr/bin/env python3
# amzn_chain_multi.py
#
# Fetch 3 nearest AMZN option expiries (SMART exchange), 15 strikes each (calls & puts)
# Displays bid/ask, IV, delta, and mid price.

import os, math, asyncio
from datetime import datetime, timezone
from ib_insync import IB, Stock, Option, util

SYM = 'AMZN'
EXCHANGE = 'SMART'
CURRENCY = 'USD'
STRIKES_AROUND = 7       # 7 below + 7 above + ATM ≈ 15 strikes
WAIT_SEC = 3.5
GENERIC_TICKS = '106'    # include model data (implied vol, greeks)
N_EXPIRIES = 3           # number of nearest expiries to display

# ---------- Helper Functions ----------

def safe(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v

def pick_expirations(chains, n=3):
    today = datetime.now(timezone.utc).date()
    rec = next((r for r in chains if getattr(r, 'exchange', None) == 'SMART'), chains[0])
    exps = sorted({datetime.strptime(e, '%Y%m%d').date() for e in rec.expirations})
    exps = [e for e in exps if e >= today]
    if not exps:
        exps = sorted({datetime.strptime(e, '%Y%m%d').date() for e in rec.expirations})
    return rec, [e.strftime('%Y%m%d') for e in exps[:n]]

def nearest_strikes(all_strikes, spot, n_each_side=7):
    s = sorted(float(x) for x in all_strikes if x and x > 0)
    if not s: return []
    idx = min(range(len(s)), key=lambda i: abs(s[i] - spot))
    lo, hi = max(0, idx - n_each_side), min(len(s), idx + n_each_side + 1)
    return s[lo:hi]

# ---------- Main Script ----------

async def main():
    host = os.getenv("IB_HOST", "127.0.0.1")
    port = int(os.getenv("IB_PORT", "4001"))
    cid = int(os.getenv("IB_CLIENT_ID", "1101"))

    ib = IB()
    await ib.connectAsync(host, port, clientId=cid, timeout=10)
    ib.reqMarketDataType(1)

    # --- Qualify underlying ---
    stk = Stock(SYM, EXCHANGE, CURRENCY, primaryExchange='NASDAQ')
    stk = (await ib.qualifyContractsAsync(stk))[0]

    # --- Get spot price ---
    t = ib.reqMktData(stk, '', False, False)
    await asyncio.sleep(1)
    spot = next((x for x in (t.marketPrice(), t.last, t.close) if x and not math.isnan(x)), None)
    if not spot:
        ib.reqMarketDataType(4)
        t = ib.reqMktData(stk, '', False, False)
        await asyncio.sleep(1)
        spot = next((x for x in (t.marketPrice(), t.last, t.close) if x and not math.isnan(x)), None)
    if not spot:
        raise RuntimeError("No AMZN spot price.")
    print(f"\n📈 AMZN spot: {spot:.2f} USD")

    # --- Get option chains ---
    chains = await ib.reqSecDefOptParamsAsync(SYM, '', 'STK', stk.conId)
    if not chains:
        raise RuntimeError("No option chain returned.")
    rec, expiries = pick_expirations(chains, N_EXPIRIES)
    all_strikes = sorted(rec.strikes)
    trading_class = getattr(rec, 'tradingClass', SYM)

    for expiry in expiries:
        strikes_sel = nearest_strikes(all_strikes, spot, STRIKES_AROUND)
        opts = []
        for k in strikes_sel:
            for right in ('C', 'P'):
                opts.append(Option(
                    symbol=SYM,
                    lastTradeDateOrContractMonth=expiry,
                    strike=float(k),
                    right=right,
                    exchange=EXCHANGE,
                    currency=CURRENCY,
                    tradingClass=trading_class
                ))
        opts = await ib.qualifyContractsAsync(*opts)
        for c in opts:
            ib.reqMktData(c, GENERIC_TICKS, False, False)
        await asyncio.sleep(WAIT_SEC)

        tickers = [ib.ticker(c) for c in opts]
        rows = []
        for k in strikes_sel:
            c = next((t for t in tickers if t and t.contract.right == 'C' and abs(t.contract.strike - k) < 1e-6), None)
            p = next((t for t in tickers if t and t.contract.right == 'P' and abs(t.contract.strike - k) < 1e-6), None)

            def mid(b, a): return None if (b is None or a is None) else round((b + a) / 2, 3)
            def greeks(t, field): g = getattr(t, 'modelGreeks', None); return safe(getattr(g, field, None)) if g else None

            rows.append({
                'Strike': k,
                'C_Bid': safe(getattr(c, 'bid', None)),
                'C_Ask': safe(getattr(c, 'ask', None)),
                'C_Mid': mid(getattr(c, 'bid', None), getattr(c, 'ask', None)),
                'C_Delta': greeks(c, 'delta'),
                'C_IV': greeks(c, 'impliedVol'),
                'P_Bid': safe(getattr(p, 'bid', None)),
                'P_Ask': safe(getattr(p, 'ask', None)),
                'P_Mid': mid(getattr(p, 'bid', None), getattr(p, 'ask', None)),
                'P_Delta': greeks(p, 'delta'),
                'P_IV': greeks(p, 'impliedVol'),
            })

        print(f"\n=== Expiry {expiry} | Strikes {len(strikes_sel)} ===")
        print(util.df(rows).to_string(index=False))

    ib.disconnect()

if __name__ == "__main__":
    asyncio.run(main())