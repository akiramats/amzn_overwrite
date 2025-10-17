#!/usr/bin/env python3
# amzn_chain_10.py — fixed for modern IB API (OptionChain objects)

import os, math, asyncio
from datetime import datetime, timezone
from ib_insync import IB, Stock, Option, util

SYM = 'AMZN'
EXCHANGE = 'SMART'
CURRENCY = 'USD'
STRIKES_AROUND = 5
WAIT_SEC = 3.0
GENERIC_TICKS = '106'  # model data

def pick_nearest_expiration(chains):
    """Return (record, expiry) for the soonest expiry >= today."""
    today = datetime.now(timezone.utc).date()
    # choose the record with exchange == SMART if exists
    rec = next((r for r in chains if getattr(r, 'exchange', None) == 'SMART'), chains[0])
    exps = sorted({datetime.strptime(e, '%Y%m%d').date() for e in rec.expirations})
    for e in exps:
        if e >= today:
            return rec, e.strftime('%Y%m%d')
    return rec, exps[-1].strftime('%Y%m%d')

def nearest_strikes(all_strikes, spot, n_each_side=5):
    s = sorted(float(x) for x in all_strikes if x and x > 0)
    if not s:
        return []
    idx = min(range(len(s)), key=lambda i: abs(s[i] - spot))
    lo, hi = max(0, idx - n_each_side), min(len(s), idx + n_each_side + 1)
    return s[lo:hi]

def safe(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v

async def main():
    host = os.getenv("IB_HOST", "127.0.0.1")
    port = int(os.getenv("IB_PORT", "4001"))
    cid = int(os.getenv("IB_CLIENT_ID", "1101"))

    ib = IB()
    await ib.connectAsync(host, port, clientId=cid, timeout=10)
    ib.reqMarketDataType(1)

    # qualify underlying
    stk = Stock(SYM, EXCHANGE, CURRENCY, primaryExchange='NASDAQ')
    stk = (await ib.qualifyContractsAsync(stk))[0]

    # spot
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

    # option chains
    chains = await ib.reqSecDefOptParamsAsync(SYM, '', 'STK', stk.conId)
    if not chains:
        raise RuntimeError("No option chains returned.")
    rec, expiry = pick_nearest_expiration(chains)
    strikes_all = sorted(rec.strikes)
    strikes_sel = nearest_strikes(strikes_all, spot, STRIKES_AROUND)
    if not strikes_sel:
        raise RuntimeError("No strikes found near spot.")

    trading_class = getattr(rec, 'tradingClass', SYM)
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
        call = next((t for t in tickers if t.contract.right == 'C' and abs(t.contract.strike - k) < 1e-6), None)
        put = next((t for t in tickers if t.contract.right == 'P' and abs(t.contract.strike - k) < 1e-6), None)
        def g(t, field): return safe(getattr(getattr(t, 'modelGreeks', None), field, None)) if t else None
        rows.append({
            'Strike': k,
            'C_Bid': safe(getattr(call, 'bid', None)),
            'C_Ask': safe(getattr(call, 'ask', None)),
            'C_IV': g(call, 'impliedVol'),
            'P_Bid': safe(getattr(put, 'bid', None)),
            'P_Ask': safe(getattr(put, 'ask', None)),
            'P_IV': g(put, 'impliedVol'),
        })

    print(f"\nAMZN spot {spot:.2f} | Expiry {expiry} | Showing {len(strikes_sel)} strikes")
    print(util.df(rows).to_string(index=False))
    ib.disconnect()

if __name__ == '__main__':
    asyncio.run(main())