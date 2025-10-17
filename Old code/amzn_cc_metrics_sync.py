#!/usr/bin/env python3
# amzn_cc_metrics_sync.py
#
# Synchronous ib_insync version (no asyncio.run).
# Shows AMZN covered-call metrics + synthetic short carry (LockPnL/sh, AnnOnSpot%).

import os, math, random
from datetime import datetime, timezone
from ib_insync import IB, Stock, Option, util

SYM = 'AMZN'
EXCHANGE = 'SMART'
CURRENCY = 'USD'
PRIMARY = 'NASDAQ'
STRIKES_AROUND = 7        # strikes each side of spot
WAIT_SEC = 3.5            # seconds to let option quotes/greeks populate
GENERIC_TICKS = '106'     # model greeks (implied vol, delta)

def safe(v):
    return None if v is None or (isinstance(v, float) and (math.isnan(v))) else v

def pick_expiration(chains):
    """Pick soonest >= today from the SMART chain record (fallback to first)."""
    today = datetime.now(timezone.utc).date()
    rec = next((r for r in chains if getattr(r, 'exchange', None) == 'SMART'), chains[0])
    exps = sorted({datetime.strptime(e, '%Y%m%d').date() for e in rec.expirations})
    exps = [e for e in exps if e >= today] or [exps[-1]]
    return rec, exps[0].strftime('%Y%m%d')

def nearest_strikes(all_strikes, spot, n_each_side):
    s = sorted(float(x) for x in all_strikes if x and x > 0)
    idx = min(range(len(s)), key=lambda i: abs(s[i] - spot))
    lo = max(0, idx - n_each_side)
    hi = min(len(s), idx + n_each_side + 1)
    return s[lo:hi]

def fetch_spot_with_fallback(ib: IB, stk: Stock, poll_live=5.0, poll_delayed=6.0):
    """Try LIVE first, then DELAYED, polling for a valid price."""
    # LIVE
    ib.reqMarketDataType(1)
    ib.reqMktData(stk, '', False, False)
    steps = max(1, int(poll_live / 0.25))
    for _ in range(steps):
        ib.sleep(0.25)
        tk = ib.ticker(stk)
        if not tk: 
            continue
        v = next((x for x in (tk.marketPrice(), tk.last, tk.close) if safe(x)), None)
        if v is not None:
            return float(v)

    # DELAYED
    ib.reqMarketDataType(4)
    ib.reqMktData(stk, '', False, False)
    steps = max(1, int(poll_delayed / 0.25))
    for _ in range(steps):
        ib.sleep(0.25)
        tk = ib.ticker(stk)
        if not tk: 
            continue
        v = next((x for x in (tk.marketPrice(), tk.last, tk.close) if safe(x)), None)
        if v is not None:
            return float(v)

    return None

def main():
    host = os.getenv("IB_HOST", "127.0.0.1")
    port = int(os.getenv("IB_PORT", "4001"))
    cid_env = os.getenv("IB_CLIENT_ID", "")
    client_id = int(cid_env) if cid_env.strip() else random.randint(1000, 9999)

    ib = IB()
    try:
        print(f"Connecting to IB @ {host}:{port} clientId={client_id} …")
        ib.connect(host, port, clientId=client_id, timeout=10)

        # Underlying
        stk = Stock(SYM, EXCHANGE, CURRENCY, primaryExchange=PRIMARY)
        stk = ib.qualifyContracts(stk)[0]

        # Spot with robust fallback
        spot = fetch_spot_with_fallback(ib, stk)
        if spot is None:
            raise RuntimeError("Failed to fetch AMZN spot (live and delayed). Check market data permissions and that only one client is connected.")

        # Option chain metadata
        chains = ib.reqSecDefOptParams(SYM, '', 'STK', stk.conId)
        if not chains:
            raise RuntimeError("reqSecDefOptParams returned no data.")
        rec, expiry = pick_expiration(chains)
        all_strikes = sorted(rec.strikes)
        strikes_sel = nearest_strikes(all_strikes, spot, STRIKES_AROUND)
        trading_class = getattr(rec, 'tradingClass', SYM)

        # Build both calls & puts at those strikes
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
            raise RuntimeError("qualifyContracts returned no option contracts; check earlier errors.")

        # Subscribe quotes/greeks
        for c in contracts:
            ib.reqMktData(c, GENERIC_TICKS, False, False)
        ib.sleep(WAIT_SEC)

        # Ticker map
        ticks = {(c.right, round(float(c.strike), 4)): ib.ticker(c) for c in contracts}

        today = datetime.now(timezone.utc).date()
        dte = (datetime.strptime(expiry, '%Y%m%d').date() - today).days or 1

        rows = []
        for k in strikes_sel:
            kf = round(float(k), 4)
            ct = ticks.get(('C', kf))
            pt = ticks.get(('P', kf))
            if not ct or not pt:
                continue

            c_bid, c_ask = safe(ct.bid), safe(ct.ask)
            p_bid, p_ask = safe(pt.bid), safe(pt.ask)
            if c_bid is None or c_ask is None or p_ask is None:
                # need call bid/ask and put ask for conversion carry
                continue

            c_mid = (c_bid + c_ask) / 2.0
            g = getattr(ct, 'modelGreeks', None)
            delta = safe(getattr(g, 'delta', None))
            iv = safe(getattr(g, 'impliedVol', None))

            # Covered-call metrics
            premium_pct = (c_mid / spot) * 100.0
            ann_yield = premium_pct * (365.0 / dte)
            prob_itm = (abs(delta) if delta is not None else 0.0) * 100.0
            breakeven = spot - c_mid

            # Synthetic short (conversion) carry vs spot:
            # LockPnL/sh = (K - S) + Call_bid - Put_ask
            lock_pnl_sh = (kf - spot) + (c_bid - p_ask)
            ann_on_spot = (lock_pnl_sh / spot) * 100.0 * (365.0 / dte)

            if (delta is not None) and (abs(delta) <= 0.25):
                rows.append({
                    'Expiry': expiry,
                    'DTE': dte,
                    'Strike': float(kf),
                    'Bid': round(c_bid, 3),
                    'Ask': round(c_ask, 3),
                    'Mid': round(c_mid, 3),
                    'Delta': round(delta, 3),
                    'IV': None if iv is None else round(iv, 3),
                    'Premium%': round(premium_pct, 2),
                    'AnnYield%': round(ann_yield, 1),
                    'ProbITM%': round(prob_itm, 1),
                    'Breakeven': round(breakeven, 2),
                    'LockPnL/sh': round(lock_pnl_sh, 3),
                    'AnnOnSpot%': round(ann_on_spot, 2),
                })

        print(f"\n📈 AMZN spot {spot:.2f} | Expiry {expiry} | DTE {dte}")
        if rows:
            df = util.df(rows).sort_values(['AnnYield%','Strike'], ascending=[False, True])
            print(df.to_string(index=False))
        else:
            print("No strikes passed Δ ≤ 0.25 (or quotes/greeks not available yet). "
                  "Try increasing WAIT_SEC, confirm delayed options permissions, or loosen the delta filter.")

    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

if __name__ == "__main__":
    main()