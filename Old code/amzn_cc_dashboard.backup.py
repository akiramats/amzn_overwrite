# amzn_cc_dashboard.py
# Minimal live dashboard via IB Gateway/TWS using ib_insync
# Shows: AMZN spot, 4 nearest expiries, CALLs with delta < 0.25,
# prob ITM, IV, delta, net premium and annualized return (after fees).

import os, math, asyncio, sys
from datetime import date, datetime
from typing import List, Tuple
import pandas as pd
from ib_insync import IB, Stock, Option, util

# ---- CONFIG (adjust if you want) --------------------------------------------
HOST     = os.getenv("IB_HOST", "127.0.0.1")
# Live: TWS=7496, IB Gateway=4001 | Paper: TWS=7497, IB Gateway=4002
PORT     = int(os.getenv("IB_PORT", "4001"))      # <- live Gateway default
CLIENTID = int(os.getenv("IB_CLIENT_ID", "1101"))
BASE_CCY = os.getenv("BASE_CCY", "USD")           # for display only
MAX_CONTRACTS = 48                                 # keep under pacing limits
STRIKE_PCT_UPPER = 0.25                            # consider OTM calls up to +25%
FEE_PER_CONTRACT = float(os.getenv("FEE_PER_CONTRACT", "0.85"))  # all-in est. USD
# -----------------------------------------------------------------------------

util.patchAsyncio()

def _parse_expiry(e: str) -> date:
    s = str(e).strip()
    if "-" in s:  # 'YYYY-MM-DD'
        return datetime.strptime(s, "%Y-%m-%d").date()
    if len(s) == 8 and s.isdigit():  # 'YYYYMMDD'
        return datetime.strptime(s, "%Y%m%d").date()
    raise ValueError(f"Unrecognized expiry: {e}")

def _norm_cdf(x: float) -> float:
    # standard normal CDF without scipy
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _inv_norm_cdf(p: float) -> float:
    # Acklam approximation (sufficient here); p in (0,1)
    if p <= 0.0 or p >= 1.0:
        return float("nan")
    a1=-39.6968302866538; a2=220.946098424521; a3=-275.928510446969
    a4=138.357751867269;  a5=-30.6647980661472; a6=2.50662827745924
    b1=-54.4760987982241; b2=161.585836858041; b3=-155.698979859887
    b4=66.8013118877197;  b5=-13.2806815528857
    c1=-7.78489400243029e-03; c2=-0.322396458041136; c3=-2.40075827716184
    c4=-2.54973253934373;   c5=4.37466414146497;  c6=2.93816398269878
    d1=7.78469570904146e-03; d2=0.32246712907004; d3=2.445134137143; d4=3.75440866190742
    plow=0.02425; phigh=1.0-plow
    if p < plow:
        q = math.sqrt(-2.0*math.log(p))
        x = (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1)
    elif p > phigh:
        q = math.sqrt(-2.0*math.log(1.0-p))
        x = -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1)
    else:
        q = p-0.5; r=q*q
        x = (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q / (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1)
    # one Newton step
    e = _norm_cdf(x)-p
    x = x - e/(math.sqrt(2*math.pi)*math.exp(-x*x/2))
    return x

async def main():
    ib = IB()
    try:
        print(f"Connecting to IB @ {HOST}:{PORT} clientId={CLIENTID} ...")
        await ib.connectAsync(HOST, PORT, clientId=CLIENTID, timeout=8)
        if not ib.isConnected():
            print("Failed to connect.")
            return

        # live data
        ib.reqMarketDataType(1)

        # Underlying — SMART + primary exchange = best for options mapping
        amzn = Stock('AMZN', 'SMART', 'USD', primaryExchange='NASDAQ')
        [amzn] = await ib.qualifyContractsAsync(amzn)

        # streaming for spot
        t_under = ib.reqMktData(amzn, '', snapshot=False)
        for _ in range(40):  # up to ~8s
            await asyncio.sleep(0.2)
            spot = next((px for px in (t_under.marketPrice(), t_under.last, t_under.close)
                         if px is not None and math.isfinite(px)), None)
            if spot: break
        if not spot:
            # fallback to 1-min historical
            bars = await ib.reqHistoricalDataAsync(
                amzn, endDateTime='', durationStr='1 D', barSizeSetting='1 min',
                whatToShow='TRADES', useRTH=False, formatDate=1
            )
            spot = float(bars[-1].close) if bars else float('nan')

        print(f"\nAMZN spot ~ {spot:.2f} {BASE_CCY}\n")

        # Option chain meta (SMART for options)
        chain = await ib.reqSecDefOptParamsAsync(amzn.symbol, '', amzn.secType, amzn.conId)
        if not chain:
            print("No option chain metadata.")
            return
        meta = next((x for x in chain if x.exchange == 'SMART'), chain[0])

        # 4 nearest expiries in the future
        today = date.today()
        exps = sorted([_parse_expiry(e) for e in meta.expirations if _parse_expiry(e) >= today])[:4]
        if not exps:
            print("No near expiries.")
            return

        # Candidate strikes: >= spot up to +25% OTM (for covered-call OTM)
        strikes = sorted(k for k in meta.strikes if (k >= spot and (k-spot)/spot <= STRIKE_PCT_UPPER))
        if not strikes:
            print("No OTM strikes found within range.")
            return

        # --- Build option prototypes for 4 nearest expiries, OTM strikes ---
        per_expiry_budget = max(1, MAX_CONTRACTS // len(exps))
        protos = []
        for e in exps:
            near = sorted(strikes, key=lambda k: abs(k - spot))[:per_expiry_budget]
            for k in near:
                protos.append(Option('AMZN', e.strftime('%Y%m%d'), k, 'C', exchange='SMART', currency='USD'))

        # --- Resolve each prototype into a real contract and subscribe by conId ---
        real_contracts = []
        for p in protos:
            cds = await ib.reqContractDetailsAsync(p)
            if cds:
                # Use conId (most robust way to avoid 'no security definition')
                real_contracts.append(Option(conId=cds[0].contract.conId, exchange='SMART'))

        if not real_contracts:
            print("No qualified option contracts via ContractDetails.")
            return

        # Ask for live; if nothing comes we’ll switch to delayed-frozen during the wait loop
        ib.reqMarketDataType(1)  # 1=REALTIME, 3=DELAYED_FROZEN
        for c in real_contracts:
            ib.reqMktData(c, '', snapshot=False)      # NBBO
            ib.reqMktData(c, '106', snapshot=False)   # model/Greeks

        # wait up to ~10s for either NBBO or model values; if none after 4s, flip to delayed-frozen
        for i in range(20):  # 20 * 0.5s = 10s
            await asyncio.sleep(0.5)
            if i == 8:
                # If still nothing after ~4s, flip to delayed-frozen so you see values outside RTH
                ib.reqMarketDataType(3)

            ready = True
            for c in real_contracts:
                t = ib.ticker(c)
                if not t:
                    ready = False; break
                has_quotes = (t.bid is not None) or (t.ask is not None)
                has_model  = getattr(getattr(t,'modelGreeks',None),'optPrice',None) is not None
                if not (has_quotes or has_model):
                    ready = False; break
            if ready:
                break

        # Fallback during closed hours: try Frozen, then Delayed-Frozen
        if not ready:
            ib.reqMarketDataType(2)  # Frozen
            for _ in range(10):
                await asyncio.sleep(0.5)
                ready = True
                for c in real_contracts:
                    t = ib.ticker(c)
                    if not t:
                        ready = False; break
                    has_quotes = (t.bid is not None) or (t.ask is not None)
                    has_model  = getattr(getattr(t,'modelGreeks',None),'optPrice',None) is not None
                    if not (has_quotes or has_model):
                        ready = False; break
                if ready:
                    break
        if not ready:
            ib.reqMarketDataType(4)  # Delayed-Frozen
            await asyncio.sleep(2.0)

        # ---------------- Build rows (this was indented incorrectly before) -----------
        rows = []
        today = date.today()
        for c in real_contracts:
            t = ib.ticker(c)
            if not t:
                continue

            # --- Quote fields ---
            bid = t.bid if (t.bid is not None and math.isfinite(t.bid)) else float('nan')
            ask = t.ask if (t.ask is not None and math.isfinite(t.ask)) else float('nan')
            mid = (bid + ask) / 2 if (math.isfinite(bid) and math.isfinite(ask)) else float('nan')

            # --- Model greeks / theoretical price ---
            g = getattr(t, 'modelGreeks', None)
            model_px = float(getattr(g, 'optPrice', float('nan'))) if g else float('nan')
            delta    = float(getattr(g, 'delta', float('nan'))) if g else float('nan')
            iv       = float(getattr(g, 'impliedVol', float('nan'))) if g else float('nan')

            # --- Fallback: use model price if NBBO mid is missing ---
            if not math.isfinite(mid):
                mid = model_px

            # --- Filter for low-delta calls (covered-call zone) ---
            if not (math.isfinite(delta) and abs(delta) < 0.25):
                continue

            # DTE
            exp_dt = _parse_expiry(c.lastTradeDateOrContractMonth)
            dte = (exp_dt - today).days

            # Prob ITM ≈ N(d2) using delta = N(d1) ⇒ d2 = N^-1(delta) - σ√T
            prob_itm = None
            if math.isfinite(iv) and iv > 0 and dte >= 0 and math.isfinite(delta) and 0.0 < delta < 1.0:
                T = max(dte, 0) / 365.0
                d1 = _inv_norm_cdf(delta)
                d2 = d1 - iv*math.sqrt(T) if T > 0 else d1
                prob_itm = _norm_cdf(d2)

            # Premium & AR (per contract), net of fees
            net_prem = None
            ar = None
            if math.isfinite(mid) and mid > 0 and dte > 0 and math.isfinite(spot) and spot > 0:
                gross = mid * 100.0
                net_prem = gross - FEE_PER_CONTRACT
                ar = (net_prem / (spot * 100.0)) * (365.0 / dte) * 100.0  # percent

            rows.append({
                "Expiry": exp_dt.isoformat(),
                "DTE": dte,
                "Strike": float(c.strike),
                "Bid": None if not math.isfinite(bid) else round(bid, 2),
                "Ask": None if not math.isfinite(ask) else round(ask, 2),
                "Mid": None if not math.isfinite(mid) else round(mid, 2),
                "Delta": None if not math.isfinite(delta) else round(delta, 4),
                "IV%": None if not math.isfinite(iv) else round(iv*100.0, 2),
                "ProbITM%": None if prob_itm is None else round(prob_itm*100.0, 2),
                "NetPrem($)": None if net_prem is None else round(net_prem, 2),
                "AR% (net)": None if ar is None else round(ar, 2)
            })

        df = pd.DataFrame(rows)
        if df.empty:
            print("No contracts passed the Δ<0.25 filter (or quotes not populated yet).")
            return

        # Sort best by AR then DTE
        df = df.sort_values(by=["AR% (net)", "DTE"], ascending=[False, True]).reset_index(drop=True)

        # Print tidy output
        print("==== AMZN Covered-Call Scanner (live) ====")
        print(f"Spot: {spot:.2f} {BASE_CCY}   Fees/contract: ${FEE_PER_CONTRACT:.2f}")
        print(f"Nearest expiries: {', '.join([d.isoformat() for d in exps])}\n")
        with pd.option_context('display.max_rows', 200, 'display.width', 140):
            print(df.to_string(index=True))

    finally:
        try:
            if ib.isConnected():
                ib.disconnect()
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(main())