#!/usr/bin/env python3
from ib_insync import IB, Stock, Option
import math, sys

HOST, PORT, CID = '127.0.0.1', 4001, 52525

def ok(x):  # positive usable price
    return x is not None and isinstance(x, (int, float)) and x > 0 and not math.isnan(x)

def fmt(x):
    return '—' if not ok(x) else f'{x:.2f}'

def qualify_option(ib, sym, expiry, k, right):
    probe = Option(sym, expiry, float(k), right.upper(), 'SMART', currency='USD')
    det = ib.reqContractDetails(probe)
    if not det:
        raise RuntimeError(f'No contract details for {sym} {expiry} {right}{k}')
    return det[0].contract

def snap(ib, contract):
    t = ib.reqMktData(contract, genericTickList='', snapshot=True, regulatorySnapshot=False)
    ib.sleep(3.0)  # give snapshot time
    b, a = t.bid, t.ask
    m = (b + a)/2.0 if ok(b) and ok(a) else None
    return b, a, m

def main():
    if len(sys.argv) == 5:
        sym, exp, strike, right = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    else:
        sym, exp, strike, right = 'AMZN', '20251024', 220.0, 'C'

    ib = IB()
    print(f'Connecting {HOST}:{PORT} clientId={CID} …')
    ib.connect(HOST, PORT, clientId=CID, timeout=15)

    try:
        # just to confirm the session is alive
        stk = Stock(sym, 'SMART', 'USD', primaryExchange='NASDAQ')
        [stk] = ib.qualifyContracts(stk)
        ut = ib.reqMktData(stk, '', snapshot=True, regulatorySnapshot=False)
        ib.sleep(2.0)
        spot = ut.marketPrice() or ut.last or ut.close
        print(f'Spot {sym}: {spot:.2f}' if spot else f'Spot {sym}: —')

        opt = qualify_option(ib, sym, exp, strike, right)
        print('Qualified:', opt)

        # 1) Real-time FROZEN (needs real-time OPRA)
        ib.reqMarketDataType(2)
        b2, a2, m2 = snap(ib, opt)
        print(f'[type 2 frozen]        Bid: {fmt(b2)}  Ask: {fmt(a2)}  Mid: {fmt(m2)}')

        # 2) DELAYED-FROZEN (needs delayed OPRA to API)
        ib.reqMarketDataType(4)
        b4, a4, m4 = snap(ib, opt)
        print(f'[type 4 delayed-frozen] Bid: {fmt(b4)}  Ask: {fmt(a4)}  Mid: {fmt(m4)}')

        if not any(ok(x) for x in (b2, a2, b4, a4)):
            print('\nNo usable frozen quotes yet (still —). This indicates:')
            print(' • OPRA (US Options) data isn’t reaching the API (real-time or delayed), or')
            print(' • OPRA agreements not accepted in Account Management, or')
            print(' • TWS/IBG needs a restart after enabling permissions.')
    finally:
        ib.disconnect()

if __name__ == '__main__':
    main()