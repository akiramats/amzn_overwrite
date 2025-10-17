# ib_test_one.py (tweak)
import os, asyncio, math
from ib_insync import IB, Stock, Option

HOST=os.getenv("IB_HOST","127.0.0.1"); PORT=int(os.getenv("IB_PORT","4001")); CID=int(os.getenv("IB_CLIENT_ID","1101"))

async def main():
    ib=IB(); await ib.connectAsync(HOST, PORT, clientId=CID, timeout=8)

    # Since you’re off-hours, go straight to delayed-frozen
    ib.reqMarketDataType(3)

    # Qualify underlying and subscribe first
    [amzn] = await ib.qualifyContractsAsync(Stock('AMZN','SMART','USD', primaryExchange='NASDAQ'))
    ib.reqMktData(amzn, '', snapshot=False)
    # Wait a moment for delayed quote
    for _ in range(20):
        await asyncio.sleep(0.3)
        t = ib.ticker(amzn)
        if t and (t.marketPrice() is not None or t.last is not None or t.close is not None):
            break

    # Resolve a real option contract (you already saw 762757074)
    cds = await ib.reqContractDetailsAsync(Option('AMZN','20251017',215,'C', exchange='SMART', currency='USD'))
    if not cds:
        print("Couldn’t resolve that option again (secdef hiccup)."); ib.disconnect(); return
    opt = Option(conId=cds[0].contract.conId, exchange='SMART')

    # Subscribe to option: NBBO + model greeks
    ib.reqMktData(opt, '106', snapshot=False)

    # Poll a few seconds
    for _ in range(40):
        await asyncio.sleep(0.3)
        t = ib.ticker(opt)
        if not t: continue
        bid, ask = t.bid, t.ask
        g = getattr(t, 'modelGreeks', None)
        if (not (bid is None or math.isnan(bid))) or (not (ask is None or math.isnan(ask))) or (g and g.delta is not None):
            print("Bid/Ask:", bid, ask, "| ModelPx:", getattr(g,'optPrice',None), "Delta:", getattr(g,'delta',None), "IV:", getattr(g,'impliedVol',None))
            break
    else:
        print("Still no data — likely no delayed options permission or the farm is sleepy off-hours.")

    ib.disconnect()

if __name__ == "__main__":
    asyncio.run(main())