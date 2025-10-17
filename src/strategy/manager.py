from ib_insync import *
from src.utils.helpers import dte_from_ib_expiry, mid_price

def current_short_calls(ib: IB, account=None):
    out = []
    for p in ib.positions():
        if account and p.account != account:
            continue
        c = p.contract
        if c.secType == "OPT" and c.right == "C" and p.position < 0:
            out.append(p)
    return out

def place_sto(ib: IB, opt, qty: int, limit: float):
    if os.getenv("DRY_RUN", "0") == "1":
        logger.info(f"[DRY_RUN] Would SELL {qty}x {opt.localSymbol} @ {limit}")
        return {"status": "DRY_RUN", "symbol": opt.localSymbol, "qty": qty, "limit": limit}

    order = LimitOrder("SELL", qty, limit)
    trade = ib.placeOrder(opt, order)
    trade.waitUntilFilled(timeout=60)
    logger.info(f"Placed SELL {qty}x {opt.localSymbol} @ {limit}, status={trade.orderStatus.status}")
    return trade

def manage_positions(ib: IB, cfg: dict, account=None):
    warnings = []
    for pos in current_short_calls(ib, account):
        c = pos.contract
        t = ib.reqMktData(c, "100,101,104,106", False, False)
        ib.sleep(1.0)
        g = t.modelGreeks
        delta = abs(g.delta) if g else None
        mark = mid_price(t.bid, t.ask, t.last)
        dte = dte_from_ib_expiry(c.lastTradeDateOrContractMonth)
        ib.cancelMktData(c)

        if delta is not None and dte > cfg["management"]["mandatory_roll_min_dte"]:
            if delta >= cfg["management"]["mandatory_roll_delta"]:
                warnings.append(f"ROLL needed: {c.localSymbol} Δ={delta:.2f} DTE={dte}")
    return warnings
