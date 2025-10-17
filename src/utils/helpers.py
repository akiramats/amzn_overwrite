import datetime as dt
from math import isfinite

def dte_from_ib_expiry(exp_str: str) -> int:
    expiry = dt.datetime.strptime(exp_str, "%Y%m%d").date()
    today = dt.date.today()
    return (expiry - today).days

def mid_price(bid, ask, last=None):
    if isfinite(bid or float('nan')) and isfinite(ask or float('nan')) and bid and ask:
        return round((bid + ask) / 2, 2)
    return round(last or 0.0, 2)
