#!/usr/bin/env python3
# scan_to_json.py — run one scan and dump JSON for webapp_static

import os, json, random
from datetime import datetime
from ib_insync import IB, Stock
import importlib

# import your main metrics module
cc = importlib.import_module('amzn_cc_metrics_4x_sync_v2')

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_PATH = os.path.join(OUT_DIR, "last_scan.json")

def run_scan_once():
    host = os.getenv("IB_HOST", "127.0.0.1")
    port = int(os.getenv("IB_PORT", "4001"))
    cid_env = os.getenv("IB_CLIENT_ID", "")
    client_id = int(cid_env) if cid_env.strip() else random.randint(2000, 9999)

    ACCOUNT_ID = os.getenv('IB_ACCOUNT', cc.ACCOUNT_ID)
    # prefer delayed outside RTH unless you override
    cc.FORCE_MKT_TYPE = int(os.getenv('IB_MKT_TYPE', '4'))

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=10)

        # Quiet the incomplete-account warning
        def _quiet_errors(reqId, code, msg, contract):
            if code in (10275,):
                return
            print(f"[IB] {code} {msg}")
        ib.errorEvent += _quiet_errors

        accts = ib.managedAccounts()
        if ACCOUNT_ID not in accts:
            raise RuntimeError(f"Account {ACCOUNT_ID} not in managedAccounts: {accts}")

        stk = Stock(cc.SYM, cc.EXCHANGE, cc.CURRENCY, primaryExchange=cc.PRIMARY)
        stk = ib.qualifyContracts(stk)[0]

        spot = cc.fetch_spot_with_fallback(ib, stk)
        if spot is None:
            raise RuntimeError("Failed to fetch spot")

        chains = ib.reqSecDefOptParams(cc.SYM, '', 'STK', stk.conId)
        if not chains:
            raise RuntimeError("reqSecDefOptParams returned no data")
        expiries = cc.pick_nearest_expiries(chains, n=cc.MAX_EXPIRIES)

        blocks = []
        for expiry in expiries:
            # NOTE: new signature — pass stk AND spot
            rows, best_kh, best_ann_on_spot, dyn_diag = cc.one_expiry_block(ib, stk, spot, expiry)
            blocks.append({
                "expiry": expiry,
                "best_kh": best_kh,
                "best_ann_on_spot": best_ann_on_spot,
                "calib": dyn_diag,
                "rows": rows or []
            })

        payload = {
            "asof": datetime.now().isoformat(timespec="seconds"),
            "spot": round(float(spot), 2),
            "account": ACCOUNT_ID,
            "symbol": cc.SYM,
            "blocks": blocks
        }

        os.makedirs(OUT_DIR, exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[ok] wrote {OUT_PATH}")
        return 0

    except Exception as e:
        # Still write something so the UI can show the error
        os.makedirs(OUT_DIR, exist_ok=True)
        err_payload = {
            "asof": datetime.now().isoformat(timespec="seconds"),
            "error": str(e)
        }
        with open(OUT_PATH, "w") as f:
            json.dump(err_payload, f, indent=2)
        print(f"[warn] wrote error to {OUT_PATH}: {e}")
        return 1

    finally:
        try:
            ib.disconnect()
        except Exception:
            pass

if __name__ == "__main__":
    raise SystemExit(run_scan_once())