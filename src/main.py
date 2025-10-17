import os, yaml
from dotenv import load_dotenv
from ib_insync import IB
from apscheduler.schedulers.background import BackgroundScheduler
from loguru import logger

from src.utils.ib_client import connect_ib   # must call reqMarketDataType(4) inside
from src.strategy.strike_selector import pick_contract
from src.strategy.manager import manage_positions, place_sto

def load_cfg():
    with open("config/aggressive_income.yaml","r") as f:
        return yaml.safe_load(f)

def get_positions(ib: IB):
    """
    Fetch positions without reqAccountUpdates (avoids 'Group name cannot be null').
    Filter to the configured account if provided.
    """
    acct = os.getenv("IB_ACCOUNT_ID") or ""
    # ib.positions() issues reqPositions under the hood and waits
    positions = ib.positions()
    if acct:
        positions = [p for p in positions if p.account == acct]
    return positions

def sell_if_needed(ib: IB, cfg: dict):
    acct = os.getenv("IB_ACCOUNT_ID") or ""
    positions = get_positions(ib)

    shares = sum(
        p.position for p in positions
        if p.contract.symbol == "AMZN" and p.contract.secType == "STK"
    )
    short_calls = [
        p for p in positions
        if p.contract.secType == "OPT" and p.contract.right == "C" and p.position < 0
    ]

    max_contracts = min(cfg["assets"]["shares_held"] // 100,
                        cfg["assets"]["contracts_per_cycle"])
    existing = abs(sum(int(p.position) for p in short_calls))
    missing = max_contracts - existing
    if missing <= 0:
        logger.info("No new calls to sell.")
        return

    target_delta = cfg["objectives"]["base_target_delta"]
    sel = pick_contract(
        ib,
        target_delta=target_delta,
        min_dte=cfg["tenor"]["min_dte"],
        max_dte=cfg["tenor"]["max_dte"]
    )
    if not sel:
        logger.warning("No suitable contract found.")
        return

    opt = sel["opt"]
    mid = sel["mid"]
    qty = missing
    limit = max(0.01, round(mid - 0.01, 2))
    logger.info(f"Selling {qty}x {opt.localSymbol} at ~{limit} (mid={mid})")
    place_sto(ib, opt, qty, limit)

def main_loop():
    load_dotenv()                      # load .env
    cfg = load_cfg()                   # strategy config
    ib = connect_ib()                  # connect (reqMarketDataType(4) inside)
    ib.sleep(1.0)                      # let Gateway settle

    # Diagnostics
    logger.info(f"Connected to IB. Account={os.getenv('IB_ACCOUNT_ID')}")
    logger.info(f"Managed accounts: {ib.managedAccounts()}")

    # Initial trade attempt
    try:
        sell_if_needed(ib, cfg)
    except Exception as e:
        logger.exception(f"sell_if_needed failed: {e}")

    # Periodic risk/management loop
    def manage_job():
        try:
            warns = manage_positions(ib, cfg)
            for w in warns:
                logger.warning(w)
        except Exception as e:
            logger.exception(f"manage_positions failed: {e}")

    sched = BackgroundScheduler()
    sched.add_job(manage_job, "interval", minutes=5)
    sched.start()

    try:
        while True:
            ib.sleep(1.0)
    except KeyboardInterrupt:
        sched.shutdown()
        ib.disconnect()

if __name__ == "__main__":
    main_loop()