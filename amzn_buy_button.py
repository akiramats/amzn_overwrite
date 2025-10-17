#!/usr/bin/env python3
import os
import tkinter as tk
from tkinter import ttk
from ib_insync import IB, Stock, MarketOrder

# ---------- Config from env ----------
HOST = os.getenv('IB_HOST', '127.0.0.1')
PORT = int(os.getenv('IB_PORT', '4002'))
CLIENT_ID = int(os.getenv('IB_CLIENT_ID', '7777'))
ACCOUNT_ID = os.getenv('IB_ACCOUNT', '')  # optional filter

SYMBOL = 'AMZN'
EXCHANGE = 'SMART'
CURRENCY = 'USD'
PRIMARY = 'NASDAQ'
# Keep a small cache from execId -> last fee seen

# ---------- IB + GUI ----------
ib = IB()

# Map IB orderId -> Trade (so we can cancel by selection)
TRADE_BY_ID = {}
LAST_FEES = {}  # {execId: (commission_abs, currency)}


root = tk.Tk()
root.title("AMZN Spot — One-Click Buy")
root.geometry("920x620")

# --- Top controls ---
toolbar = ttk.Frame(root, padding=8)
toolbar.pack(fill='x')

btn_connect = ttk.Button(toolbar, text="Connect")
btn_buy = ttk.Button(toolbar, text="Buy Spot")
btn_cancel_all = ttk.Button(toolbar, text="Cancel All")
btn_cancel_sel = ttk.Button(toolbar, text="Cancel Selected")   # NEW
btn_refresh = ttk.Button(toolbar, text="Refresh")
btn_fetch_fills = ttk.Button(toolbar, text="Fetch Today Fills")  # <-- ADD THIS

# pack them in order
btn_connect.pack(side='left', padx=(0,6))
btn_buy.pack(side='left', padx=6)
btn_cancel_all.pack(side='left', padx=6)
btn_cancel_sel.pack(side='left', padx=6)
btn_refresh.pack(side='left', padx=6)
btn_fetch_fills.pack(side='left', padx=6)   # <-- ADD THIS

acct_label = ttk.Label(toolbar, text="Account: (not connected)")
acct_label.pack(side='right')

def fetch_today_fills():
    try:
        if not ib.isConnected():
            append_log("Not connected.")
            return

        fills = ib.reqExecutions()
        if not fills:
            append_fill("— No executions returned —")
            return

        for exd in fills:  # ExecutionDetail
            ex, c = exd.execution, exd.contract
            cr = getattr(exd, 'commissionReport', None)
            if cr and cr.commission is not None:
                fee_str = f"{abs(float(cr.commission)):.2f} {cr.currency or ''}"
            else:
                fee_str = "n/a"   # some backfills don’t include fees, esp. on paper
            append_fill(
                f"📜 (hist) {ex.side} {ex.shares} {c.symbol} @ {ex.price} "
                f"(orderId={ex.orderId}, time={ex.time}, fee={fee_str})"
            )
    except Exception as e:
        append_log(f"⚠️ fetch_today_fills: {e}")

btn_fetch_fills.config(command=fetch_today_fills)

# --- Positions panel ---
pos_frame = ttk.Labelframe(root, text="Positions", padding=8)
pos_frame.pack(fill='x', padx=8, pady=(4,2))

cols_pos = ("account","symbol","qty","avgCost","last","uPnL")
tree_pos = ttk.Treeview(pos_frame, columns=cols_pos, show="headings", height=5)
for c, h, w in zip(
    cols_pos,
    ("account","symbol","qty","avgCost","last","uPnL"),
    (140,120,100,140,120,120)
):
    tree_pos.heading(c, text=h)
    tree_pos.column(c, width=w, anchor='center')
tree_pos.pack(fill='x')

# --- Open Orders panel ---
oo_frame = ttk.Labelframe(root, text="Open Orders", padding=8)
oo_frame.pack(fill='both', expand=True, padx=8, pady=(2,2))

cols_oo = ("orderId","action","qty","otype","status","filled","avgFill")
tree_oo = ttk.Treeview(oo_frame, columns=cols_oo, show="headings", height=10)
heads = ["OrderID","Side","Qty","Type","Status","Filled","AvgFill"]
for c, h, w in zip(cols_oo, heads, (90,70,70,90,140,90,90)):
    tree_oo.heading(c, text=h)
    tree_oo.column(c, width=w, anchor='center')
tree_oo.pack(fill='both', expand=True)

# Double-click a row to cancel it
tree_oo.bind("<Double-1>", lambda e: None or cancel_selected_order())

# --- Fills panel ---
fills_frame = ttk.Labelframe(root, text="Last Fills", padding=8)
fills_frame.pack(fill='x', padx=8, pady=(2,6))
fills_text = tk.Text(fills_frame, height=6, wrap='word')
fills_text.pack(fill='x')

# --- Log panel ---
log_frame = ttk.Labelframe(root, text="Log", padding=8)
log_frame.pack(fill='both', expand=False, padx=8, pady=(0,6))
log = tk.Text(log_frame, height=6, wrap='word')
log.pack(fill='x')

def append_log(msg: str):
    log.insert('end', msg + "\n")
    log.see('end')

def append_fill(msg: str):
    fills_text.insert('end', msg + "\n")
    fills_text.see('end')

# ---------- Helpers to (re)populate tables ----------
def refresh_positions():
    try:
        # Clear table
        for i in tree_pos.get_children():
            tree_pos.delete(i)

        poss = [p for p in ib.positions() if (not ACCOUNT_ID or p.account == ACCOUNT_ID)]
        if not poss:
            return

        # Request snapshot quotes for each position once
        contracts = [p.contract for p in poss]
        # Qualify any unqualified contracts
        contracts = ib.qualifyContracts(*contracts)
        # Snapshot market data (quick)
        for c in contracts:
            ib.reqMktData(c, "", True, False)
        ib.sleep(0.4)

        # Build rows
        for p in poss:
            c = p.contract
            tk = ib.ticker(c)
            last = tk.marketPrice() or tk.last or 0.0
            # very simple UPL ≈ (last - avgCost) * position
            upl = (last - float(p.avgCost or 0.0)) * float(p.position or 0.0)
            tree_pos.insert(
                '', 'end',
                values=(
                    p.account,
                    getattr(c, 'symbol', ''),
                    float(p.position or 0.0),
                    round(float(p.avgCost or 0.0), 4),
                    round(float(last or 0.0), 4),
                    round(float(upl or 0.0), 2),
                )
            )
    except Exception as e:
        append_log(f"⚠️ positions(): {e}")

def refresh_open_orders():
    """
    Rebuild the open orders table. Each row's iid is the actual orderId so
    selection -> orderId -> TRADE_BY_ID lookup -> cancel works.
    """
    try:
        ib.reqOpenOrders()

        # Clear current rows
        for iid in tree_oo.get_children():
            tree_oo.delete(iid)

        for tr in ib.trades():
            o = tr.order
            st = tr.orderStatus

            # ignore rows that don't have a real orderId
            if getattr(o, "orderId", 0) <= 0:
                continue

            if ACCOUNT_ID and getattr(o, 'account', '') and o.account != ACCOUNT_ID:
                continue

            # show only active-ish orders in the table
            if st.status in ('Filled','Cancelled','Inactive'):
                continue

            TRADE_BY_ID[o.orderId] = tr
            values = (
                o.orderId,
                o.action,
                float(o.totalQuantity or 0),
                o.orderType,
                st.status,
                float(st.filled or 0.0),
                float(st.avgFillPrice or 0.0),
            )
            # use orderId as the iid!
            tree_oo.insert('', 'end', iid=str(o.orderId), values=values)

    except Exception as e:
        append_log(f"⚠️ openOrders: {e}")

def repaint_all():
    refresh_positions()
    refresh_open_orders()

# ---------- IB Event wiring (FIXED signatures) ----------
def on_open_order(trade):
    try:
        o, st = trade.order, trade.orderStatus
        if getattr(o, "orderId", 0) > 0:
            TRADE_BY_ID[o.orderId] = trade
        append_log(f"📝 openOrder id={o.orderId} {o.action} {o.totalQuantity} {o.orderType} | status={st.status}")
    except Exception as e:
        append_log(f"⚠️ on_open_order: {e}")
    refresh_open_orders()

def on_order_status(trade):
    try:
        o, st = trade.order, trade.orderStatus
        if getattr(o, "orderId", 0) > 0:
            TRADE_BY_ID[o.orderId] = trade
        append_log(f"ℹ️ orderStatus id={o.orderId} status={st.status} filled={st.filled} avg={st.avgFillPrice}")
    except Exception as e:
        append_log(f"⚠️ on_order_status: {e}")
    refresh_open_orders()
    refresh_positions()

def on_commission_report(trade, fill, report):
    """
    Called when IB sends the CommissionReport (fees) for a fill.
    CommissionReport fields: execId, commission, currency, realizedPNL, yield, yieldRedemptionDate
    """
    try:
        ex = getattr(fill, 'execution', None)
        exec_id = getattr(report, 'execId', '?')
        order_id = getattr(ex, 'orderId', '?')

        # commission may be negative (debit); show absolute
        comm = getattr(report, 'commission', None)
        curr = getattr(report, 'currency', '') or ''
        if comm is not None:
            fee_abs = abs(float(comm))
            LAST_FEES[exec_id] = (fee_abs, curr)
            append_fill(f"💸 Fee: execId={exec_id} orderId={order_id} fee={fee_abs:.2f} {curr}")
        else:
            append_fill(f"💸 Fee: execId={exec_id} orderId={order_id} fee=n/a")
    except Exception as e:
        append_log(f"⚠️ on_commission_report: {e}")

def on_exec_details(trade, fill):
    try:
        ex = fill.execution
        sym, side, qty, px, exec_id = fill.contract.symbol, ex.side, ex.shares, ex.price, ex.execId

        # show cached fee if commission report arrived first
        fee_str = "pending"
        if exec_id in LAST_FEES:
            fee_abs, curr = LAST_FEES[exec_id]
            fee_str = f"{fee_abs:.2f} {curr}"
        else:
            cr = getattr(fill, 'commissionReport', None)
            if cr and cr.commission is not None:
                fee_abs = abs(float(cr.commission))
                fee_str = f"{fee_abs:.2f} {cr.currency}"
                LAST_FEES[exec_id] = (fee_abs, cr.currency or "")

        append_fill(
            f"✅ Fill: {sym} {side} {qty} @ {px} "
            f"(orderId={ex.orderId}, execId={exec_id}, fee={fee_str})"
        )
    except Exception as e:
        append_log(f"⚠️ on_exec_details: {e}")

    refresh_open_orders()
    refresh_positions()

# Re-wire events (note: just pass the function objects)
ib.openOrderEvent.clear()
ib.orderStatusEvent.clear()
ib.execDetailsEvent.clear()
ib.commissionReportEvent.clear()
ib.openOrderEvent += on_open_order
ib.orderStatusEvent += on_order_status
ib.execDetailsEvent += on_exec_details
ib.commissionReportEvent += on_commission_report

# ---------- Actions ----------
def do_connect():
    try:
        if ib.isConnected():
            append_log("Already connected.")
            return
        ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=8)
        acct_label.config(text=f"Account: {', '.join(ib.managedAccounts()) or '(none)'}")
        append_log(f"✅ Connected {HOST}:{PORT} (clientId={CLIENT_ID})")
        if ACCOUNT_ID:
            append_log(f"✅ Using account filter: {ACCOUNT_ID}")
        repaint_all()
    except Exception as e:
        append_log(f"❌ Connect error: {e}")

def do_buy_spot():
    try:
        if not ib.isConnected():
            append_log("Not connected.")
            return
        contract = Stock(SYMBOL, EXCHANGE, CURRENCY, primaryExchange=PRIMARY)
        contract = ib.qualifyContracts(contract)[0]
        order = MarketOrder('BUY', 1, account=ACCOUNT_ID or None)
        order.outsideRth = True
        trade = ib.placeOrder(contract, order)
        append_log(f"🟢 Sent BUY 1 {SYMBOL} @ MARKET (acct={order.account or 'default'}) | orderId={order.orderId or 'pending'}")
    except Exception as e:
        append_log(f"❌ placeOrder error: {e}")

def do_cancel_all():
    try:
        if not ib.isConnected():
            append_log("Not connected.")
            return
        ib.reqGlobalCancel()
        append_log("⛔ Requested: Cancel ALL orders")
    except Exception as e:
        append_log(f"❌ cancel error: {e}")

def cancel_selected_order():
    """
    Cancel just the highlighted order(s) in the Open Orders table.
    """
    try:
        sel = tree_oo.selection()
        if not sel:
            append_log("ⓘ Select a row first.")
            return

        for iid in sel:
            try:
                order_id = int(iid)
            except ValueError:
                append_log(f"⚠️ Bad row id {iid}")
                continue

            if order_id <= 0:
                append_log(f"⚠️ OrderID {order_id} is not cancellable.")
                continue

            tr = TRADE_BY_ID.get(order_id)
            if not tr:
                append_log(f"⚠️ Order {order_id} not found (may be completed).")
                continue

            ib.cancelOrder(tr.order)
            append_log(f"🛑 Sent cancel for orderId={order_id}")

    except Exception as e:
        append_log(f"❌ cancel_selected_order: {e}")

def do_refresh():
    repaint_all()
    append_log("🔄 Refreshed.")

# ---------- Wire buttons ----------
btn_connect.config(command=do_connect)
btn_buy.config(command=do_buy_spot)
btn_cancel_all.config(command=do_cancel_all)
btn_cancel_sel.config(command=cancel_selected_order)  # NEW
btn_refresh.config(command=do_refresh)

# ---------- Safety net: periodic refresh ----------
def tick():
    if ib.isConnected():
        refresh_open_orders()
    root.after(1500, tick)

root.after(1500, tick)

# ---------- Start ----------
append_log("Tip: set env vars IB_HOST/IB_PORT/IB_CLIENT_ID/IB_ACCOUNT before running.")
append_log(f"Ready. Click 'Connect' then 'Buy Spot'.")
root.mainloop()