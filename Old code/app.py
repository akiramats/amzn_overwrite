# ---- Streamlit worker thread may not have a loop yet ----
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
# ---------------------------------------------------------

from ib_insync import IB, Stock, Option, util
util.patchAsyncio()  # safe to call more than once

import os, pandas as pd
import streamlit as st
from src.analysis.analyze_chain import analyze_amzn
from ib_insync import util
util.patchAsyncio()  # compatible with Streamlit/notebooks


st.set_page_config(page_title="AMZN Overwrite Scanner", layout="wide")
st.title("AMZN Covered Call – Paper/Delayed Scanner (IBKR)")

min_dte = st.sidebar.slider("Min DTE", 1, 30, 7)
max_dte = st.sidebar.slider("Max DTE", min_dte, 45, 21)
max_rows = st.sidebar.slider("Max rows", 10, 100, 40)

with st.spinner("Scanning AMZN delayed options…"):
    rows = analyze_amzn((min_dte, max_dte), max_rows=max_rows)

st.info(f"IB host={os.getenv('IB_HOST','127.0.0.1')} port={os.getenv('IB_PORT','4002')} DRY_RUN={os.getenv('DRY_RUN','1')}")

try:
    rows = analyze_amzn((min_dte, max_dte), max_rows=max_rows)
    if not rows:
        st.warning("No rows returned (check Gateway Paper is green and API settings).")
    else:
        df = pd.DataFrame([{
            "Expiry": r.expiry,
            "DTE": r.dte,
            "Strike": r.strike,
            "Bid": r.bid,
            "Ask": r.ask,
            "Mid": r.mid,
            "Delta": r.delta,
            "IV": r.iv,
            "Premium%": r.premium_perc,
            "AnnYield%": r.ann_yield_simple*100.0,
            "ProbCalled%": None if r.prob_called is None else r.prob_called*100.0,
            "Breakeven": r.breakeven
        } for r in rows])
        st.dataframe(df, use_container_width=True)

        st.subheader("Quick filters")
        col1, col2, col3 = st.columns(3)
        with col1:
            min_ann = st.number_input("Min AnnYield% (simple)", value=10.0)
        with col2:
            max_prob = st.number_input("Max ProbCalled%", value=35.0)
        with col3:
            min_dte_f = st.number_input("Min DTE (filter)", value=float(min_dte))

        f = df[
            (df["AnnYield%"] >= min_ann) &
            (df["ProbCalled%"].fillna(999) <= max_prob) &
            (df["DTE"] >= min_dte_f)
        ]
        st.markdown("### Candidates")
        st.dataframe(f.sort_values(["AnnYield%","DTE"], ascending=[False, True]), use_container_width=True)

except Exception as e:
    st.exception(e)