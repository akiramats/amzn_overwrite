import pandas as pd
import yfinance as yf

def find_two_week_jumps(
    ticker: str,
    start: str = "2010-01-01",
    end: str | None = None,
    pct: float = 0.15,
    fwd_days: int = 10,
):
    """
    Find dates where the *maximum* close within the next `fwd_days`
    is >= (1 + pct) * today's close.

    Returns:
      all_rows: DataFrame with Close, future Close_t+1..t+N, max_fwd_ret, hit flag
      jumps:    Filtered rows where the condition is True
    """

    # yfinance auto-adjusts by default now; that's fine for % moves
    df = yf.download(ticker, start=start, end=end, progress=False)[["Close"]].dropna()

    # Force a **Series**, even if you slice with [["Close"]]
    px = df["Close"].copy()
    if isinstance(px, pd.DataFrame):                   # extra guard, just in case
        px = px.squeeze("columns")
    px = pd.Series(px.values, index=px.index, name="Close")

    # Build forward close columns as aligned Series
    fwd_mat = pd.concat(
        {f"Close_t+{i}": px.shift(-i) for i in range(1, fwd_days + 1)},
        axis=1,
    )

    # Max forward return within horizon
    max_fwd_px = fwd_mat.max(axis=1)
    max_fwd_ret = (max_fwd_px / px) - 1.0

    # Assemble output (all are index-aligned Series/DataFrames)
    hit_col = f"hit_+{int(pct*100)}%_within_{fwd_days}d"
    all_rows = pd.concat(
        [px, fwd_mat, max_fwd_ret.rename(f"max_fwd_ret_{fwd_days}d")],
        axis=1,
    )
    all_rows[hit_col] = max_fwd_ret.ge(pct)

    jumps = all_rows[all_rows[hit_col]].copy()
    return all_rows, jumps


if __name__ == "__main__":
    all_rows, jump_df = find_two_week_jumps(
        "AMZN", start="2010-01-01", end=None, pct=0.15, fwd_days=10
    )
    print(f"Total rows: {len(all_rows):,} | Jumps found: {len(jump_df):,}")
    # Show a few examples
    cols_to_show = ["Close", "max_fwd_ret_10d"]
    print(jump_df[cols_to_show].head(12))

    # 🔍 Check for jumps in the post-2020 period
    recent = jump_df.loc["2023-01-01":]
    print(f"\nPost-2020 jumps found: {len(recent)}")
    if not recent.empty:
        print(recent.tail(10))
    else:
        # inspect top few biggest forward moves anyway
        tmp = all_rows.copy()
        tmp["max_fwd_ret_10d"] = (tmp["max_fwd_ret_10d"] * 100).round(2)
        print("\nNo +15% 10-day jumps since 2023 — here are the largest instead:")
        print(tmp.sort_values("max_fwd_ret_10d", ascending=False).head(10))