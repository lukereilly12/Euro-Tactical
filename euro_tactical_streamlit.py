# coding: utf-8
"""
Euro Tactical — Streamlit Web App
==================================
Mobile-friendly rebalancer. Open in any browser, works on phone.

Deploy free: https://share.streamlit.io
Run locally: streamlit run euro_tactical_streamlit.py
"""

import warnings
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    st.error("yfinance not installed. Check requirements.txt.")
    st.stop()

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
CURRENT_RISK    = ['EQQQ.DE', 'VUSA.AS', '4GLD.DE', 'CL2.PA', 'EMIM.AS']
CURRENT_CASH    = 'EUN6.DE'
EURO_UNIVERSE   = ['EXS1.DE', 'EUNA.DE', '4GLD.DE']
EURO_CASH       = 'EUN6.DE'
BOND_TICKER     = 'EUNA.DE'
VOL_REF_TICKER  = 'EXS1.DE'
DCA_TICKERS     = {'IWDA.AS': 'iShares MSCI World', 'ZPRX.DE': 'SPDR World Small Cap'}

ENTER_WIN           = 205
EXIT_WIN            = 195
MOM_LOOKBACK_DAYS   = 252
VOL_LOOKBACK        = 20
TOP_N               = 2
MIN_LEG_WEIGHT      = 0.25
MAX_LEG_WEIGHT      = 0.75
DEF_VOL_THRESHOLD   = 0.25
DEF_CUT             = 0.50
MAX_DAILY_MOVE      = 0.25

# ─────────────────────────────────────────────
# Core helpers (same logic as live script)
# ─────────────────────────────────────────────
def _download_raw(tickers: List[str], start: str = "2003-01-01") -> pd.DataFrame:
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.dropna(how="all")


def _clean(prices: pd.DataFrame) -> pd.DataFrame:
    cleaned = prices.copy()
    daily_ret = cleaned.pct_change()
    for col in cleaned.columns:
        bad = daily_ret[col].abs() > MAX_DAILY_MOVE
        cleaned.loc[bad, col] = np.nan
        cleaned[col] = cleaned[col].ffill()
    return cleaned


def _ma(series: pd.Series, w: int) -> pd.Series:
    return series.rolling(w, min_periods=w).mean()


def _hyst(price: pd.Series, enter: int, exit_: int) -> pd.Series:
    ma_e, ma_x = _ma(price, enter), _ma(price, exit_)
    sig = pd.Series(index=price.index, dtype=float)
    state = 0.0
    for t in price.index:
        p, e, x = price.loc[t], ma_e.loc[t], ma_x.loc[t]
        if any(np.isnan(v) for v in (p, e, x)):
            sig.loc[t] = state
            continue
        if state == 0 and p > e:
            state = 1.0
        elif state == 1 and p < x:
            state = 0.0
        sig.loc[t] = state
    return sig.ffill().fillna(0.0)


def _month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(idx.to_period("M").to_timestamp("M")).unique().intersection(idx)


def _ensure_cash(prices: pd.DataFrame, idx: pd.DatetimeIndex, cash: str) -> pd.DataFrame:
    px = prices.copy()
    if cash not in px.columns:
        px = px.join(pd.Series(1.0, index=idx, name=cash), how="outer")
    return px


def _snap(idx: pd.DatetimeIndex, d):
    rd = pd.Timestamp(d)
    if rd in idx:
        return rd
    snapped = idx.asof(rd)
    return None if pd.isna(snapped) else snapped


def _cur_weights(prices, reb):
    cols = [t for t in CURRENT_RISK if t in prices.columns]
    px = _ensure_cash(prices[cols].dropna(how="all"), prices.index, CURRENT_CASH)
    sigs = pd.DataFrame({t: _hyst(px[t], ENTER_WIN, EXIT_WIN) for t in cols}).reindex(px.index).fillna(0)
    rd = _snap(sigs.index, reb)
    w = pd.Series(0.0, index=px.columns)
    if rd is None:
        w[CURRENT_CASH] = 1.0
        return w, {}
    on = [a for a in cols if a in sigs.columns and sigs.loc[rd, a] > 0.5]
    sig_state = {a: (sigs.loc[rd, a] > 0.5) for a in cols}
    if not on:
        w[CURRENT_CASH] = 1.0
    else:
        for a in on:
            w[a] = 1.0 / len(on)
    return w, sig_state


def _eur_weights(prices, reb):
    cols = [t for t in EURO_UNIVERSE if t in prices.columns]
    px = _ensure_cash(prices[cols].dropna(how="all"), prices.index, EURO_CASH)
    elig = pd.DataFrame({t: _hyst(px[t], ENTER_WIN, EXIT_WIN) for t in cols}).reindex(px.index).fillna(0)
    mom  = (px / px.shift(MOM_LOOKBACK_DAYS) - 1).reindex(px.index)
    vol  = px.pct_change().rolling(VOL_LOOKBACK).std() * np.sqrt(252)
    rd   = _snap(elig.index, reb)
    w    = pd.Series(0.0, index=px.columns)
    if rd is None:
        w[EURO_CASH] = 1.0
        return w
    ea = [a for a in cols if a in elig.columns and elig.loc[rd, a] > 0.5]
    if not ea:
        w[EURO_CASH] = 1.0
        return w
    m = mom.loc[rd, ea].dropna().sort_values(ascending=False)
    if m.empty:
        w[EURO_CASH] = 1.0
        return w
    picks = m.index[:TOP_N].tolist()
    vols  = vol.loc[rd, picks].replace(0, np.nan).dropna()
    inv   = 1.0 / vols if not vols.empty else pd.Series(1.0 / len(picks), index=picks)
    w_leg = (inv / inv.sum()).clip(MIN_LEG_WEIGHT, MAX_LEG_WEIGHT)
    w_leg /= w_leg.sum()
    for a in w_leg.index:
        w[a] = float(w_leg[a])
    return w


def _bond_weight(prices, reb):
    px = _ensure_cash(prices[[c for c in [BOND_TICKER, EURO_CASH] if c in prices.columns]].dropna(how="all"), prices.index, EURO_CASH)
    w  = pd.Series(0.0, index=px.columns)
    if BOND_TICKER not in px.columns:
        w[EURO_CASH] = 1.0
        return w
    sig = _hyst(px[BOND_TICKER], ENTER_WIN, EXIT_WIN)
    rd  = _snap(sig.index, reb)
    if rd is None:
        w[EURO_CASH] = 1.0
        return w
    w[BOND_TICKER if sig.loc[rd] > 0.5 else EURO_CASH] = 1.0
    return w


def _sleeve_rets(px, start, tickers, cash):
    sub = _ensure_cash(px[[c for c in tickers + [cash] if c in px.columns]].copy(), px.index, cash)
    risk_cols = [c for c in tickers if c in sub.columns]
    sigs = pd.DataFrame({t: _hyst(sub[t], ENTER_WIN, EXIT_WIN) for t in risk_cols}).reindex(sub.index).fillna(0)
    rbd  = _month_ends(sub.index)
    wts  = pd.DataFrame(0.0, index=sub.index, columns=sub.columns)
    prev = pd.Series(0.0, index=sub.columns)
    for dt in rbd:
        if dt < start:
            continue
        on = [a for a in risk_cols if a in sigs.columns and sigs.loc[dt, a] > 0.5]
        nw = pd.Series(0.0, index=sub.columns)
        if not on:
            nw[cash] = 1.0
        else:
            for a in on:
                nw[a] = 1.0 / len(on)
        prev = nw.copy()
        wts.loc[dt] = prev
    wts = wts.replace(0.0, np.nan).ffill().fillna(0.0)
    return (wts * sub.pct_change().fillna(0.0)).sum(axis=1)


def _eur_rets(px, start):
    sub  = _ensure_cash(px[[c for c in EURO_UNIVERSE + [EURO_CASH] if c in px.columns]].copy(), px.index, EURO_CASH)
    cols = [c for c in EURO_UNIVERSE if c in sub.columns]
    elig = pd.DataFrame({t: _hyst(sub[t], ENTER_WIN, EXIT_WIN) for t in cols}).reindex(sub.index).fillna(0)
    mom  = (sub / sub.shift(MOM_LOOKBACK_DAYS) - 1).reindex(sub.index)
    vol  = sub.pct_change().rolling(VOL_LOOKBACK).std() * np.sqrt(252)
    rbd  = _month_ends(sub.index)
    wts  = pd.DataFrame(0.0, index=sub.index, columns=sub.columns)
    for dt in rbd:
        if dt < start:
            continue
        ea = [a for a in cols if a in elig.columns and elig.loc[dt, a] > 0.5]
        w  = pd.Series(0.0, index=sub.columns)
        if not ea:
            w[EURO_CASH] = 1.0
            wts.loc[dt] = w
            continue
        m = mom.loc[dt, ea].dropna().sort_values(ascending=False)
        if m.empty:
            w[EURO_CASH] = 1.0
            wts.loc[dt] = w
            continue
        picks = m.index[:TOP_N].tolist()
        vols  = vol.loc[dt, picks].replace(0, np.nan).dropna()
        inv   = 1.0 / vols if not vols.empty else pd.Series(1.0 / len(picks), index=picks)
        w_leg = (inv / inv.sum()).clip(MIN_LEG_WEIGHT, MAX_LEG_WEIGHT)
        w_leg /= w_leg.sum()
        for a in w_leg.index:
            w[a] = float(w_leg[a])
        wts.loc[dt] = w
    wts = wts.replace(0.0, np.nan).ffill().fillna(0.0)
    return (wts * sub.pct_change().fillna(0.0)).sum(axis=1)


def _bond_rets(px, start):
    sub = _ensure_cash(px[[c for c in [BOND_TICKER, EURO_CASH] if c in px.columns]].copy(), px.index, EURO_CASH)
    if BOND_TICKER not in sub.columns:
        return sub[EURO_CASH].pct_change().fillna(0.0)
    sig = _hyst(sub[BOND_TICKER], ENTER_WIN, EXIT_WIN)
    rbd = _month_ends(sub.index)
    wts = pd.DataFrame(0.0, index=sub.index, columns=sub.columns)
    for dt in rbd:
        if dt < start:
            continue
        w = pd.Series(0.0, index=sub.columns)
        w[BOND_TICKER if sig.loc[dt] > 0.5 else EURO_CASH] = 1.0
        wts.loc[dt] = w
    wts = wts.replace(0.0, np.nan).ffill().fillna(0.0)
    return (wts * sub.pct_change().fillna(0.0)).sum(axis=1)


def _latest_prices(tickers: List[str]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    try:
        px = yf.download(tickers, period="1mo", auto_adjust=True, progress=False)["Close"]
        if isinstance(px, pd.Series):
            px = px.to_frame()
        last = px.ffill().iloc[-1]
        for t in tickers:
            result[t] = float(last.get(t, np.nan))
    except Exception:
        for t in tickers:
            result[t] = np.nan
    missing = [t for t in tickers if np.isnan(result.get(t, np.nan))]
    for t in missing:
        try:
            s = yf.download(t, period="3mo", auto_adjust=True, progress=False)["Close"]
            if not s.empty:
                result[t] = float(s.ffill().iloc[-1])
        except Exception:
            pass
    return result


# ─────────────────────────────────────────────
# Cached signal computation
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def compute_signals():
    all_tickers = list(set(CURRENT_RISK + [CURRENT_CASH] + EURO_UNIVERSE + [EURO_CASH, BOND_TICKER, VOL_REF_TICKER]))
    raw    = _download_raw(all_tickers)
    prices = _clean(raw)

    idx      = prices.index
    me_dates = _month_ends(idx)
    reb_date = me_dates[-1]

    w_cur,  sig_state = _cur_weights(prices, reb_date)
    w_eur              = _eur_weights(prices, reb_date)
    w_bond             = _bond_weight(prices, reb_date)

    # Vol parity blend
    rets      = prices.pct_change().dropna()
    start_win = rets.index.max() - pd.Timedelta(days=500)
    rets_win  = rets.loc[rets.index >= (reb_date - pd.Timedelta(days=400))]

    r_cur  = _sleeve_rets(prices, start_win, CURRENT_RISK, CURRENT_CASH).reindex(rets_win.index).fillna(0)
    r_eur  = _eur_rets(prices, start_win).reindex(rets_win.index).fillna(0)

    cur_vol = r_cur.rolling(VOL_LOOKBACK).std().iloc[-1] * np.sqrt(252) if len(r_cur) > VOL_LOOKBACK else np.nan
    eur_vol = r_eur.rolling(VOL_LOOKBACK).std().iloc[-1] * np.sqrt(252) if len(r_eur) > VOL_LOOKBACK else np.nan

    if any(v is None or np.isnan(v) or v <= 0 for v in (cur_vol, eur_vol)):
        w_blend_cur, w_blend_eur = 0.5, 0.5
    else:
        inv_c = 1.0 / cur_vol
        inv_e = 1.0 / eur_vol
        w_blend_cur = float(np.clip(inv_c / (inv_c + inv_e), MIN_LEG_WEIGHT, MAX_LEG_WEIGHT))
        w_blend_eur = 1.0 - w_blend_cur

    # Defensive overlay
    vol_ann, overlay_cut = np.nan, False
    if VOL_REF_TICKER in prices.columns:
        vr      = prices[VOL_REF_TICKER].pct_change()
        vol_ann = vr.rolling(VOL_LOOKBACK).std().iloc[-1] * np.sqrt(252) if len(vr) > VOL_LOOKBACK else 0.0
        overlay_cut = vol_ann > DEF_VOL_THRESHOLD

    eq_mult   = (1.0 - DEF_CUT) if overlay_cut else 1.0
    bond_mult = 1.0 - eq_mult

    w_eq   = (w_cur * w_blend_cur).add(w_eur * w_blend_eur, fill_value=0.0)
    w_unl  = (w_eq * eq_mult).add(w_bond * bond_mult, fill_value=0.0)
    w_unl  = w_unl[w_unl.index.notnull()].fillna(0.0)
    if w_unl.sum() > 0:
        w_unl /= w_unl.sum()

    return {
        "reb_date":    reb_date,
        "w_unlevered": w_unl,
        "w_blend_cur": w_blend_cur,
        "w_blend_eur": w_blend_eur,
        "vol_ann":     vol_ann,
        "overlay_cut": overlay_cut,
        "w_bond":      w_bond,
        "sig_state":   sig_state,          # {ticker: True/False}
        "bond_on":     bool(w_bond.get(BOND_TICKER, 0) > 0.5),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def compute_dca_signals():
    tickers = list(DCA_TICKERS.keys())
    results = {}
    try:
        raw = yf.download(tickers, period="260d", auto_adjust=True, progress=False)["Close"]
        if isinstance(raw, pd.Series):
            raw = raw.to_frame()
        raw = raw.ffill().dropna(how="all")
    except Exception:
        return results
    for ticker in tickers:
        if ticker not in raw.columns:
            continue
        series = raw[ticker].dropna()
        if len(series) < 200:
            continue
        ma200  = series.rolling(200).mean().iloc[-1]
        price  = series.iloc[-1]
        pct    = (price / ma200 - 1) * 100
        as_of  = series.index[-1].date()
        results[ticker] = {
            "name":     DCA_TICKERS[ticker],
            "price":    price,
            "ma200":    ma200,
            "pct":      pct,
            "above":    price > ma200,
            "as_of":    as_of,
        }
    return results


# ─────────────────────────────────────────────
# Page layout
# ─────────────────────────────────────────────
st.set_page_config(page_title="Euro Tactical", page_icon="📊", layout="centered")
st.title("📊 Euro Tactical Rebalancer")

# ── Load signals ──────────────────────────────
with st.spinner("Downloading prices & computing signals… (cached after first load)"):
    try:
        data = compute_signals()
    except Exception as e:
        st.error(f"Failed to compute signals: {e}")
        st.stop()

w_unl        = data["w_unlevered"]
reb_date     = data["reb_date"]
w_blend_cur  = data["w_blend_cur"]
w_blend_eur  = data["w_blend_eur"]
vol_ann      = data["vol_ann"]
overlay_cut  = data["overlay_cut"]
sig_state    = data["sig_state"]
bond_on      = data["bond_on"]

st.caption(f"Signal date: **{pd.Timestamp(reb_date).strftime('%d %b %Y')}** · prices cached 1 hr")

# ── Signal summary ────────────────────────────
with st.expander("📡 Current Signals", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current blend", f"{w_blend_cur:.0%}", help="Weight given to Current sleeve (vol-parity)")
        st.metric("Euro Rot blend", f"{w_blend_eur:.0%}")
    with col2:
        def_label = "🔴 CUT 50%" if overlay_cut else "🟢 No cut"
        vol_str   = f"{vol_ann:.1%}" if not np.isnan(vol_ann) else "n/a"
        st.metric("Defensive overlay", def_label, delta=f"EXS1 vol {vol_str}", delta_color="off")
        st.metric("Bond Trend", "🟢 Long Bonds" if bond_on else "🔴 Cash")

    st.markdown("**Current sleeve signals**")
    for t in CURRENT_RISK:
        on = sig_state.get(t, False)
        st.markdown(f"{'🟢' if on else '⚪'} `{t}` — {'**ON**' if on else 'off'}")

    st.markdown("**Target weights**")
    active = w_unl[w_unl > 1e-6].sort_values(ascending=False)
    for t, w in active.items():
        bar = "█" * int(w * 20)
        st.markdown(f"`{t:<10}` {w:5.1%}  {bar}")

# ── Holdings input ────────────────────────────
st.divider()
st.subheader("Your Holdings")

needed = sorted(w_unl[w_unl > 1e-6].index.tolist())

with st.form("holdings_form"):
    total_value = st.number_input(
        "Total portfolio value (€)", min_value=0.0, value=10000.0, step=1000.0, format="%.2f"
    )
    mode = st.radio("Enter holdings as", ["Values (€)", "Shares"], horizontal=True)
    min_trade = st.number_input("Ignore trades smaller than (€)", min_value=0.0, value=100.0, step=50.0)

    st.markdown("**Current positions** — enter 0 if none")
    holdings_input: Dict[str, float] = {}
    for t in needed:
        label = t if mode == "Values (€)" else f"{t}"
        holdings_input[t] = st.number_input(label, min_value=0.0, value=0.0, step=1.0, format="%.4f", key=f"hold_{t}")

    submitted = st.form_submit_button("🧮 Generate Order Ticket", use_container_width=True, type="primary")

# ── Order ticket ──────────────────────────────
if submitted:
    if mode == "Shares":
        with st.spinner("Fetching live prices…"):
            live_px = _latest_prices(needed)
    else:
        live_px = {}

    current_values: Dict[str, float] = {}
    for t in needed:
        raw_val = holdings_input[t]
        if mode == "Values (€)":
            current_values[t] = raw_val
        else:
            px = live_px.get(t, np.nan)
            if px and not np.isnan(px) and px > 0:
                current_values[t] = raw_val * px
            else:
                st.warning(f"No live price for {t} — treating as €0. Check the ticker.")
                current_values[t] = 0.0

    targets = {t: float(w_unl.get(t, 0.0)) * total_value for t in needed}
    diffs   = {t: targets[t] - current_values.get(t, 0.0) for t in needed}

    buys  = [(t, diffs[t]) for t in needed if diffs[t] >= min_trade]
    sells = [(t, diffs[t]) for t in needed if diffs[t] <= -min_trade]

    st.divider()
    st.subheader("📋 Order Ticket")

    if buys:
        st.markdown("**BUY**")
        for t, dv in sorted(buys, key=lambda x: -x[1]):
            col_a, col_b = st.columns([2, 1])
            col_a.markdown(f"🟢 `{t}`")
            col_b.markdown(f"**+€{dv:,.0f}**")
            if mode == "Shares":
                px = live_px.get(t, np.nan)
                if px and not np.isnan(px):
                    col_b.caption(f"~{dv/px:.2f} shares @ {px:.2f}")

    if sells:
        st.markdown("**SELL**")
        for t, dv in sorted(sells, key=lambda x: x[1]):
            col_a, col_b = st.columns([2, 1])
            col_a.markdown(f"🔴 `{t}`")
            col_b.markdown(f"**€{dv:,.0f}**")
            if mode == "Shares":
                px = live_px.get(t, np.nan)
                if px and not np.isnan(px):
                    col_b.caption(f"~{dv/px:.2f} shares @ {px:.2f}")

    if not buys and not sells:
        st.success("No trades required — portfolio is within tolerance.")

    net = sum(diffs[t] for t in needed)
    st.metric("Net cash required", f"€{net:,.0f}",
              help="Positive = add cash. Negative = cash freed by sells.")

    with st.expander("Current vs Target detail"):
        rows = []
        for t in needed:
            rows.append({
                "Ticker": t,
                "Current (€)": f"{current_values.get(t,0):,.0f}",
                "Target (€)":  f"{targets.get(t,0):,.0f}",
                "Diff (€)":    f"{diffs.get(t,0):+,.0f}",
                "Target %":    f"{w_unl.get(t,0):.1%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── DCA buy-zone ──────────────────────────────
st.divider()
st.subheader("📈 DCA Buy-Zone Check")
st.caption("IWDA.AS & ZPRX.DE — only top up when price is above 200-day MA")

with st.spinner("Checking 200-day MA…"):
    dca = compute_dca_signals()

if not dca:
    st.warning("Could not load DCA price data.")
else:
    for ticker, info in dca.items():
        sign  = "+" if info["pct"] >= 0 else ""
        above = info["above"]
        color = "🟢" if above else "🔴"
        label = "BUY ZONE — add to position" if above else "BELOW MA — hold cash (EUN6.DE)"

        with st.container(border=True):
            st.markdown(f"{color} **{ticker}** — {info['name']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Price", f"{info['price']:.2f}", help=f"As of {info['as_of']}")
            c2.metric("200d MA", f"{info['ma200']:.2f}")
            c3.metric("vs MA", f"{sign}{info['pct']:.1f}%", delta_color="normal" if above else "inverse")
            st.info(label) if above else st.warning(label)

st.caption("Prices cached 1 hour. Refresh the page to force an update.")
