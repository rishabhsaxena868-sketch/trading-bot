# nifty_options_bot_pro_ready.py
"""
Institutional-Grade NIFTY / BANKNIFTY Options Bot
- Paper & LIVE trading
- Auto ATM/OTM CE/PE selection
- Trailing SL / ATR target
- Candlestick patterns + SMA/RSI/MACD signals
- Dashboard always shows valid LTP & option preview
"""

import os
import json
import math
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date, timedelta, time
import pytz
from streamlit_autorefresh import st_autorefresh

try:
    from kiteconnect import KiteConnect
except ImportError:
    KiteConnect = None
# ------------------- Token Save/Load -------------------
TOKEN_FILE = "token.json"

def save_token_file(token_data):
    try:
        with open(TOKEN_FILE, "w") as f:
            json.dump(token_data, f)
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not save token: {e}")

def load_token_file():
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}
# Optional autorefresh (kept exactly as you had)
try:
    from streamlit_autorefresh import st_autorefresh
    count = st_autorefresh(interval=600_000, limit=None, key="news_refresh")  # refresh every 10 min
except Exception:
    pass

# App config
st.set_page_config(page_title="NSE Intraday Scanner — Merged Bot (Paper + Live)", layout="wide")

# Market session detection (India)
market_open = time(9, 15)
market_close = time(15, 30)
now = datetime.now(pytz.timezone("Asia/Kolkata")).time()

# Detect mode & market status
is_live = 'kite' in st.session_state and st.session_state.kite
is_market_open = market_open <= now <= market_close

if is_live:
    if is_market_open:
        st.success("✅ LIVE MODE — Market Open — Trading in real time")
    else:
        st.warning("✅ LIVE MODE — Market Closed — Orders will be AMO")
else:
    if is_market_open:
        st.info("📝 PAPER MODE — Market Open — Simulated orders")
    else:
        st.info("📝 PAPER MODE — Market Closed — Simulated AMO")


# ------------------- News & Global Trend -------------------
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Nifty Options Bot PRO", layout="wide")

# ------------------- Constants -------------------
UNDERLYINGS = {
    "NIFTY": {"ticker": "^NSEI", "nfo_prefix": "NIFTY", "lot": 75},
    "BANKNIFTY": {"ticker": "^NSEBANK", "nfo_prefix": "BANKNIFTY", "lot": 25}
}
ROUND_TO = {"NIFTY": 50, "BANKNIFTY": 100}

# ------------------- Indicators -------------------
def sma(series, window): return series.rolling(window).mean()
def rsi(series, period=14):
    delta=series.diff()
    up=delta.clip(lower=0)
    down=-1*delta.clip(upper=0)
    ma_up=up.ewm(com=period-1,adjust=False).mean()
    ma_down=down.ewm(com=period-1,adjust=False).mean()
    rs=ma_up/ma_down
    return 100-(100/(1+rs))
def macd(series,n_fast=12,n_slow=26,n_signal=9):
    ema_fast=series.ewm(span=n_fast,adjust=False).mean()
    ema_slow=series.ewm(span=n_slow,adjust=False).mean()
    macd_line=ema_fast-ema_slow
    signal_line=macd_line.ewm(span=n_signal,adjust=False).mean()
    hist=macd_line-signal_line
    return macd_line,signal_line,hist
def atr(df,period=14):
    df['H-L']=df['High']-df['Low']
    df['H-C']=abs(df['High']-df['Close'].shift())
    df['L-C']=abs(df['Low']-df['Close'].shift())
    tr=df[['H-L','H-C','L-C']].max(axis=1)
    return tr.rolling(period).mean().iloc[-1]
def supertrend(df, period=10, multiplier=3):
    hl2 = (df['High'] + df['Low']) / 2
    atr_val = atr(df, period)
    upperband = hl2 + multiplier*atr_val
    lowerband = hl2 - multiplier*atr_val
    trend = []
    for i in range(len(df)):
        if i==0: trend.append(True)
        else:
            if df['Close'].iloc[i] > upperband.iloc[i-1]: trend.append(True)
            elif df['Close'].iloc[i] < lowerband.iloc[i-1]: trend.append(False)
            else: trend.append(trend[-1])
    return trend

def adx(df, period=14):
    df['TR'] = df['High'] - df['Low']
    df['+DM'] = df['High'].diff()
    df['-DM'] = df['Low'].diff().abs()
    return df['TR'].rolling(period).mean().iloc[-1]

def vwap(df):
    return (df['Close']*df['Volume']).cumsum()/df['Volume'].cumsum()

# ------------------- OHLC Fetch -------------------
@st.cache_data(ttl=60)
def fetch_ohlc(ticker, period="30d", interval="15m"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        if df.empty:
            return pd.DataFrame()
        df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df = df.rename(columns={"Adj Close": "Close"})
        return df
    except:
        return pd.DataFrame()

def fetch_ohlc_safe(ticker, period="30d", interval="15m"):
    df = fetch_ohlc(ticker, period, interval)
    if df.empty:
        df = fetch_ohlc(ticker, period="60d", interval="1h")
    return df

# ------------------- LTP -------------------
def get_underlying_ltp(kite, symbol, fallback_df=None):
    try:
        if kite:
            inst = f"NSE:{symbol}"
            ltp_data = kite.ltp(inst)
            return float(ltp_data[inst]["last_price"])
    except:
        pass
    if fallback_df is not None and not fallback_df.empty:
        return float(fallback_df["Close"].iloc[-1])
    return None

# ------------------- Option Helpers -------------------
def nearest_strike(price, step):
    return int(round(price / step) * step)

def build_nfo_symbol(underlying, expiry_date, strike, opt_type):
    typ = "CE" if opt_type.upper().startswith("C") else "PE"
    y = expiry_date.strftime("%d%b%Y").upper()
    return f"{underlying}{y}{strike}{typ}"

# ------------------- Global Trend & News -------------------
def get_global_trend():
    indices = {"S&P500":"^GSPC", "Dow":"^DJI", "Nasdaq":"^IXIC", "DAX":"^GDAXI", "Nikkei":"^N225"}
    bullish = bearish = 0
    for name, ticker in indices.items():
        try:
            df = yf.download(ticker, period="2d", interval="1d", progress=False)
            if len(df) < 2:
                continue
            if df["Close"].iloc[-1] > df["Close"].iloc[-2]:
                bullish += 1
            else:
                bearish += 1
        except:
            continue
    return "BULL" if bullish >= bearish else "BEAR"

def fetch_news_sentiment(query="NIFTY OR BANKNIFTY", max_news=5):
    api_key = st.secrets.get("NEWSAPI_KEY", "")  # ✅ now reads from secrets
    if not api_key:
        return 0
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&pageSize={max_news}&apiKey={api_key}"
    try:
        response = requests.get(url).json()
        articles = response.get("articles", [])
        if not articles:
            return 0
        scores = []
        for art in articles:
            title = art.get("title", "")
            score = analyzer.polarity_scores(title)["compound"]
            scores.append(score)
        return sum(scores) / len(scores) if scores else 0
    except:
        return 0

# ------------------- Candlestick -------------------
def detect_candlestick_patterns(df):
    patterns = {}
    if df.empty or len(df) < 2:
        return patterns
    last, prev = df.iloc[-1], df.iloc[-2]
    if prev["Close"] < prev["Open"] and last["Close"] > last["Open"] and last["Close"] > prev["Open"] and last["Open"] < prev["Close"]:
        patterns["Bullish Engulfing"] = True
    if prev["Close"] > prev["Open"] and last["Close"] < last["Open"] and last["Open"] > prev["Close"] and last["Close"] < prev["Open"]:
        patterns["Bearish Engulfing"] = True
    return patterns

# ------------------- Signal -------------------
def underlying_signal(df):
    if df.empty or len(df) < 10:
        return "NEUTRAL", {}

    s20, s50 = sma(df["Close"], 20), sma(df["Close"], 50)
    macd_line, macd_sig, macd_hist = macd(df["Close"])
    rsi14 = rsi(df["Close"], 14)

    bull_tech = (s20.iloc[-2] <= s50.iloc[-2] and s20.iloc[-1] > s50.iloc[-1] and macd_hist.iloc[-1] > 0)
    bear_tech = (s20.iloc[-2] >= s50.iloc[-2] and s20.iloc[-1] < s50.iloc[-1] and macd_hist.iloc[-1] < 0)

    global_trend = get_global_trend()
    bull_global = global_trend == "BULL"
    bear_global = global_trend == "BEAR"

    news_score = fetch_news_sentiment()
    bull_news = news_score > 0.1
    bear_news = news_score < -0.1

    bull_count = sum([bull_tech, bull_global, bull_news])
    bear_count = sum([bear_tech, bear_global, bear_news])

    if bull_count > bear_count:
        signal = "BULL"
    elif bear_count > bull_count:
        signal = "BEAR"
    else:
        signal = "NEUTRAL"

    meta = {
        "rsi": rsi14.iloc[-1],
        "macd_hist": macd_hist.iloc[-1],
        "news_score": news_score,
        "global_trend": global_trend,
    }
    return signal, meta

# ------------------- Position Sizing -------------------
def size_qty_by_capital(capital, price, allocation_pct, underlying_choice):
    if price <= 0:
        return 0
    alloc = capital * allocation_pct
    qty = int(math.floor(alloc / price / UNDERLYINGS[underlying_choice]["lot"]))
    return max(qty, 1)

# ------------------- Orders -------------------
def place_real_order(kite, tradingsymbol, tx_type, qty, product="MIS"):
    try:
        return kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NFO,
            tradingsymbol=tradingsymbol,
            transaction_type=tx_type,
            quantity=int(qty * UNDERLYINGS[underlying_choice]["lot"]),
            order_type=kite.ORDER_TYPE_MARKET,
            product=getattr(kite, f"PRODUCT_{product}", "MIS"),
        )
    except:
        return None

# ------------------- Trailing SL -------------------
def monitor_trailing_sl(kite):
    if "open_positions" not in st.session_state:
        return
    for sym, trade in list(st.session_state["open_positions"].items()):
        ltp = get_underlying_ltp(kite, sym)
        if ltp and (ltp <= trade.get("trail_sl", 0) or ltp >= trade.get("target", 0)):
            st.info(f"Closing {sym} at LTP={ltp:.2f}")
            st.session_state["open_positions"].pop(sym)


# ------------------- Streamlit UI -------------------
st.title("⚡ Nifty/BANKNIFTY Options Bot PRO")
st.caption("Paper-mode by default. Test carefully before enabling LIVE trading.")
with st.sidebar:
    st.header("⚙️ Trade Settings")
    sl_pct = st.slider("Stop Loss %", 2, 20, 10) / 100      # default 10%
    tgt_pct = st.slider("Target %", 2, 50, 20) / 100        # default 20%
    trail_pct = st.slider("Trailing SL %", 1, 10, 1) / 100  # default 1%

with st.sidebar:
    st.header("Settings")
    underlying_choice = st.selectbox("Underlying", list(UNDERLYINGS.keys()))
    option_type = st.selectbox("Option Type", ["CE (Call)", "PE (Put)"])
    allocation_pct = st.slider("Capital per trade (%)", 1, 100, 50) / 100.0
    starting_cap = st.number_input("Starting Capital (₹)", 20000, 2000000, 20000, 1000)
    st.session_state['starting_capital'] = starting_cap
    live_trading = st.checkbox("Enable LIVE trading", False)
    
with st.sidebar:
    st.header("🌍 Global Indices")

    try:
        sgx = yf.Ticker("^NSEI").history(period="1d")["Close"].iloc[-1]  # SGX Nifty proxy
        dow = yf.Ticker("^DJI").history(period="1d")["Close"].iloc[-1]
        vix = yf.Ticker("^VIX").history(period="1d")["Close"].iloc[-1]

        st.metric("SGX Nifty", f"{sgx:.2f}")
        st.metric("Dow Futures", f"{dow:.2f}")
        st.metric("India VIX", f"{vix:.2f}")
    except Exception as e:
        st.warning(f"⚠️ Could not fetch global indices: {e}")


st.markdown("---")
st.subheader("Kite API")

# --- Load from Streamlit Secrets ---
API_KEY     = st.secrets.get("API_KEY", "")
API_SECRET  = st.secrets.get("API_SECRET", "")

# Try to load saved token from file (if any)
token_data = load_token_file()
kite = None
access_token_to_use = token_data.get("access_token") if token_data.get("date") == date.today().isoformat() else None

# Auto-connect if valid token exists
if access_token_to_use:
    try:
        kite = KiteConnect(api_key=API_KEY)
        kite.set_access_token(access_token_to_use)
        st.session_state.kite = kite
        st.success("✅ Auto-connected using saved Access Token!")
    except Exception as e:
        st.warning(f"Saved token invalid/expired: {e}")
        access_token_to_use = None

# Request Token input (only if auto-login fails)
request_token = st.text_input("Request Token (only if needed)", type="password")

# Manual refresh if needed
if st.button("Create Access Token") or not access_token_to_use:
    if request_token:
        try:
            kite = KiteConnect(api_key=API_KEY)
            data = kite.generate_session(request_token, api_secret=API_SECRET)
            kite.set_access_token(data["access_token"])
            st.session_state.kite = kite

            # Save token to file
            save_token_file({
                "access_token": data["access_token"],
                "date": date.today().isoformat(),
            })

            st.success("✅ Zerodha connected! New token generated and saved.")
            st.info("⚠️ Token saved locally and auto-used until midnight.")
        except Exception as e:
            st.error(f"Login failed: {e}")
    else:
        st.warning("Please enter the Request Token to generate Access Token.")

# Status
if "kite" in st.session_state:
    st.success("Connected to Zerodha ✅")
else:
    st.warning("Not connected yet.")




# ------------------- Tabs (mobile safe) -------------------
try:
    tab1, tab2, tab3 = st.tabs(["📊 Signals", "📑 Order Preview", "📂 Positions"])

    # ------------------- Tab 1: Signals -------------------
    with tab1:
        interval = st.selectbox("Chart Interval", ["5m", "15m", "30m"], index=1)
        lookback = st.selectbox("History", ["7d", "14d", "30d"], index=1)

        with st.spinner("Fetching data..."):
            try:
                df = fetch_ohlc_safe(
                    UNDERLYINGS[underlying_choice]["ticker"],
                    period=lookback,
                    interval=interval,
                )
            except Exception as e:
                st.error(f"⚠️ Error fetching OHLC: {e}")
                df = pd.DataFrame()

        st.metric("Data rows", len(df))

        if not df.empty:
            with st.expander("Show last 8 rows"):
                st.dataframe(df.tail(8))

            try:
                signal, meta = underlying_signal(df)
                st.subheader(f"Signal: {signal}")
                st.json(meta)
                st.markdown(f"**Global Trend:** {meta.get('global_trend', 'N/A')}")
                st.markdown(f"**News Sentiment Score:** {meta.get('news_score', 0):.2f}")

                patterns = detect_candlestick_patterns(df)
                if patterns:
                    with st.expander("Candlestick Patterns"):
                        st.write(patterns)
            except Exception as e:
                st.warning(f"⚠️ Could not calculate signals: {e}")
        else:
            st.info("No data available for signals")

    # ------------------- Tab 2: Order Preview -------------------
    with tab2:
      st.subheader("Order Preview")
    try:
        ltp = get_underlying_ltp(
            kite,
            UNDERLYINGS[underlying_choice]["ticker"],
            fallback_df=df if not df.empty else None,
        )
    except Exception:
        ltp = None

    if ltp is None and not df.empty:
        ltp = df["Close"].iloc[-1]

    st.metric("Underlying LTP", f"{ltp:.2f}" if ltp else "N/A")

    if ltp:
        try:
            expiry_input = date.today() + timedelta(days=7)
            strike = nearest_strike(ltp, ROUND_TO[underlying_choice])
            opt_side = "CE" if option_type.startswith("CE") else "PE"
            nfo_symbol = build_nfo_symbol(
                UNDERLYINGS[underlying_choice]["nfo_prefix"],
                expiry_input,
                strike,
                opt_side,
            )
            approx_premium = max(1.0, ltp * 0.01)
            qty_preview = size_qty_by_capital(
                starting_cap, approx_premium, allocation_pct, underlying_choice
            )

            # --- Apply SL/Target/Trailing SL ---
            stop_loss_price = round(approx_premium * (1 - sl_pct), 1)
            target_price    = round(approx_premium * (1 + tgt_pct), 1)
            trail_sl_price  = round(approx_premium * (1 - trail_pct), 1)

            st.write("Candidate Option Symbol:", nfo_symbol)
            st.write(f"Approx premium: ₹{approx_premium:.2f}, Qty preview: {qty_preview}")
            st.write(f"Stop Loss: ₹{stop_loss_price:.2f}")
            st.write(f"Target: ₹{target_price:.2f}")
            st.write(f"Trailing SL (initial): ₹{trail_sl_price:.2f}")

# ------------------- Option Greeks -------------------
try:
    from math import log, sqrt, exp
    from scipy.stats import norm

    def bs_greeks(S, K, T, r, sigma, option_type="C"):
        """Black-Scholes Greeks calculator"""
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        if option_type == "C":  # Call
            delta = norm.cdf(d1)
        else:  # Put
            delta = -norm.cdf(-d1)

        gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
        theta = -(S * norm.pdf(d1) * sigma) / (2 * sqrt(T)) - \
                (r * K * exp(-r * T) * (norm.cdf(d2) if option_type == "C" else norm.cdf(-d2)))
        vega = S * norm.pdf(d1) * sqrt(T)
        rho = K * T * exp(-r * T) * (norm.cdf(d2) if option_type == "C" else -norm.cdf(-d2))

        return delta, gamma, theta, vega, rho

    # Approx Greeks calculation
    S = ltp
    K = strike
    T = max((expiry_input - date.today()).days / 365, 1/365)  # avoid div 0
    r = 0.06   # 6% risk free rate
    sigma = 0.2  # assume 20% IV (could replace with NSE IV API later)

    opt_type_flag = "C" if opt_side == "CE" else "P"
    delta, gamma, theta, vega, rho = bs_greeks(S, K, T, r, sigma, opt_type_flag)

    st.markdown("### Option Greeks")
    st.write(f"Delta: {delta:.4f}")
    st.write(f"Gamma: {gamma:.4f}")
    st.write(f"Theta: {theta:.4f}")
    st.write(f"Vega: {vega:.4f}")
    st.write(f"Rho: {rho:.4f}")

except Exception as e:
    st.warning(f"⚠️ Could not calculate Greeks: {e}")


        except Exception as e:
            st.warning(f"⚠️ Could not build order preview: {e}")

    st.markdown("---")
    col_exec1, col_exec2 = st.columns(2)

    with col_exec1:
        if st.button("Scan & Place (paper/live)"):
            try:
                if signal == "BULL":
                    opt_type, tx_type = "CE", "BUY"
                elif signal == "BEAR":
                    opt_type, tx_type = "PE", "BUY"
                else:
                    st.info("Signal NEUTRAL")
                    tx_type = None

                if tx_type and ltp:
                    strike = nearest_strike(ltp, ROUND_TO[underlying_choice])
                    nfo_symbol = build_nfo_symbol(
                        UNDERLYINGS[underlying_choice]["nfo_prefix"],
                        expiry_input,
                        strike,
                        opt_type,
                    )
                    qty = size_qty_by_capital(
                        starting_cap, approx_premium, allocation_pct, underlying_choice
                    )

                    if live_trading and kite:
                        order_id = place_real_order(kite, nfo_symbol, tx_type, qty)
                        if order_id:
                            st.success(f"LIVE order placed. ID:{order_id}")
                    else:
                        trade_id = f"PAPER-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        st.session_state.setdefault("open_positions", {})[nfo_symbol] = {
                            "id": trade_id,
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "symbol": nfo_symbol,
                            "side": tx_type,
                            "entry": approx_premium,
                            "stop": approx_premium * (1 - sl_pct),
                            "target": approx_premium * (1 + tgt_pct),
                            "trail_sl": approx_premium * (1 - trail_pct),
                            "qty": int(qty),
                            "mode": "PAPER",
                        }
                        st.success(f"PAPER trade recorded: {nfo_symbol} qty={qty} "
                                   f"price={approx_premium:.2f} SL={stop_loss_price:.2f} "
                                   f"TGT={target_price:.2f} TrailSL={trail_sl_price:.2f}")
            except Exception as e:
                st.error(f"⚠️ Could not place trade: {e}")

        with col_exec2:
            if st.button("Close All Paper Positions"):
                if "open_positions" in st.session_state and st.session_state["open_positions"]:
                    for sym in list(st.session_state["open_positions"].keys()):
                        st.session_state["open_positions"].pop(sym)
                        st.info(f"Closed paper pos {sym}")
                else:
                    st.info("No open positions")

    # ------------------- Tab 3: Positions -------------------
    with tab3:
        st.subheader("Open Positions")
        try:
            open_pos = st.session_state.get("open_positions", {})
            if open_pos:
                st.dataframe(pd.DataFrame(open_pos).T)
            else:
                st.info("No open positions")
        except Exception as e:
            st.warning(f"⚠️ Could not display positions: {e}")

except Exception as e:
    st.error(f"⚠️ Tabs could not load. Please refresh. ({e})")

# ------------------- Auto-refresh & Trailing SL -------------------
st_autorefresh(interval=30000, key="auto_refresh")
try:
    monitor_trailing_sl(kite)
except Exception as e:
    st.warning(f"⚠️ Trailing SL monitor error: {e}")
def monitor_trailing_sl(kite=None):
    open_pos = st.session_state.get("open_positions", {})
    if not open_pos:
        return

    updated = False
    for sym, pos in list(open_pos.items()):
        ltp = get_underlying_ltp(kite, sym) if kite else None
        if not ltp:
            continue

        # --- Update trailing SL ---
        new_trail = max(pos["trail_sl"], ltp * (1 - trail_pct))
        if new_trail > pos["trail_sl"]:
            pos["trail_sl"] = new_trail
            updated = True

        # --- Exit logic ---
        if ltp <= pos["stop"] or ltp <= pos["trail_sl"] or ltp >= pos["target"]:
            st.warning(f"Closing {sym} at {ltp:.2f} (hit SL/Target)")
            st.session_state["open_positions"].pop(sym)
            updated = True

    if updated:
        st.experimental_rerun()


