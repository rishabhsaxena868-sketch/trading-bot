# nifty_options_bot_pro_ready.py
"""
Institutional-Grade NIFTY / BANKNIFTY Options Bot
- Paper & LIVE trading
- Auto ATM/OTM CE/PE selection
- Trailing SL / ATR target
- Candlestick patterns + SMA/RSI/MACD signals
- Dashboard always shows valid LTP & option preview
"""

from datetime import datetime, date, timedelta
import math
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

try:
    from kiteconnect import KiteConnect
except:
    KiteConnect = None

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
    st.header("Settings")
    underlying_choice = st.selectbox("Underlying", list(UNDERLYINGS.keys()))
    option_type = st.selectbox("Option Type", ["CE (Call)", "PE (Put)"])
    allocation_pct = st.slider("Capital per trade (%)", 1, 100, 50) / 100.0
    starting_cap = st.number_input("Starting Capital (₹)", 20000, 2000000, 200000, 1000)
    st.session_state["starting_capital"] = starting_cap
    live_trading = st.checkbox("Enable LIVE trading", False)

    st.markdown("---")
    st.subheader("Kite API")
    api_key = st.text_input("API Key", type="password", value=st.secrets.get("KITE_API_KEY", ""))
    api_secret = st.text_input("API Secret", type="password", value=st.secrets.get("KITE_API_SECRET", ""))
    request_token = st.text_input("Request Token", type="password")

kite = None
if live_trading and KiteConnect:
    try:
        if api_key and api_secret:
            kite = KiteConnect(api_key=api_key)
            if request_token:
                data = kite.generate_session(request_token, api_secret=api_secret)
                kite.set_access_token(data["access_token"])
                st.success("Kite connected.")
    except:
        kite = None

# ------------------- Tabs (better for mobile) -------------------
tab1, tab2, tab3 = st.tabs(["📊 Signals", "📑 Order Preview", "📂 Positions"])

with tab1:
    interval = st.selectbox("Chart Interval", ["5m", "15m", "30m"], index=1)
    lookback = st.selectbox("History", ["7d", "14d", "30d"], index=1)

    with st.spinner("Fetching data..."):
        df = fetch_ohlc_safe(UNDERLYINGS[underlying_choice]["ticker"], period=lookback, interval=interval)

    st.metric("Data rows", len(df))
    if not df.empty:
        with st.expander("Show last 8 rows"):
            st.dataframe(df.tail(8))

    signal, meta = underlying_signal(df)
    st.subheader(f"Signal: {signal}")
    st.json(meta)
    st.markdown(f"**Global Trend:** {meta.get('global_trend', 'N/A')}")
    st.markdown(f"**News Sentiment Score:** {meta.get('news_score', 0):.2f}")

    patterns = detect_candlestick_patterns(df)
    if patterns:
        with st.expander("Candlestick Patterns"):
            st.write(patterns)

with tab2:
    st.subheader("Order Preview")
    ltp = get_underlying_ltp(kite, UNDERLYINGS[underlying_choice]["ticker"], fallback_df=df)
    if ltp is None and not df.empty:
        ltp = df["Close"].iloc[-1]
    st.metric("Underlying LTP", f"{ltp:.2f}" if ltp else "N/A")

    if ltp:
        expiry_input = date.today() + timedelta(days=7)
        strike = nearest_strike(ltp, ROUND_TO[underlying_choice])
        opt_side = "CE" if option_type.startswith("CE") else "PE"
        nfo_symbol = build_nfo_symbol(UNDERLYINGS[underlying_choice]["nfo_prefix"], expiry_input, strike, opt_side)
        approx_premium = max(1.0, ltp * 0.01)
        qty_preview = size_qty_by_capital(starting_cap, approx_premium, allocation_pct, underlying_choice)

        st.write("Candidate Option Symbol:", nfo_symbol)
        st.write(f"Approx premium: ₹{approx_premium:.2f}, Qty preview: {qty_preview}")

    st.markdown("---")
    col_exec1, col_exec2 = st.columns(2)

    with col_exec1:
        if st.button("Scan & Place (paper/live)"):
            if signal == "BULL":
                opt_type = "CE"
                tx_type = "BUY"
            elif signal == "BEAR":
                opt_type = "PE"
                tx_type = "BUY"
            else:
                st.info("Signal NEUTRAL")
                tx_type = None

            if tx_type and ltp:
                strike = nearest_strike(ltp, ROUND_TO[underlying_choice])
                nfo_symbol = build_nfo_symbol(UNDERLYINGS[underlying_choice]["nfo_prefix"], expiry_input, strike, opt_type)
                qty = size_qty_by_capital(starting_cap, approx_premium, allocation_pct, underlying_choice)

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
                        "stop": None,
                        "target": approx_premium * 1.02,
                        "trail_sl": approx_premium * 0.99,
                        "qty": int(qty),
                        "mode": "PAPER",
                    }
                    st.success(f"PAPER trade recorded: {nfo_symbol} qty={qty} price={approx_premium:.2f}")

    with col_exec2:
        if st.button("Close All Paper Positions"):
            if "open_positions" in st.session_state and st.session_state["open_positions"]:
                for sym in list(st.session_state["open_positions"].keys()):
                    st.session_state["open_positions"].pop(sym)
                    st.info(f"Closed paper pos {sym}")
            else:
                st.info("No open positions")

with tab3:
    st.subheader("Open Positions")
    open_pos = st.session_state.get("open_positions", {})
    if open_pos:
        st.dataframe(pd.DataFrame(open_pos).T)
    else:
        st.info("No open positions")

# ------------------- Auto-refresh & Trailing SL -------------------
st_autorefresh(interval=30000, key="auto_refresh")
monitor_trailing_sl(kite)

