# ultra_legend_bot.py
"""
Ultra Legend — merged Advance + Super bots (Long-Only)
Features:
 - Single file: indicators, patterns, scanner, paper/live trading, Zerodha integration.
 - Uses yfinance for historical OHLC, optional NewsApi + VADER for sentiment.
 - Long-only default; can toggle.
 - Auto mode with st_autorefresh to scan every X minutes.
 - Saves Zerodha access token once per day to zerodha_token.json (request token needed only once/day).
 - Designed to run on Streamlit Cloud (put secrets in secrets.toml).
"""

from datetime import datetime, date, time
import time as pytime
import os
import json
import math

import pytz
import streamlit as st

# core libs
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except Exception as e:
    st.error("Missing core packages. Install yfinance pandas numpy.")
    raise

# optional libs (news & sentiment)
try:
    from newsapi import NewsApiClient
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import requests
    from bs4 import BeautifulSoup
except Exception:
    NewsApiClient = None
    SentimentIntensityAnalyzer = None
    requests = None
    BeautifulSoup = None

# optional autorefresh
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# optional kiteconnect (imported lazily when live)
try:
    from kiteconnect import KiteConnect  # may fail on environments without kiteconnect
except Exception:
    KiteConnect = None

# -------------------------
# Config / Globals
# -------------------------
IST = pytz.timezone("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
TOKEN_FILE = "zerodha_token.json"
TRADE_LOG_CSV = "ultra_legend_trades.csv"

# -------------------------
# Utility helpers
# -------------------------
def now_ist_str():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

def get_secret(key: str, default=""):
    """Try st.secrets, fallback to env var, else default"""
    try:
        v = st.secrets.get(key)
        if v:
            return v
    except Exception:
        pass
    return os.environ.get(key, default)

def save_local_token(access_token: str):
    try:
        data = {"access_token": access_token, "date": date.today().isoformat()}
        with open(TOKEN_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        st.warning(f"Failed to save token locally: {e}")

def load_local_token():
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def token_valid(token_data):
    return bool(token_data and token_data.get("date") == date.today().isoformat())

# -------------------------
# Session-state defaults
# -------------------------
def ensure_state():
    st.session_state.setdefault("open_positions", {})     # symbol -> position dict
    st.session_state.setdefault("trade_history", [])      # list of trades
    st.session_state.setdefault("starting_capital", 100000.0)
    st.session_state.setdefault("trade_counter", 1)
    st.session_state.setdefault("scanner_signals", [])
    st.session_state.setdefault("kite", None)
    st.session_state.setdefault("failed_fetches", [])
    st.session_state.setdefault("last_warnings", [])
    st.session_state.setdefault("auto_scan_count", 0)

ensure_state()

def log_fetch_failure(sym, err):
    st.session_state.failed_fetches.append({"time": now_ist_str(), "symbol": sym, "error": str(err)})

def log_warning(msg):
    st.session_state.last_warnings.append({"time": now_ist_str(), "msg": str(msg)})

def next_trade_id():
    tid = f"T{st.session_state.trade_counter:06d}"
    st.session_state.trade_counter += 1
    return tid

# -------------------------
# Universe (merged)
# Use Advance bot's universe (kept unique + sorted)
# -------------------------
NIFTY50 = sorted(list({
    "RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","HINDUNILVR","HDFC","SBIN","BHARTIARTL","KOTAKBANK",
    "ITC","LT","AXISBANK","ASIANPAINT","BAJFINANCE","MARUTI","WIPRO","ULTRACEMCO","SUNPHARMA","ONGC",
    "NTPC","TITAN","POWERGRID","NESTLEIND","ADANIPORTS","JSWSTEEL","TATAMOTORS","TATASTEEL","BAJAJFINSV",
    "HCLTECH","TECHM","HDFCLIFE","BPCL","CIPLA","DIVISLAB","DRREDDY","BRITANNIA","GRASIM","HEROMOTOCO",
    "BAJAJ-AUTO","EICHERMOT","COALINDIA","INDUSINDBK","UPL","SHREECEM","M&M","SBILIFE","APOLLOHOSP","HINDALCO"
}))

# -------------------------
# Sentiment helpers
# -------------------------
if NewsApiClient is not None and SentimentIntensityAnalyzer is not None:
    analyzer = SentimentIntensityAnalyzer()
    # NewsAPI key will come from secrets or sidebar input; we create client lazily
    def news_client_from_key(key):
        try:
            if key:
                return NewsApiClient(api_key=key)
        except Exception:
            return None
        return None

    @st.cache_data(ttl=3600)
    def get_sentiment_for_stock(stock_name: str, api_key: str):
        """Returns average compound VADER score across top articles (float)"""
        try:
            if not api_key or not stock_name:
                return 0.0
            client = news_client_from_key(api_key)
            if not client:
                return 0.0
            resp = client.get_everything(q=stock_name, language='en', sort_by='relevancy', page_size=20)
            headlines = [a.get("title","") for a in resp.get("articles",[])]
            if not headlines:
                return 0.0
            scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
            return float(sum(scores) / len(scores))
        except Exception:
            return 0.0

    @st.cache_data(ttl=600)
    def scrape_moneycontrol_news(keyword, max_headlines=5):
        if requests is None or BeautifulSoup is None:
            return []
        try:
            url = "https://www.moneycontrol.com/news/markets/"
            resp = requests.get(url, timeout=8)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.find_all("a", class_="clearfix", limit=200)
            headlines = []
            for link in links:
                title = link.get_text(strip=True)
                if keyword.lower() in title.lower():
                    headlines.append(title)
                    if len(headlines) >= max_headlines:
                        break
            return headlines
        except Exception:
            return []
else:
    def get_sentiment_for_stock(stock_name: str, api_key: str): return 0.0
    def scrape_moneycontrol_news(keyword, max_headlines=5): return []

# -------------------------
# Indicators & Candles (Super Bot engine)
# -------------------------
def sma(series, window):
    return series.rolling(window).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=(period-1), adjust=False).mean()
    ma_down = down.ewm(com=(period-1), adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.fillna(50)

def macd(series, n_fast=12, n_slow=26, n_signal=9):
    ema_fast = series.ewm(span=n_fast, adjust=False).mean()
    ema_slow = series.ewm(span=n_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=n_signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calculate_indicators_super(df):
    """Add RSI, MACD, BB, ATR, SuperTrend, EMAs, volume spike"""
    d = df.copy()
    # Normalise column names if needed
    cols = {c.lower(): c for c in d.columns if isinstance(c, str)}
    rename_map = {}
    for k,v in cols.items():
        if k == 'open': rename_map[v] = 'Open'
        if k == 'high': rename_map[v] = 'High'
        if k == 'low': rename_map[v] = 'Low'
        if k == 'close': rename_map[v] = 'Close'
        if k == 'volume': rename_map[v] = 'Volume'
        if k.startswith('adj close'): rename_map[v] = 'Close'
    if rename_map:
        d = d.rename(columns=rename_map)

    if 'Close' not in d.columns:
        raise ValueError("Close column missing in data")

    close = d['Close'].astype(float)
    high = d['High'].astype(float) if 'High' in d.columns else close
    low = d['Low'].astype(float) if 'Low' in d.columns else close
    vol = d['Volume'].astype(float) if 'Volume' in d.columns else None

    # RSI
    d['RSI'] = rsi(close)

    # MACD
    macd_line, macd_signal, macd_hist = macd(close)
    d['MACD'] = macd_line; d['MACD_SIGNAL'] = macd_signal; d['MACD_HIST'] = macd_hist

    # Bollinger
    sma20 = close.rolling(20).mean(); std20 = close.rolling(20).std(ddof=0)
    d['BB_MID'] = sma20; d['BB_UP'] = sma20 + 2*std20; d['BB_LOW'] = sma20 - 2*std20
    d['BB_WIDTH'] = (d['BB_UP'] - d['BB_LOW']) / d['BB_MID']

    # ATR components
    d['H-L'] = high - low
    d['H-PC'] = (high - close.shift(1)).abs()
    d['L-PC'] = (low - close.shift(1)).abs()
    d['TR'] = d[['H-L','H-PC','L-PC']].max(axis=1)
    d['ATR'] = d['TR'].rolling(10).mean().fillna(method='bfill')

    # SuperTrend-ish
    m = 3.0
    d['BASIC_UB'] = (high + low) / 2.0 + m * d['ATR']
    d['BASIC_LB'] = (high + low) / 2.0 - m * d['ATR']
    d['FINAL_UB'] = d['BASIC_UB'].copy(); d['FINAL_LB'] = d['BASIC_LB'].copy()
    for i in range(1, len(d)):
        if d['Close'].iat[i-1] <= d['FINAL_UB'].iat[i-1]:
            d['FINAL_UB'].iat[i] = min(d['BASIC_UB'].iat[i], d['FINAL_UB'].iat[i-1])
        else:
            d['FINAL_UB'].iat[i] = d['BASIC_UB'].iat[i]
        if d['Close'].iat[i-1] >= d['FINAL_LB'].iat[i-1]:
            d['FINAL_LB'].iat[i] = max(d['BASIC_LB'].iat[i], d['FINAL_LB'].iat[i-1])
        else:
            d['FINAL_LB'].iat[i] = d['BASIC_LB'].iat[i]
    d['SUPERTREND'] = np.where(d['Close'] <= d['FINAL_UB'], 'DOWN', 'UP')

    # EMAs
    d['EMA20'] = close.ewm(span=20, adjust=False).mean()
    d['EMA50'] = close.ewm(span=50, adjust=False).mean()
    d['EMA200'] = close.ewm(span=200, adjust=False).mean()

    if vol is not None:
        d['VOL_MA20'] = vol.rolling(20).mean()
        d['VOL_SPIKE'] = vol >= 1.5 * d['VOL_MA20']
    else:
        d['VOL_MA20'] = np.nan; d['VOL_SPIKE'] = False

    return d

# Candlestick pattern detector (kept from Super Bot)
def detect_candlestick_patterns(df):
    patterns = []; strength = 0
    if df is None or df.empty or len(df) < 2:
        return patterns, strength
    prev = df.iloc[-2]; cur = df.iloc[-1]
    lower = {c.lower(): c for c in df.columns if isinstance(c,str)}
    try:
        o1 = float(prev[lower.get('open','Open')]); h1 = float(prev[lower.get('high','High')]); l1 = float(prev[lower.get('low','Low')]); c1 = float(prev[lower.get('close','Close')])
        o2 = float(cur[lower.get('open','Open')]); h2 = float(cur[lower.get('high','High')]); l2 = float(cur[lower.get('low','Low')]); c2 = float(cur[lower.get('close','Close')])
    except Exception:
        return patterns, strength

    body1 = abs(c1 - o1); body2 = abs(c2 - o2); rng2 = max(h2 - l2, 1e-9)

    # Bullish Engulfing
    if (c2 > o2) and (c1 < o1) and (c2 >= o1) and (o2 <= c1):
        patterns.append("Bullish Engulfing"); strength += 3
    # Hammer
    lower_shadow = min(o2, c2) - l2
    if body2 < 0.3 * rng2 and lower_shadow > 2 * body2:
        patterns.append("Hammer"); strength += 2
    # Doji
    if body2 <= 0.1 * rng2:
        patterns.append("Doji"); strength += 1

    # Morning star (simple)
    if len(df) >= 3:
        p2 = df.iloc[-3]
        try:
            p2o = float(p2[lower.get('open','Open')]); p2c = float(p2[lower.get('close','Close')])
            if (p2c < p2o) and (c1 < o1) and (c2 > o2) and (c2 >= (p2o + p2c)/2):
                patterns.append("Morning Star"); strength += 3
        except Exception:
            pass

    return patterns, strength

# -------------------------
# fetch_history (safe)
# -------------------------
@st.cache_data(ttl=300)
def fetch_history(symbol: str, period: str="30d", interval: str="15m"):
    """Return (df, error). Normalizes column names to Titlecase Open/High/Low/Close/Volume."""
    try:
        if not symbol or not isinstance(symbol, str):
            return None, "Invalid symbol"
        ticker = f"{symbol}.NS"
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    except Exception as e:
        log_fetch_failure(symbol, e)
        return None, f"Download error for {symbol}: {e}"

    if df is None or df.empty:
        log_fetch_failure(symbol, "Empty data")
        return None, f"No data for {symbol}"

    # Normalize columns
    col_map = {}
    for c in df.columns:
        if not isinstance(c, str):
            continue
        lc = c.strip().lower()
        if lc.startswith("open"): col_map[c] = "Open"
        elif lc.startswith("high"): col_map[c] = "High"
        elif lc.startswith("low"): col_map[c] = "Low"
        elif lc.startswith("close"): col_map[c] = "Close"
        elif lc.startswith("volume"): col_map[c] = "Volume"
    if col_map:
        df = df.rename(columns=col_map)

    # ensure DateTime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    required = {"Open","High","Low","Close"}
    if not required.issubset(set([c for c in df.columns if isinstance(c,str)])):
        log_fetch_failure(symbol, f"Missing OHLC {list(df.columns)}")
        return None, f"Missing OHLC columns for {symbol}. Columns: {list(df.columns)}"

    return df, None

# -------------------------
# Position sizing
# -------------------------
def calculate_qty(capital, entry_price, stop_price, risk_pct, max_cap_alloc_pct=25, min_qty=1):
    try:
        if entry_price is None or stop_price is None or entry_price <= 0:
            return 0
        entry = float(entry_price); stop = float(stop_price)
        per_share_risk = abs(entry - stop)
        if per_share_risk <= 0:
            return 0
        risk_amount = float(capital) * (float(risk_pct) / 100.0)
        raw_qty = int(risk_amount // per_share_risk)
        alloc_cap = float(capital) * (float(max_cap_alloc_pct) / 100.0)
        max_qty_by_cap = int(alloc_cap // entry) if entry > 0 else 0
        qty = min(raw_qty, max_qty_by_cap)
        if qty < min_qty:
            return 0
        return int(qty)
    except Exception:
        return 0

# -------------------------
# Zerodha helpers
# -------------------------
def get_live_price(kite, symbol):
    try:
        inst = f"NSE:{symbol}"
        ltp = kite.ltp(inst)
        return ltp.get(inst, {}).get('last_price')
    except Exception:
        return None

def place_real_order_with_zerodha(kite, symbol, side, qty, tag=None, long_only=True):
    """
    Places market MIS order. Returns dict or None.
    Respects long_only: blocks SELL entries if long_only True.
    """
    try:
        if long_only and side == "SELL":
            st.warning(f"SELL blocked by long-only: {symbol}")
            return None
        txn = kite.TRANSACTION_TYPE_BUY if side == 'BUY' else kite.TRANSACTION_TYPE_SELL
        tag = tag or f"ULTRA_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NSE,
            tradingsymbol=symbol,
            transaction_type=txn,
            quantity=int(qty),
            order_type=kite.ORDER_TYPE_MARKET,
            product=kite.PRODUCT_MIS,
            tag=tag
        )
        # poll orders
        for _ in range(12):
            orders = kite.orders()
            od = next((o for o in orders if str(o.get('order_id')) == str(order_id)), None)
            if od and od.get('status') in ("COMPLETE","OPEN","TRIGGERED","TRIGGER PENDING"):
                filled_qty = int(od.get('filled_quantity', 0) or 0)
                avg_price = float(od.get('average_price') or 0.0)
                return {'order_id': order_id, 'filled_qty': filled_qty, 'avg_price': avg_price}
            pytime.sleep(1)
    except Exception as e:
        st.error(f"Zerodha order failed for {symbol}: {e}")
    return None

def sync_zerodha_positions():
    if 'kite' not in st.session_state or st.session_state.kite is None:
        st.warning("Zerodha not connected.")
        return
    try:
        kite = st.session_state.kite
        pos = kite.positions().get('day', []) or []
        live_symbols = {p['tradingsymbol'] for p in pos if p.get('quantity',0) > 0}
        bot_syms = list(st.session_state.open_positions.keys())
        for s in bot_syms:
            p = st.session_state.open_positions.get(s)
            if p and p.get('mode') == 'LIVE' and s not in live_symbols:
                ltp = get_live_price(kite, s) or p.get('entry')
                pnl = (ltp - p['entry']) * p['qty'] if p['side']=='BUY' else (p['entry'] - ltp)*p['qty']
                st.session_state.trade_history.append({
                    'TradeID': p.get('id','manual'),
                    'OpenTime': p.get('time'),
                    'CloseTime': now_ist_str(),
                    'Symbol': s,
                    'Side': p.get('side'),
                    'Qty': p.get('qty'),
                    'Entry': p.get('entry'),
                    'Exit': ltp,
                    'PnL': float(pnl),
                    'Reason': 'Manual Exit (Zerodha)'
                })
                del st.session_state.open_positions[s]
                st.success(f"Synced manual exit for {s} | PnL: ₹{pnl:.2f}")
    except Exception as e:
        st.error(f"Sync failed: {e}")

# -------------------------
# Place trade (paper/live)
# -------------------------
def place_trade(symbol, side, entry_price, sl, tgt, qty, tsl_pct=0.0, live_trading=False, long_only=True):
    # Enforce long-only
    if long_only and side == "SELL":
        st.warning(f"SELL blocked (long-only): {symbol}")
        return

    trade_id = next_trade_id()
    nowts = now_ist_str()
    filled_price = float(entry_price)
    filled_qty = int(qty)
    order_id = None
    mode = 'PAPER'
    note = 'PAPER-FILLED'

    if live_trading and st.session_state.get('kite'):
        res = place_real_order_with_zerodha(st.session_state.kite, symbol, side, qty, tag=f"{trade_id}_{symbol}", long_only=long_only)
        if not res:
            st.error(f"Live order failed for {symbol}. Aborting open.")
            return
        filled_qty = int(res.get('filled_qty', filled_qty))
        filled_price = float(res.get('avg_price') or filled_price)
        order_id = res.get('order_id')
        mode = 'LIVE'
        note = f"ZERODHA-FILLED ({order_id})"

    st.session_state.open_positions[symbol] = {
        'id': trade_id,
        'time': nowts,
        'symbol': symbol,
        'side': side,
        'entry': float(filled_price),
        'stop': float(sl) if sl is not None else None,
        'target': float(tgt) if tgt is not None else None,
        'qty': int(filled_qty),
        'order_id': order_id,
        'mode': mode,
        'note': note,
        'tsl_pct': float(tsl_pct)
    }

    st.session_state.trade_history.append({
        'TradeID': trade_id, 'OpenTime': nowts, 'Symbol': symbol, 'Side': side,
        'Qty': int(filled_qty), 'Entry': float(filled_price), 'Exit': None, 'PnL': 0.0, 'Mode': mode, 'Note': note
    })
    st.success(f"Opened {mode} position {symbol} | {side} | Qty {filled_qty} @ ₹{filled_price:.2f}")

# -------------------------
# Exit checks & trailing
# -------------------------
def check_exits_with_price(sym, price, live_trading=False):
    positions = st.session_state.open_positions
    if sym not in positions:
        return
    pos = positions[sym]
    side = pos['side']; entry = pos['entry']; stop = pos['stop']; target = pos['target']; qty = pos['qty']; mode = pos.get('mode','PAPER')
    tsl_pct = pos.get('tsl_pct', 0.0)
    exit_reason = None; exit_price = None

    if tsl_pct and tsl_pct > 0 and side == 'BUY':
        new_tsl = price * (1 - tsl_pct/100.0)
        if new_tsl > (pos.get('stop') or 0):
            pos['stop'] = new_tsl
            st.session_state.open_positions[sym]['stop'] = new_tsl
            st.info(f"TSL for {sym} updated to ₹{new_tsl:.2f}")

    if side == 'BUY':
        if stop is not None and price <= stop:
            exit_price = stop; exit_reason = 'Stop Loss'
        elif target is not None and price >= target:
            exit_price = target; exit_reason = 'Target Hit'

    if exit_reason and exit_price is not None:
        pnl = (exit_price - entry) * qty
        st.session_state.trade_history.append({
            'TradeID': pos.get('id', f"{sym}_{datetime.now().strftime('%H%M%S')}"),
            'Symbol': sym, 'Side': side, 'Qty': qty, 'Entry': entry, 'Exit': exit_price,
            'PnL': float(pnl), 'Reason': exit_reason, 'CloseTime': now_ist_str(), 'Mode': mode, 'OrderID': pos.get('order_id')
        })
        if mode == 'LIVE' and live_trading and st.session_state.get('kite'):
            try:
                kite = st.session_state.kite
                txn_type = kite.TRANSACTION_TYPE_SELL if side == 'BUY' else kite.TRANSACTION_TYPE_BUY
                kite.place_order(variety=kite.VARIETY_REGULAR, exchange=kite.EXCHANGE_NSE, tradingsymbol=sym, transaction_type=txn_type, quantity=int(qty), order_type=kite.ORDER_TYPE_MARKET, product=kite.PRODUCT_MIS)
                st.info(f"LIVE automated exit order sent for {sym} | {exit_reason}")
            except Exception as e:
                st.error(f"Failed live exit for {sym}: {e}")
        del st.session_state.open_positions[sym]
        st.success(f"Exited {sym} | {side} | {exit_reason} | PnL: ₹{pnl:.2f}")

# -------------------------
# Watchlist filter (merge of both bots)
# -------------------------
def get_filtered_watchlist(df_dict, bb_width_min_pct=2.0, vol_mult=1.5):
    """Return list of symbols meeting filters:
       - sufficient length
       - vol spike
       - EMA20 & EMA50 positive alignment
       - Bollinger width (percent) >= threshold
       - gap <= threshold
    """
    watchlist = []
    for symbol, df in df_dict.items():
        try:
            if df is None or len(df) < 50:
                continue
            latest = df.iloc[-1]; prev = df.iloc[-2]
            # volume spike
            if 'Volume' in df.columns:
                avg_vol = df['Volume'].rolling(20).mean().iloc[-2]
                if not np.isnan(avg_vol) and latest['Volume'] < vol_mult * avg_vol:
                    continue
            # trend: price above EMA20 & EMA50
            if not (latest.get('Close',0) > df['EMA20'].iloc[-1] and latest.get('Close',0) > df['EMA50'].iloc[-1]):
                continue
            # BB width percent
            if 'BB_UP' in df.columns and 'BB_LOW' in df.columns:
                bb_w = ((df['BB_UP'].iloc[-1] - df['BB_LOW'].iloc[-1]) / latest.get('Close',1)) * 100.0
                if bb_w < bb_width_min_pct:
                    continue
            # gap filter
            gap_pct = abs((latest.get('Open',0) - prev.get('Close',0)) / prev.get('Close',1)) * 100.0
            if gap_pct > 2.0:
                continue
            watchlist.append(symbol)
        except Exception:
            continue
    return watchlist

# -------------------------
# Strategy: analyze and trade
# -------------------------
def analyze_and_trade(symbol, df, live_trading=False, settings=None):
    """Hybrid strategy: score using EMA stack, MACD, SuperTrend, RSI, volume spike + candlestick patterns.
       Trades only if sentiment positive (if enabled) and long-only entry is honored.
    """
    if settings is None:
        settings = {}
    # Portfolio gates
    starting_cap = float(st.session_state.get('starting_capital', 100000.0))
    total_pnl = sum([float(t.get('PnL', 0.0)) for t in st.session_state.get('trade_history', [])])
    daily_loss_limit = starting_cap * (float(settings.get('max_daily_loss_pct', st.session_state.get('max_daily_loss_pct', 2.0))) / 100.0)
    if total_pnl < -daily_loss_limit:
        log_warning("Daily loss limit reached; skipping entries.")
        return

    if len(st.session_state.open_positions) >= int(settings.get('max_open_positions', st.session_state.get('max_open_positions', 10))):
        return

    try:
        d = calculate_indicators_super(df)
    except Exception as e:
        log_fetch_failure(symbol, f"Indicator calc failed: {e}")
        return

    patterns, pat_strength = detect_candlestick_patterns(d)
    last = d.iloc[-1]
    # quick qualifiers
    vol_spike = bool(d.get('VOL_SPIKE', False) and d['VOL_SPIKE'].iloc[-1]) if 'VOL_SPIKE' in d.columns else False
    bb_width = float(d.get('BB_WIDTH', 0.0).iloc[-1]) if 'BB_WIDTH' in d.columns else 0.0
    super_up = (last.get('SUPERTREND') == 'UP')
    ema_stack = last['EMA20'] > last['EMA50'] > last['EMA200'] if all(k in last.index for k in ['EMA20','EMA50','EMA200']) else False
    macd_ok = last['MACD'] > last['MACD_SIGNAL'] if 'MACD' in last.index and 'MACD_SIGNAL' in last.index else False
    rsi_ok = 40 < last.get('RSI',50) < 70

    # scoring
    score = 0
    if ema_stack: score += 20
    if macd_ok: score += 12
    if super_up: score += 10
    if vol_spike: score += 8
    score += pat_strength * 3
    if bb_width >= 0.02: score += 5

    # sentiment gating (if enabled)
    senti_score = 0.0
    if settings.get('use_sentiment'):
        try:
            senti_score = get_sentiment_for_stock(symbol, settings.get('news_api_key', ""))
            # require positive sentiment to add votes
            if senti_score > settings.get('sentiment_threshold', 0.1):
                score += 6
        except Exception:
            senti_score = 0.0

    # require minimum score + candlestick confirmation
    qualifies = False
    if score >= settings.get('min_score', 30):
        if ("Bullish Engulfing" in patterns or "Hammer" in patterns or pat_strength >= 2):
            if rsi_ok and macd_ok and super_up:
                qualifies = True

    if not qualifies:
        return

    # sizing
    capital = float(starting_cap)
    risk_pct = float(settings.get('risk_pct', st.session_state.get('risk_pct', 1.0)))
    max_alloc = float(settings.get('max_cap_alloc_pct', st.session_state.get('max_cap_per_trade',25)))
    entry_price = float(last['Close'])
    stop_price = float(last['Low'] * 0.995) if last['Low'] < entry_price else float(last['Low'])
    raw_qty = calculate_qty(capital, entry_price, stop_price, risk_pct, max_alloc)
    if raw_qty <= 0:
        return
    qty = raw_qty
    # strength boost
    if score >= 60:
        boost = float(settings.get('strength_boost', st.session_state.get('strength_boost',25)))
        qty = int(min(qty * (1.0 + boost/100.0), int((capital * max_alloc/100.0) // entry_price)))
    if qty <= 0:
        return

    # target & sl
    stop_loss = stop_price
    tgt = round(entry_price * (1.0 + settings.get('target_pct', 0.02)), 2)  # default 2% target
    tsl_pct = float(settings.get('tsl_pct', st.session_state.get('tsl_pct', 0.0)))

    # Long-only enforcement is done at place_trade, but we also enforce here for clarity
    if settings.get('long_only', True):
        side = "BUY"
    else:
        side = "BUY"  # this bot is long-first; we keep buy-only provided by user

    st.info(f"Placing {'LIVE' if live_trading else 'PAPER'} BUY {symbol} | Qty={qty} | Entry={entry_price:.2f} | SL={stop_loss:.2f} | TGT={tgt:.2f} | Sentiment={senti_score:.3f}")
    place_trade(symbol, side, entry_price, stop_loss, tgt, qty, tsl_pct, live_trading=live_trading, long_only=settings.get('long_only', True))

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="⚡ Ultra Legend — Long-Only Bot", layout="wide")
st.title("⚡ Ultra Legend — Hybrid Long-Only Bot")

# Sidebar controls
with st.sidebar:
    st.header("Connection & Mode")
    mode = st.radio("Mode", ["Auto (scan NIFTY50)", "Manual selection"], index=0)
    interval = st.selectbox("Interval", ["5m","15m","30m","60m"], index=1)
    period = st.selectbox("History Period", ["7d","14d","30d","60d"], index=1)

    st.markdown("---")
    st.subheader("Trading Mode")
    auto_place = st.checkbox("Auto place trades (will auto-open when found)", value=False)
    live_trading = st.checkbox("Enable LIVE trading (use with caution)", value=False)
    long_only_toggle = st.checkbox("Enforce Long-Only (block SELL entries)", value=True)

    st.markdown("---")
    st.subheader("Risk & Money Mgmt")
    st.session_state.starting_capital = st.number_input("Starting capital (₹)", value=float(st.session_state.starting_capital), step=1000.0)
    st.session_state.risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    st.session_state.max_cap_per_trade = st.slider("Max capital per trade (%)", 5, 50, 25, step=5)
    st.session_state.max_open_positions = st.number_input("Max open positions", min_value=1, max_value=50, value=10)
    st.session_state.max_daily_loss_pct = st.number_input("Daily loss limit (%)", min_value=0.5, max_value=50.0, value=2.0, step=0.5)
    st.session_state.tsl_pct = st.number_input("Trailing Stop-Loss (%)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    st.session_state.strength_boost = st.slider("Strength boost (%)", 0, 100, 25, step=5)

    st.markdown("---")
    st.subheader("Signals & Filters")
    use_sentiment = st.checkbox("Require positive news sentiment to trade", value=True)
    news_api_key_input = st.text_input("NewsAPI key (optional)", type="password", value=get_secret("NEWSAPI_KEY", ""))
    sentiment_threshold = st.number_input("Sentiment threshold (compound)", min_value=-1.0, max_value=1.0, value=0.1, step=0.01)
    min_score = st.number_input("Min technical score to qualify", min_value=0, max_value=200, value=30, step=1)
    target_pct_ui = st.number_input("Target (%)", min_value=0.1, max_value=20.0, value=2.0, step=0.1)

    st.markdown("---")
    st.subheader("Auto Scan")
    auto_scan = st.checkbox("Enable Auto Scan (periodic)", value=False)
    auto_scan_min = st.number_input("Auto-scan every X minutes", min_value=1, max_value=60, value=5, step=1)
    if st_autorefresh is not None and auto_scan:
        # Execute autorefresh by incrementing a counter - we also use this flag later to run scan
        _ = st_autorefresh(interval=auto_scan_min * 60 * 1000, limit=None, key="auto_scan")

    st.markdown("---")
    st.subheader("Zerodha (LIVE) Connect")
    input_api_key = st.text_input("Kite API Key (optional)", type="password", value=get_secret("API_KEY",""))
    input_api_secret = st.text_input("Kite API Secret (optional)", type="password", value=get_secret("API_SECRET",""))
    input_request_token = st.text_input("Request Token (enter only when creating access token today)")
    if st.button("Create Access Token / Connect") or (live_trading and input_request_token):
        if KiteConnect is None:
            st.error("kiteconnect not installed. pip install kiteconnect")
        else:
            API_KEY = input_api_key or get_secret("API_KEY","")
            API_SECRET = input_api_secret or get_secret("API_SECRET","")
            if not API_KEY or not API_SECRET or not input_request_token:
                st.warning("Provide API_KEY, API_SECRET and Request Token (from Kite login flow).")
            else:
                try:
                    kite = KiteConnect(api_key=API_KEY)
                    data = kite.generate_session(input_request_token, api_secret=API_SECRET)
                    kite.set_access_token(data["access_token"])
                    st.session_state.kite = kite
                    save_local_token(data["access_token"])
                    st.success("Connected to Zerodha and token saved locally.")
                except Exception as e:
                    st.error(f"Zerodha login failed: {e}")

    # If live and saved token exists auto-connect
    if live_trading and KiteConnect is not None and st.session_state.get('kite') is None:
        token_data = load_local_token()
        API_KEY = input_api_key or get_secret("API_KEY","")
        if token_valid(token_data) and API_KEY:
            try:
                kite = KiteConnect(api_key=API_KEY)
                kite.set_access_token(token_data["access_token"])
                st.session_state.kite = kite
                st.success("Auto-connected to Zerodha using saved token.")
            except Exception:
                pass

    st.markdown("---")
    st.subheader("Universe Selection")
    if mode.startswith("Manual"):
        selected_symbols = st.multiselect("Symbols (NIFTY50 preloaded)", options=NIFTY50, default=["RELIANCE","TCS","INFY"])
    else:
        selected_symbols = []

    st.markdown("---")
    st.button("Run One Scan Now", key="run_manual_scan")

# -------------------------
# Main: scanning & trading
# -------------------------
col1, col2 = st.columns([3,1])
with col1:
    st.subheader("Scanner & Signals")

    # Decide symbols to scan
    if mode.startswith("Auto"):
        symbols_to_scan = NIFTY50
    else:
        symbols_to_scan = selected_symbols

    run_manual = st.session_state.get("run_manual_scan", False) or st.button("Run Scan") or (auto_scan and st.session_state.auto_scan_count==0)
    # note: if auto_scan via st_autorefresh triggered above, the page reloads and keeps running

    if run_manual or (auto_scan and True):
        # increment autoscans counter to avoid duplicate immediate runs
        st.session_state.auto_scan_count = st.session_state.get("auto_scan_count",0) + 1

        # fetch histories
        df_dict = {}
        progress = st.progress(0)
        total = max(len(symbols_to_scan),1)
        i = 0
        for sym in symbols_to_scan:
            i += 1
            progress.progress(int(i/total*100))
            df_hist, err = fetch_history(sym, period=period, interval=interval)
            if err or df_hist is None or df_hist.empty:
                log_fetch_failure(sym, err or "empty")
                continue
            try:
                df_ind = calculate_indicators_super(df_hist)
            except Exception as e:
                log_fetch_failure(sym, f"indicator_err:{e}")
                continue
            df_dict[sym] = df_ind

        # filter watchlist
        watch = get_filtered_watchlist(df_dict, bb_width_min_pct=2.0, vol_mult=1.5)
        st.success(f"Filtered watchlist ({len(watch)}): {watch}")
        # analyze & optionally place trades
        settings = {
            'risk_pct': st.session_state.get('risk_pct', 1.0),
            'max_cap_alloc_pct': st.session_state.get('max_cap_per_trade', 25),
            'tsl_pct': st.session_state.get('tsl_pct', 0.0),
            'strength_boost': st.session_state.get('strength_boost', 25),
            'use_sentiment': use_sentiment,
            'news_api_key': news_api_key_input,
            'sentiment_threshold': sentiment_threshold,
            'min_score': min_score,
            'long_only': long_only_toggle,
            'max_open_positions': st.session_state.get('max_open_positions', 10),
            'max_daily_loss_pct': st.session_state.get('max_daily_loss_pct', 2.0),
            'risk_pct': st.session_state.get('risk_pct', 1.0),
            'target_pct': target_pct_ui / 100.0
        }

        # run analysis on filtered list
        for sym in watch:
            try:
                analyze_and_trade(sym, df_dict[sym], live_trading=(live_trading and st.session_state.get('kite') is not None), settings=settings)
            except Exception as e:
                log_warning(f"analyze error {sym}: {e}")
                st.error(f"Error analyzing {sym}: {e}")

    # show scan history & signals
    st.markdown("---")
    st.subheader("Recent Signals & Scanner Log")
    if st.session_state.scanner_signals:
        df_signals = pd.DataFrame(st.session_state.scanner_signals, columns=["Time","Symbol","Signal","Price","Stop","Target"])
        st.dataframe(df_signals.tail(200), use_container_width=True)
    else:
        st.info("No scanner signals yet (run a scan).")

with col2:
    st.subheader("Bot Status & Controls")
    total_pnl = sum([float(t.get('PnL',0.0)) for t in st.session_state.trade_history])
    pnl_color = "green" if total_pnl >= 0 else "red"
    st.markdown(f"**Total PnL:** <span style='color:{pnl_color}'>₹{total_pnl:.2f}</span>", unsafe_allow_html=True)
    st.markdown(f"**Open Positions:** {len(st.session_state.open_positions)}")
    st.markdown(f"**Trade Count:** {len(st.session_state.trade_history)}")
    st.markdown("---")
    st.subheader("Debug / Fetch Failures")
    if st.session_state.failed_fetches:
        st.dataframe(pd.DataFrame(st.session_state.failed_fetches).sort_values("time", ascending=False).head(50))
    else:
        st.info("No fetch failures logged.")
    st.markdown("---")
    st.subheader("Warnings")
    if st.session_state.last_warnings:
        st.dataframe(pd.DataFrame(st.session_state.last_warnings).sort_values("time", ascending=False).head(50))
    else:
        st.info("No warnings.")

# -------------------------
# Open positions and trade history
# -------------------------
st.markdown("---")
st.subheader("Open Positions")
if st.session_state.open_positions:
    pos_df = pd.DataFrame.from_dict(st.session_state.open_positions, orient='index')
    st.dataframe(pos_df, use_container_width=True)
else:
    st.info("No open positions")

st.markdown("---")
st.subheader("Trade History (recent)")
if st.session_state.trade_history:
    hist_df = pd.DataFrame(st.session_state.trade_history).tail(200)
    st.dataframe(hist_df, use_container_width=True)
    if st.button("Export trade history to CSV"):
        hist_df.to_csv(TRADE_LOG_CSV, index=False)
        st.success(f"Saved to {TRADE_LOG_CSV}")
else:
    st.info("No trade history yet.")

# -------------------------
# Live loop: check exits & update LTP
# -------------------------
if live_trading and st.session_state.get('kite'):
    try:
        kite = st.session_state.kite
        for sym, pos in list(st.session_state.open_positions.items()):
            if pos.get('mode') == 'LIVE':
                ltp = get_live_price(kite, sym)
                if ltp is not None:
                    st.session_state.open_positions[sym]['LTP'] = ltp
                    check_exits_with_price(sym, ltp, live_trading=True)
    except Exception as e:
        log_warning(f"live update error: {e}")

st.caption("Ultra Legend — merged from Advance + Super. Review live order code and test in paper mode before using real money.")
