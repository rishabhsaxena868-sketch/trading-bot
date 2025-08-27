# ultra_hybrid_longonly_bot.py
"""
Ultra Hybrid Long-Only Trading Bot (Streamlit)
Merged from: advance + super bots (deduped, errors fixed, long-only)
Requirements: streamlit, yfinance, pandas, numpy, kiteconnect (optional), requests, bs4, newsapi, vaderSentiment (optional)
Run: streamlit run ultra_hybrid_longonly_bot.py
"""

from datetime import datetime, time, date
import pytz
import streamlit as st
import json
import os
import math
import time as pytime

# third-party core
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except Exception as e:
    st.error("Missing packages: install yfinance, pandas, numpy.")
    raise

# optional sentiment libs
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

# optional kiteconnect
try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None

# -------------------------
# Config / Globals
# -------------------------
IST = pytz.timezone("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
TOKEN_FILE = "zerodha_token.json"
STATE_DB = "ultra_trades_log.csv"

# -------------------------
# Session-state defaults
# -------------------------
def ensure_state():
    if 'open_positions' not in st.session_state: st.session_state.open_positions = {}
    if 'trade_history' not in st.session_state: st.session_state.trade_history = []
    if 'starting_capital' not in st.session_state: st.session_state.starting_capital = 100000.0
    if 'trade_counter' not in st.session_state: st.session_state.trade_counter = 1
    if 'scanner_signals' not in st.session_state: st.session_state.scanner_signals = []
    if 'kite' not in st.session_state: st.session_state.kite = None

ensure_state()

def now_ist_str():
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

def pct(x):
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "-"

# -------------------------
# Universe (merged)
# -------------------------
NIFTY50 = sorted(list({
    "RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","HINDUNILVR","HDFC","SBIN","BHARTIARTL","KOTAKBANK",
    "ITC","LT","AXISBANK","ASIANPAINT","BAJFINANCE","MARUTI","WIPRO","ULTRACEMCO","SUNPHARMA","ONGC",
    "NTPC","TITAN","POWERGRID","NESTLEIND","ADANIPORTS","JSWSTEEL","TATAMOTORS","TATASTEEL","BAJAJFINSV",
    "HCLTECH","TECHM","HDFCLIFE","BPCL","CIPLA","DIVISLAB","DRREDDY","BRITANNIA","GRASIM","HEROMOTOCO",
    "BAJAJ-AUTO","EICHERMOT","COALINDIA","INDUSINDBK","UPL","SHREECEM","M&M","SBILIFE","APOLLOHOSP","HINDALCO"
}))

# -------------------------
# Helper: token persistence
# -------------------------
def save_local_token(access_token: str):
    try:
        data = {"access_token": access_token, "date": date.today().isoformat()}
        with open(TOKEN_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass

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
# Optional sentiment
# -------------------------
if NewsApiClient is not None and SentimentIntensityAnalyzer is not None:
    analyzer = SentimentIntensityAnalyzer()
    newsapi = NewsApiClient(api_key=st.secrets.get("NEWSAPI_KEY",""))

    @st.cache_data(ttl=3600)
    def get_sentiment_for_stock(stock_name: str):
        try:
            if not newsapi or not stock_name:
                return 0.0
            resp = newsapi.get_everything(q=stock_name, language='en', sort_by='relevancy', page_size=20)
            headlines = [a['title'] for a in resp.get('articles',[])]
            if not headlines:
                return 0.0
            scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
            return float(sum(scores) / len(scores))
        except Exception:
            return 0.0

    @st.cache_data(ttl=600)
    def scrape_moneycontrol_news(keyword, max_headlines=5):
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
    def get_sentiment_for_stock(stock_name: str): return 0.0
    def scrape_moneycontrol_news(keyword, max_headlines=5): return []

# -------------------------
# Indicators & pattern funcs
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

# SuperTrend / ATR & Bollinger from Super bot
def calculate_indicators_super(df):
    """
    Adds many indicators to df (expects Close, High, Low, Volume columns present).
    Returns a new DataFrame with columns added. Non-destructive.
    """
    d = df.copy()
    # ensure title-case column names
    cols = {c.lower(): c for c in d.columns}
    # map to canonical names if possible
    if 'close' not in cols and 'adj close' in cols:
        d.rename(columns={cols['adj close']:'Close'}, inplace=True)
    # standardize to Title format where possible
    rename_map = {}
    for k,v in cols.items():
        if k == 'open': rename_map[v] = 'Open'
        if k == 'high': rename_map[v] = 'High'
        if k == 'low': rename_map[v] = 'Low'
        if k == 'close': rename_map[v] = 'Close'
        if k == 'volume': rename_map[v] = 'Volume'
    if rename_map:
        d = d.rename(columns=rename_map)

    if 'Close' not in d.columns:
        raise ValueError("Close column missing in historical data")

    close = d['Close'].astype(float)
    high = d['High'].astype(float) if 'High' in d.columns else close
    low = d['Low'].astype(float) if 'Low' in d.columns else close
    vol = d['Volume'].astype(float) if 'Volume' in d.columns else None

    # RSI
    d['RSI'] = rsi(close)

    # MACD
    macd_line, macd_signal, macd_hist = macd(close)
    d['MACD'] = macd_line
    d['MACD_SIGNAL'] = macd_signal
    d['MACD_HIST'] = macd_hist

    # Bollinger
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std(ddof=0)
    d['BB_MID'] = sma20
    d['BB_UP'] = sma20 + 2*std20
    d['BB_LOW'] = sma20 - 2*std20
    d['BB_WIDTH'] = (d['BB_UP'] - d['BB_LOW']) / d['BB_MID']

    # ATR components
    d['H-L'] = high - low
    d['H-PC'] = (high - close.shift(1)).abs()
    d['L-PC'] = (low - close.shift(1)).abs()
    d['TR'] = d[['H-L','H-PC','L-PC']].max(axis=1)
    d['ATR'] = d['TR'].rolling(10).mean().fillna(method='bfill')

    # SuperTrend (simple approach)
    m = 3.0
    d['BASIC_UB'] = (high + low) / 2.0 + m * d['ATR']
    d['BASIC_LB'] = (high + low) / 2.0 - m * d['ATR']
    d['FINAL_UB'] = d['BASIC_UB'].copy()
    d['FINAL_LB'] = d['BASIC_LB'].copy()
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

    # EMA stacks
    d['EMA20'] = close.ewm(span=20, adjust=False).mean()
    d['EMA50'] = close.ewm(span=50, adjust=False).mean()
    d['EMA200'] = close.ewm(span=200, adjust=False).mean()

    if vol is not None:
        d['VOL_MA20'] = vol.rolling(20).mean()
        d['VOL_SPIKE'] = vol >= 1.5 * d['VOL_MA20']
    else:
        d['VOL_MA20'] = np.nan
        d['VOL_SPIKE'] = False

    return d

# Candlestick detector (advanced)
def detect_candlestick_patterns(df):
    patterns = []
    strength = 0
    if df is None or df.empty or len(df) < 2:
        return patterns, strength
    prev = df.iloc[-2]
    cur = df.iloc[-1]
    # map names flexibly
    lower = {c.lower(): c for c in df.columns}
    o1 = float(prev[lower.get('open','Open')])
    h1 = float(prev[lower.get('high','High')])
    l1 = float(prev[lower.get('low','Low')])
    c1 = float(prev[lower.get('close','Close')])
    o2 = float(cur[lower.get('open','Open')])
    h2 = float(cur[lower.get('high','High')])
    l2 = float(cur[lower.get('low','Low')])
    c2 = float(cur[lower.get('close','Close')])

    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    rng2 = max(h2 - l2, 1e-9)

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
    # Morning star (simple 3-candle)
    if len(df) >= 3:
        p2 = df.iloc[-3]
        p2o = float(p2[lower.get('open','Open')]); p2c = float(p2[lower.get('close','Close')])
        if (p2c < p2o) and (c1 < o1) and (c2 > o2) and (c2 >= (p2o + p2c)/2):
            patterns.append("Morning Star"); strength += 3

    return patterns, strength

# -------------------------
# fetch data (yfinance)
# -------------------------
@st.cache_data(ttl=300)
def fetch_history(symbol: str, period: str="30d", interval: str="15m"):
    """
    Returns (df, error) where df columns normalized (Open/High/Low/Close/Volume if present)
    """
    try:
        ticker = f"{symbol}.NS"
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    except Exception as e:
        return pd.DataFrame(), str(e)
    if df is None or df.empty:
        return pd.DataFrame(), None

    # Normalize columns to Titlecase
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc.startswith("open"): col_map[c] = "Open"
        if lc.startswith("high"): col_map[c] = "High"
        if lc.startswith("low"): col_map[c] = "Low"
        if lc.startswith("close"): col_map[c] = "Close"
        if lc.startswith("volume"): col_map[c] = "Volume"
    if col_map:
        df = df.rename(columns=col_map)
    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df, None

# -------------------------
# Position sizing (from super bot)
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
# Zerodha live helpers
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
    Long-only enforced here too.
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
# place_trade (paper/live) - long-only enforced
# -------------------------
def next_trade_id():
    tid = f"T{st.session_state.trade_counter:06d}"
    st.session_state.trade_counter += 1
    return tid

def place_trade(symbol, side, entry_price, sl, tgt, qty, tsl_pct=0.0, live_trading=False, long_only=True):
    # Enforce long-only for entries
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

    # Save open position
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

    # Log trade open
    st.session_state.trade_history.append({
        'TradeID': trade_id, 'OpenTime': nowts, 'Symbol': symbol, 'Side': side,
        'Qty': int(filled_qty), 'Entry': float(filled_price), 'Exit': None, 'PnL': 0.0, 'Mode': mode, 'Note': note
    })
    st.success(f"Opened {mode} position {symbol} | {side} | Qty {filled_qty} @ ₹{filled_price:.2f}")

# -------------------------
# Exit / trailing check
# -------------------------
def check_exits_with_price(sym, price, live_trading=False):
    positions = st.session_state.open_positions
    if sym not in positions:
        return
    pos = positions[sym]
    side = pos['side']; entry = pos['entry']; stop = pos['stop']; target = pos['target']; qty = pos['qty']; mode = pos.get('mode','PAPER')
    tsl_pct = pos.get('tsl_pct', 0.0)
    exit_reason = None; exit_price = None

    # trailing update (for buys)
    if tsl_pct and tsl_pct > 0 and side == 'BUY':
        new_tsl = price * (1 - tsl_pct/100.0)
        if new_tsl > (pos.get('stop') or 0):
            pos['stop'] = new_tsl
            st.session_state.open_positions[sym]['stop'] = new_tsl
            st.info(f"TSL for {sym} updated to ₹{new_tsl:.2f}")

    # check
    if side == 'BUY':
        if stop is not None and price <= stop:
            exit_price = stop; exit_reason = 'Stop Loss'
        elif target is not None and price >= target:
            exit_price = target; exit_reason = 'Target Hit'

    if exit_reason and exit_price is not None:
        pnl = (exit_price - entry) * qty
        st.session_state.trade_history.append({
            'TradeID': pos.get('id', f"{sym}_{datetime.now().strftime('%H%M%S')}"),
            'Symbol': sym,
            'Side': side,
            'Qty': qty,
            'Entry': entry,
            'Exit': exit_price,
            'PnL': float(pnl),
            'Reason': exit_reason,
            'CloseTime': now_ist_str(),
            'Mode': mode,
            'OrderID': pos.get('order_id')
        })
        # Live exit if live
        if mode == 'LIVE' and live_trading and st.session_state.get('kite'):
            try:
                kite = st.session_state.kite
                txn_type = kite.TRANSACTION_TYPE_SELL if side == 'BUY' else kite.TRANSACTION_TYPE_BUY
                kite.place_order(
                    variety=kite.VARIETY_REGULAR,
                    exchange=kite.EXCHANGE_NSE,
                    tradingsymbol=sym,
                    transaction_type=txn_type,
                    quantity=int(qty),
                    order_type=kite.ORDER_TYPE_MARKET,
                    product=kite.PRODUCT_MIS
                )
                st.info(f"LIVE automated exit order sent for {sym} | {exit_reason}")
            except Exception as e:
                st.error(f"Failed live exit for {sym}: {e}")
        # remove
        del st.session_state.open_positions[sym]
        st.success(f"Exited {sym} | {side} | {exit_reason} | PnL: ₹{pnl:.2f}")

# -------------------------
# Watchlist filter (from super bot)
# -------------------------
def get_filtered_watchlist(df_dict):
    watchlist = []
    for symbol, df in df_dict.items():
        try:
            if len(df) < 50:
                continue
            latest = df.iloc[-1]; prev = df.iloc[-2]
            # volume spike
            avg_vol = df['Volume'].rolling(20).mean().iloc[-2] if 'Volume' in df.columns else np.nan
            if 'Volume' in df.columns and not np.isnan(avg_vol) and latest['Volume'] < 1.5 * avg_vol:
                continue
            # trend: price above EMA20 & EMA50 OR both below (we only care buys so require above)
            if not (latest.get('Close',0) > df['EMA20'].iloc[-1] and latest.get('Close',0) > df['EMA50'].iloc[-1]):
                continue
            # BB width filter (as percent)
            bb_w = ((df['BB_UP'].iloc[-1] - df['BB_LOW'].iloc[-1]) / latest.get('Close',1)) * 100 if 'BB_UP' in df.columns and 'BB_LOW' in df.columns else 100
            if bb_w < 2:
                continue
            # gap filter
            gap_pct = abs((latest.get('Open',0) - prev.get('Close',0)) / prev.get('Close',1)) * 100
            if gap_pct > 2:
                continue
            watchlist.append(symbol)
        except Exception:
            continue
    return watchlist

# -------------------------
# Strategy analyze & trade (hybrid long-only)
# -------------------------
def analyze_and_trade(symbol, df, live_trading=False, settings=None):
    """
    Single unified hybrid strategy; only places BUY entries.
    settings: dict with keys risk_pct, max_cap_alloc_pct, tsl_pct, strength_boost, use_sentiment
    """
    if settings is None:
        settings = {}
    # Portfolio gates
    starting_cap = float(st.session_state.get('starting_capital', 100000.0))
    total_pnl = sum([float(t.get('PnL', 0.0)) for t in st.session_state.get('trade_history', [])])
    daily_loss_limit = starting_cap * (float(st.session_state.get('max_daily_loss_pct', 0.02)) / 1.0) if False else starting_cap * 0.02  # default 2%
    if total_pnl < -daily_loss_limit:
        st.warning("Daily loss limit reached; skipping entries.")
        return

    if len(st.session_state.open_positions) >= int(st.session_state.get('max_open_positions', 10)):
        st.info("Max open positions reached; skipping entries.")
        return

    # compute indicators
    try:
        d = calculate_indicators_super(df)
    except Exception as e:
        st.error(f"Indicator calc failed for {symbol}: {e}")
        return

    # patterns & strength
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
    # sentiment
    senti = 0.0
    if settings.get('use_sentiment'):
        senti = get_sentiment_for_stock(symbol)
        if senti > 0.1: score += 3

    # require minimum score to qualify
    qualifies = False
    if score >= 30 and ("Bullish Engulfing" in patterns or "Hammer" in patterns or pat_strength >= 2):
        if rsi_ok and macd_ok and super_up:
            qualifies = True

    if not qualifies:
        return

    # sizing
    capital = float(st.session_state.get('starting_capital', 100000.0))
    risk_pct = float(settings.get('risk_pct', st.session_state.get('risk_pct', 1.0)))
    max_alloc = float(settings.get('max_cap_alloc_pct', st.session_state.get('max_cap_per_trade',25)))
    entry_price = float(last['Close'])
    stop_price = float(last['Low'] * 0.995) if last['Low'] < entry_price else float(last['Low'])
    raw_qty = calculate_qty(capital, entry_price, stop_price, risk_pct, max_alloc)
    if raw_qty <= 0:
        st.info(f"Qty 0 for {symbol}")
        return
    qty = raw_qty
    # strength boost
    if score >= 60:
        boost = float(settings.get('strength_boost', st.session_state.get('strength_boost',25)))
        qty = int(min(qty * (1.0 + boost/100.0), int((capital * max_alloc/100.0) // entry_price)))

    if qty <= 0:
        st.info(f"Final qty 0 for {symbol}")
        return

    # target & sl
    stop_loss = stop_price
    tgt = round(entry_price * 1.02, 2)  # 2% target default; tweakable
    tsl_pct = float(settings.get('tsl_pct', st.session_state.get('tsl_pct', 0.0)))

    # place trade (long-only enforced)
    st.info(f"Placing {'LIVE' if live_trading else 'PAPER'} BUY {symbol} | Qty={qty} | Entry={entry_price:.2f} | SL={stop_loss:.2f} | TGT={tgt:.2f}")
    place_trade(symbol, "BUY", entry_price, stop_loss, tgt, qty, tsl_pct, live_trading=live_trading, long_only=True)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="⚡ Ultra Hybrid Long-Only Bot", layout="wide")
st.title("⚡ Ultra Hybrid Long-Only Bot — Merged (Advance + Super)")

# Sidebar
with st.sidebar:
    st.header("Scanner & Risk Settings")
    mode = st.radio("Scan mode", ["Auto (filtered NIFTY50)", "Manual selection"], index=0)
    if mode.startswith("Manual"):
        selected_symbols = st.multiselect("Symbols (comma)", options=NIFTY50, default=["RELIANCE","TCS","INFY"])
    else:
        selected_symbols = []  # will be filled by auto-scan

    interval = st.selectbox("Interval", ["5m","15m","30m","60m"], index=1)
    period = st.selectbox("Period", ["7d","14d","30d","60d"], index=2)
    vol_multiplier = st.number_input("Volume Spike Multiplier (× Vol MA20)", min_value=1.0, value=1.5, step=0.1)
    bb_width_min = st.slider("Min Bollinger Width (%)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)/100.0
    gap_avoid_pct = st.slider("Avoid gap ≥ (percent)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)/100.0

    st.markdown("---")
    st.subheader("Risk / Money Mgmt")
    st.session_state.starting_capital = st.number_input("Starting capital (₹)", value=float(st.session_state.starting_capital), step=1000.0)
    st.session_state.risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    st.session_state.max_cap_per_trade = st.slider("Max capital per trade (%)", 5, 50, 25, step=5)
    st.session_state.max_open_positions = st.number_input("Max open positions", min_value=1, max_value=50, value=10)
    st.session_state.max_daily_loss_pct = st.number_input("Daily loss limit (%)", min_value=0.5, max_value=50.0, value=2.0, step=0.5)
    st.session_state.tsl_pct = st.number_input("Trailing Stop-Loss (%)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    st.session_state.strength_boost = st.slider("Strength boost (%)", 0, 100, 25, step=5)

    st.markdown("---")
    st.subheader("News & Alerts")
    news_api_key = st.text_input("NewsAPI key (optional)", type="password")
    use_sentiment = st.checkbox("Use sentiment weighting", value=False)
    telegram_token = st.text_input("Telegram token (optional)", type="password")
    telegram_chat_id = st.text_input("Telegram chat id (optional)")

    st.markdown("---")
    st.subheader("LIVE Zerodha")
    live_trading = st.checkbox("Enable LIVE trading (careful!)", value=False)
    long_only_toggle = True  # forced long-only; UI shows status
    st.write("Long-only mode is ON (no SELL entries).")
    st.markdown("---")
    st.subheader("Zerodha Connect (optional)")
    Z_API_KEY = st.text_input("Kite API Key (optional)", type="password")
    Z_API_SECRET = st.text_input("Kite API Secret (optional)", type="password")
    REQUEST_TOKEN = st.text_input("Request Token (if needed today)")
    if st.button("Create Access Token / Connect") or (live_trading and REQUEST_TOKEN):
        if KiteConnect is None:
            st.error("kiteconnect library not installed.")
        else:
            API_KEY = os.getenv("KITE_API_KEY") or st.secrets.get("KITE_API_KEY", "") or Z_API_KEY
            API_SECRET = os.getenv("KITE_API_SECRET") or st.secrets.get("KITE_API_SECRET", "") or Z_API_SECRET
            if not API_KEY or not API_SECRET or not REQUEST_TOKEN:
                st.warning("Provide API_KEY, API_SECRET and Request Token (from Kite login flow).")
            else:
                try:
                    kite = KiteConnect(api_key=API_KEY)
                    data = kite.generate_session(REQUEST_TOKEN, api_secret=API_SECRET)
                    kite.set_access_token(data["access_token"])
                    st.session_state.kite = kite
                    save_local_token(data["access_token"])
                    st.success("Connected to Zerodha and token saved locally.")
                except Exception as e:
                    st.error(f"Zerodha login failed: {e}")

# If live and we have saved token, try auto connect
if live_trading and KiteConnect is not None and st.session_state.get('kite') is None:
    token_data = load_local_token()
    API_KEY = os.getenv("KITE_API_KEY") or st.secrets.get("KITE_API_KEY","")
    if token_valid(token_data) and API_KEY:
        try:
            kite = KiteConnect(api_key=API_KEY)
            kite.set_access_token(token_data["access_token"])
            st.session_state.kite = kite
            st.success("Auto-connected to Zerodha using saved token.")
        except Exception:
            pass

# Main buttons
col1, col2 = st.columns([1,1])
with col1:
    if st.button("Run Scan & Analyze"):
        # sync manual exits first
        if live_trading and st.session_state.get('kite'):
            sync_zerodha_positions()
        # fetch histories
        df_dict = {}
        symbols_to_scan = []
        if mode.startswith("Auto"):
            # scan NIFTY50
            symbols_to_scan = NIFTY50
        else:
            symbols_to_scan = selected_symbols

        progress = st.progress(0)
        total = len(symbols_to_scan)
        i = 0
        for sym in symbols_to_scan:
            df_hist, err = fetch_history(sym, period=period, interval=interval)
            i += 1
            progress.progress(int(i/total*100))
            if err or df_hist.empty:
                continue
            # attach indicators used for filtering (so get_filtered_watchlist works)
            try:
                df_ind = calculate_indicators_super(df_hist)
            except Exception:
                continue
            df_dict[sym] = df_ind

        # filter
        watch = get_filtered_watchlist(df_dict)
        st.success(f"Filtered watchlist ({len(watch)}): {watch}")
        # analyze each
        settings = {
            'risk_pct': st.session_state.get('risk_pct',1.0),
            'max_cap_alloc_pct': st.session_state.get('max_cap_per_trade',25),
            'tsl_pct': st.session_state.get('tsl_pct',0.0),
            'strength_boost': st.session_state.get('strength_boost',25),
            'use_sentiment': use_sentiment
        }
        for sym in watch:
            try:
                analyze_and_trade(sym, df_dict[sym], live_trading=live_trading, settings=settings)
            except Exception as e:
                st.error(f"Error analyzing {sym}: {e}")

with col2:
    if st.button("Sync Zerodha Positions"):
        sync_zerodha_positions()

# Display results area
st.markdown("---")
st.subheader("Scanner Signals (recent)")
if st.session_state.scanner_signals:
    df_signals = pd.DataFrame(st.session_state.scanner_signals, columns=["Time","Symbol","Signal","Price","Stop","Target"]).tail(200)
    st.dataframe(df_signals, use_container_width=True)
else:
    st.info("No scanner signals yet (run a scan).")

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
    history_df = pd.DataFrame(st.session_state.trade_history).tail(200)
    st.dataframe(history_df, use_container_width=True)
    if st.button("Export trade history to CSV"):
        history_df.to_csv(STATE_DB, index=False)
        st.success(f"Saved to {STATE_DB}")
else:
    st.info("No trade history yet.")

# Live update loop: refresh LTP and check exits if live
if live_trading and st.session_state.get('kite'):
    try:
        kite = st.session_state.kite
        # iterate open positions and check exits with fresh LTP
        for sym, pos in list(st.session_state.open_positions.items()):
            if pos.get('mode') == 'LIVE':
                ltp = get_live_price(kite, sym)
                if ltp is not None:
                    st.session_state.open_positions[sym]['LTP'] = ltp
                    check_exits_with_price(sym, ltp, live_trading=True)
    except Exception as e:
        st.warning(f"Live update error: {e}")

st.caption("Ultra Hybrid Long-Only Bot — merged from Advance + Super bots. Review live order code before using real money.")
