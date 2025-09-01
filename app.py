# merged_zerodha_bot_longonly.py (with Candlestick Patterns + existing features)

from datetime import datetime, time
import pytz
import streamlit as st
import json
import os

TOKEN_FILE = "token.json"

def save_token(request_token):
    with open(TOKEN_FILE, "w") as f:
        json.dump({"request_token": request_token}, f)

def load_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as f:
            data = json.load(f)
            return data.get("request_token", "")
    return ""
import yfinance as yf
import pandas as pd
import numpy as np
import ta.momentum as ta_momentum
import ta.trend as ta_trend

# Optional imports (news + sentiment)
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

# Optional autorefresh (kept exactly as you had)
try:
    from streamlit_autorefresh import st_autorefresh
    count = st_autorefresh(interval=600_000, limit=None, key="news_refresh")
except Exception:
    pass

# App config
st.set_page_config(page_title="NSE Intraday Scanner — Merged Bot (Paper + Live)", layout="wide")
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

# ------------------ Defaults ------------------
NIFTY50 = sorted(list({
    "ADANIENT","ASIANPAINT","PGEL","AXISBANK","BAJAJ-AUTO","BAJFINANCE","BAJAJFINSV","BPCL","BHARTIARTL","INFRATEL","BRITANNIA",
    "CIPLA","COALINDIA","DIVISLAB","DRREDDY","EICHERMOT","GRASIM","HCLTECH","HDFC","HDFCBANK","HDFCLIFE",
    "HINDALCO","HINDUNILVR","ICICIBANK","ITC","INDUSINDBK","INFY","JSWSTEEL","KOTAKBANK","LT",
    "M&M","MARUTI","NTPC","NESTLEIND","ONGC","POWERGRID","RELIANCE","SBIN","SUNPHARMA","TATASTEEL",
    "TCS","TECHM","TITAN","ULTRACEMCO","UPL","WIPRO","HINDPETRO","SHREECEM","M&MFIN","PEL"
}))

# ------------------ Sentiment helpers (optimized caching) ------------------
if NewsApiClient is not None and SentimentIntensityAnalyzer is not None:
    # ⚠️ Put your own key in env or secrets; hardcoding is only for quick tests.
    newsapi = NewsApiClient(api_key='1f856b901f48461580ebf08a7c9745ee')  # optional
    analyzer = SentimentIntensityAnalyzer()

    @st.cache_data(ttl=3600)
    def get_sentiment_for_stock(stock_name):
        """Fetch sentiment (cached 1 hour)"""
        try:
            all_articles = newsapi.get_everything(q=stock_name, language='en', sort_by='relevancy', page_size=20)
            headlines = [article['title'] for article in all_articles['articles']]
            if not headlines:
                return 0.0
            scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
            return sum(scores) / len(scores)
        except Exception:
            return 0.0

    @st.cache_data(ttl=600)
    def scrape_moneycontrol_news(keyword, max_headlines=5):
        """Scrape headlines (cached 10 min)"""
        url = "https://www.moneycontrol.com/news/markets/"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.find_all("a", class_="clearfix", limit=50)
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
    def get_sentiment_for_stock(stock_name): return 0.0
    def scrape_moneycontrol_news(keyword, max_headlines=5): return []

# ------------------ Indicators ------------------
def sma(series, window):
    return series.rolling(window).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=(period-1), adjust=False).mean()
    ma_down = down.ewm(com=(period-1), adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def macd(series, n_fast=12, n_slow=26, n_signal=9):
    ema_fast = series.ewm(span=n_fast, adjust=False).mean()
    ema_slow = series.ewm(span=n_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=n_signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# ------------------ Candlestick patterns (lightweight rules) ------------------
def _body(o, c): return abs(c - o)
def _upper(o, c, h): return h - max(o, c)
def _lower(o, c, l): return min(o, c) - l

def detect_last_candle_patterns(df):
    """
    Returns a comma-separated string of patterns on the LAST bar.
    Patterns: Doji, Hammer, Inverted Hammer, Shooting Star,
              Bullish/Bearish Engulfing, Morning/Evening Star (simplified).
    """
    if df is None or df.empty: return ""
    if not all(col in df.columns for col in ["Open", "High", "Low", "Close"]): return ""

    pats = []
    if len(df) < 2:
        row = df.iloc[-1]
        o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
        hl = max(h - l, 1e-9); body = _body(o, c)
        # Doji
        if body <= 0.1 * hl: pats.append("Doji")
        return ", ".join(pats)

    prev = df.iloc[-2]
    row  = df.iloc[-1]
    po, ph, pl, pc = float(prev["Open"]), float(prev["High"]), float(prev["Low"]), float(prev["Close"])
    o,  h,  l,  c  = float(row["Open"]),  float(row["High"]),  float(row["Low"]),  float(row["Close"])

    hl = max(h - l, 1e-9); body = _body(o, c)
    upper = _upper(o, c, h); lower = _lower(o, c, l)

    # Doji
    if body <= 0.1 * hl: pats.append("Doji")

    # Hammer / Inverted Hammer (require small body, long opposite wick)
    if body <= 0.35 * hl:
        if lower >= 2 * body and upper <= body: pats.append("Hammer")
        if upper >= 2 * body and lower <= body: pats.append("Inverted Hammer")

    # Shooting Star (uptrend-ish not enforced strictly)
    if upper >= 2 * body and lower <= 0.3 * body and c < o:
        pats.append("Shooting Star")

    # Engulfing (strict body engulfing)
    prev_bear = pc < po
    prev_bull = pc > po
    curr_bull = c > o
    curr_bear = c < o
    if curr_bull and prev_bear and (o <= pc) and (c >= po) and body > (_body(po, pc) * 0.9):
        pats.append("Bullish Engulfing")
    if curr_bear and prev_bull and (o >= pc) and (c <= po) and body > (_body(po, pc) * 0.9):
        pats.append("Bearish Engulfing")

    # Morning/Evening Star (simplified 3-candle check)
    if len(df) >= 3:
        p2 = df.iloc[-3]
        p2o, p2c = float(p2["Open"]), float(p2["Close"])
        # Morning Star: down, small gap-ish middle, strong up
        if (p2c < p2o) and (pc < po) and (c > o) and (c >= (p2o + p2c) / 2):
            pats.append("Morning Star")
        # Evening Star: up, small gap-ish middle, strong down
        if (p2c > p2o) and (pc > po) and (c < o) and (c <= (p2o + p2c) / 2):
            pats.append("Evening Star")

    return ", ".join(dict.fromkeys(pats))  # unique & keep order

def recent_patterns_table(df, n=15):
    """
    Build a small table (last n bars) with pattern labels, if any.
    """
    if df is None or df.empty or not all(k in df.columns for k in ["Open","High","Low","Close"]):
        return pd.DataFrame()
    d = df.copy().tail(n + 2)  # extra rows for engulfing/star logic
    out = []
    for i in range(2, len(d)):
        sub = d.iloc[: i+1]
        label = detect_last_candle_patterns(sub)
        ts = d.index[i]
        out.append({"Time": ts, "Close": float(d["Close"].iloc[i]), "Pattern": label})
    return pd.DataFrame(out).set_index("Time")

# ------------------ Data helpers (Corrected) ------------------
@st.cache_data(ttl=300)  # cache data for 5 min to avoid refetching
def fetch_intraday(symbol, period, interval):
    try:
        df = yf.download(f"{symbol}.NS", period=period, interval=interval, progress=False, auto_adjust=False)
    except Exception as e:
        return pd.DataFrame(), str(e)
    if df.empty:
        return pd.DataFrame(), None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns.values]
    df.columns = [str(c).split(' ')[0] for c in df.columns]
    # Normalize Close name
    close_cols = [c for c in df.columns if c.lower().startswith("close")]
    if close_cols and close_cols[0] != "Close":
        df = df.rename(columns={close_cols[0]: "Close"})
    return df, None

# ------------------ Feature engineering ------------------
def compute_features(df, vol_multiplier=1.5, sma_short=20, sma_long=50):
    if df.empty or 'Close' not in df.columns:
        return df
    df = df.copy()
    # Corrected MACD, RSI, and SMA with .squeeze() for compatibility
    df['SMA_short'] = sma(df['Close'].squeeze(), sma_short)
    df['SMA_long']  = sma(df['Close'].squeeze(), sma_long)
    df['RSI14']     = rsi(df['Close'].squeeze(), 14)
    macd_line, macd_signal, macd_hist = macd(df['Close'].squeeze(), 12, 26, 9)

    df['MACD']          = macd_line
    df['MACD_SIGNAL']   = macd_signal
    df['MACD_HIST']     = macd_hist
    if 'Volume' in df.columns:
        df['Vol_MA20']  = df['Volume'].rolling(20).mean()
        # Corrected Vol_Spike with .align() to prevent misalignment errors
        left, right = df['Volume'].align(df['Vol_MA20'], join='left', axis=0)
        df['Vol_Spike'] = left > vol_multiplier * right
    else:
        df['Vol_MA20']  = np.nan
        df['Vol_Spike'] = False
    return df

# MODIFIED: vote_signal function to include sentiment and more advanced patterns
def vote_signal(df, require_all=True, vol_multiplier=1.5, daily_sentiment=0.0, sentiment_threshold=0.1):
    if df.empty or len(df) < 2:
        return "HOLD"
    prev, last = df.iloc[-2], df.iloc[-1]
    
    # Existing conditions
    try:
        sma_bull = (prev['SMA_short'] <= prev['SMA_long']) and (last['SMA_short'] > last['SMA_long'])
        sma_bear = (prev['SMA_short'] >= prev['SMA_long']) and (last['SMA_short'] < last['SMA_long'])
    except Exception:
        sma_bull = sma_bear = False
    macd_bull       = bool(last['MACD_HIST'] > 0) if pd.notna(last['MACD_HIST']) else False
    macd_bear       = bool(last['MACD_HIST'] < 0) if pd.notna(last['MACD_HIST']) else False
    rsi_ok_buy    = bool(last['RSI14'] < 70)      if pd.notna(last['RSI14']) else False
    rsi_ok_sell = bool(last['RSI14'] > 30)      if pd.notna(last['RSI14']) else False
    vol_ok = False
    if 'Volume' in df.columns and pd.notna(last.get('Vol_MA20', np.nan)):
        vol_ok = last['Volume'].squeeze() >= vol_multiplier * last['Vol_MA20'].squeeze()

    # NEW: AI Integration - Sentiment Analysis as a vote
    sentiment_bull = bool(daily_sentiment > sentiment_threshold)
    sentiment_bear = bool(daily_sentiment < -sentiment_threshold)

    # NEW: Advanced Pattern Recognition - Golden Cross + RSI Momentum
    # This requires daily data, so we re-fetch to be safe
    try:
        daily_df, _ = fetch_intraday(df.name, "200d", "1d")
        daily_df['SMA50'] = sma(daily_df['Close'], 50)
        daily_df['SMA200'] = sma(daily_df['Close'], 200)
        golden_cross = (daily_df['SMA50'].iloc[-2] <= daily_df['SMA200'].iloc[-2]) and \
                       (daily_df['SMA50'].iloc[-1] > daily_df['SMA200'].iloc[-1]) and \
                       (daily_df['RSI14'].iloc[-1] > 60)
        death_cross = (daily_df['SMA50'].iloc[-2] >= daily_df['SMA200'].iloc[-2]) and \
                      (daily_df['SMA50'].iloc[-1] < daily_df['SMA200'].iloc[-1]) and \
                      (daily_df['RSI14'].iloc[-1] < 40)
    except Exception:
        golden_cross = False
        death_cross = False

    buy_conditions  = [sma_bull, macd_bull, rsi_ok_buy, vol_ok, sentiment_bull, golden_cross]
    sell_conditions = [sma_bear, macd_bear, rsi_ok_sell, vol_ok, sentiment_bear, death_cross]
    
    buy_votes = sum(buy_conditions)
    sell_votes = sum(sell_conditions)

    if require_all:
        if all(buy_conditions):
            return "BUY"
        if all(sell_conditions):
            return "SELL"
    else: # Majority vote
        if buy_votes >= 3:
            return f"BUY ({buy_votes} votes)"
        if sell_votes >= 3:
            return f"SELL ({sell_votes} votes)"

    if sma_bull and (macd_bull or rsi_ok_buy): return "HOLD (Weak BUY)"
    if sma_bear and (macd_bear or rsi_ok_sell): return "HOLD (Weak SELL)"
    return "HOLD"


def generate_dynamic_signal(symbol):
    df, err = fetch_intraday(symbol, period="15d", interval="1d")
    if df.empty:
        return {"Symbol": symbol, "Signal": "HOLD", "Price": np.nan, "Stop_Loss": None, "Target": None}

    last_price = df['Close'].iloc[-1]
    signal = vote_signal(compute_features(df))


    if signal == "BUY":
        sl = round(last_price * 0.98, 2)
        tgt = round(last_price * 1.03, 2)
    elif signal == "SELL":
        sl = round(last_price * 1.02, 2)
        tgt = round(last_price * 0.97, 2)
    else:
        sl = tgt = None

    return {
        "Symbol": symbol,
        "Signal": signal,
        "Price": round(last_price, 2),
        "Stop_Loss": sl,
        "Target": tgt
    }

# ------------------ Dynamic Scanner helpers ------------------
def dynamic_volume_spike(symbol, period="15d", interval="1d", multiplier=1.5):
    df, err = fetch_intraday(symbol, period, interval)
    if df.empty: return False
    df = compute_features(df, vol_multiplier=multiplier)
    return df['Vol_Spike'].iloc[-1] if 'Vol_Spike' in df.columns else False

def dynamic_uptrend(symbol, period="15d", interval="1d", sma_short=5, sma_long=10):
    df, err = fetch_intraday(symbol, period, interval)
    if df.empty: return False
    # Corrected SMA calls with .squeeze()
    df['SMA_short'] = sma(df['Close'].squeeze(), sma_short)
    df['SMA_long']  = sma(df['Close'].squeeze(), sma_long)
    last = df.iloc[-1]
    return last['SMA_short'] > last['SMA_long']

def dynamic_no_large_gap(symbol):
    df, err = fetch_intraday(symbol, period="2d", interval="1d")
    if df.empty or len(df) < 2: return False
    gap = abs(df['Open'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]
    return gap < 0.02

def scan_strong_stocks(stock_list):
    strong_stocks = []
    for s in stock_list:
        try:
            if dynamic_volume_spike(s) and dynamic_uptrend(s) and dynamic_no_large_gap(s):
                strong_stocks.append(s)
        except:
            continue
    return strong_stocks

# ------------------ State & PnL ------------------
def ensure_state():
    if 'open_positions'      not in st.session_state: st.session_state.open_positions = {}
    if 'trade_history'       not in st.session_state: st.session_state.trade_history = []
    if 'starting_capital' not in st.session_state: st.session_state.starting_capital = 100000.0
    if 'trade_counter'       not in st.session_state: st.session_state.trade_counter = 1
    if 'orders'              not in st.session_state: st.session_state.orders = []

def next_trade_id():
    tid = f"T{st.session_state.trade_counter:06d}"
    st.session_state.trade_counter += 1
    return tid

def close_paper_trade(symbol, exit_price, reason):
    if symbol not in st.session_state.open_positions:
        return
    pos   = st.session_state.open_positions.pop(symbol)
    side  = pos['side']
    qty   = pos['qty']
    entry = pos['entry']
    pnl   = (exit_price - entry)*qty if side=='BUY' else (entry - exit_price)*qty
    st.session_state.trade_history.append({
        'TradeID': pos.get('id',''),
        'OpenTime': pos['time'],
        'CloseTime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Symbol': symbol,
        'Side': side,
        'Qty': qty,
        'Entry': entry,
        'Exit': float(exit_price),
        'PnL': float(pnl),
        'Reason': reason,
        'Mode': pos.get('mode','PAPER'),
        'OrderID': pos.get('order_id')
    })

# ------------------ NEW: Zerodha Live Sync & Data Helpers ------------------

def get_live_price(kite, symbol):
    """
    NEW: Fetch real-time Last Traded Price (LTP) from Zerodha.
    """
    try:
        # Construct the instrument token in the format "NSE:SYMBOL"
        instrument = f"NSE:{symbol}"
        ltp_data = kite.ltp(instrument)
        if instrument in ltp_data:
            return ltp_data[instrument]['last_price']
    except Exception as e:
        st.error(f"Failed to fetch live price for {symbol}: {e}")
    return None

def place_real_order_with_zerodha(kite, symbol, side, qty):
    """
    MODIFIED: Places a market MIS order and waits for the order to be filled
    to get the actual filled price and quantity.
    Returns a dict with filled info or None.
    """
    try:
        txn = kite.TRANSACTION_TYPE_BUY if side == 'BUY' else kite.TRANSACTION_TYPE_SELL
        # Place the order
        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NSE,
            tradingsymbol=symbol,
            transaction_type=txn,
            quantity=int(qty),
            order_type=kite.ORDER_TYPE_MARKET,
            product=kite.PRODUCT_MIS
        )
        st.info(f"Order placed for {symbol}. Order ID: {order_id}. Waiting for fill status...")

        # Wait for the order to get filled and fetch details
        import time
        for i in range(10): # Check up to 10 times (e.g., 20 seconds)
            order_details = kite.orders().get(order_id)
            if order_details and order_details['status'] == 'COMPLETE':
                filled_qty = order_details['filled_quantity']
                avg_price = order_details['average_price']
                st.success(f"Order for {symbol} filled! Qty: {filled_qty}, Price: {avg_price:.2f}")
                return {'order_id': order_id, 'filled_qty': filled_qty, 'avg_price': avg_price}
            time.sleep(2)

        st.warning(f"Order {order_id} for {symbol} not filled within time.")
        return None

    except Exception as e:
        st.error(f"Zerodha order failed for {symbol}: {e}")
        return None

def sync_zerodha_positions():
    """
    NEW: Syncs the bot's internal positions with live positions on Zerodha.
    This detects manual square-offs.
    """
    if not ('kite' in st.session_state and is_market_open):
        st.warning("Not connected to Zerodha or market is closed. Cannot sync live positions.")
        return

    kite = st.session_state.kite
    try:
        live_positions = kite.positions().get('day', [])
        live_symbols = {p['tradingsymbol'] for p in live_positions if p['product'] == 'MIS'}

        bot_positions = list(st.session_state.open_positions.keys())
        for sym in bot_positions:
            pos = st.session_state.open_positions[sym]
            if pos['mode'] == 'LIVE' and sym not in live_symbols:
                # This position was open in the bot but is no longer on Zerodha
                st.warning(f"Detected manual exit for {sym}. Syncing...")
                
                # We can't get the manual exit price easily, so we use the last known price
                live_price = get_live_price(kite, sym)
                if live_price is not None:
                    pnl = (live_price - pos['entry']) * pos['qty'] if pos['side'] == 'BUY' else (pos['entry'] - live_price) * pos['qty']
                    
                    st.session_state.trade_history.append({
                        'TradeID': pos.get('id', 'N/A'),
                        'OpenTime': pos['time'],
                        'CloseTime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Symbol': sym,
                        'Side': pos['side'],
                        'Qty': pos['qty'],
                        'Entry': pos['entry'],
                        'Exit': live_price,
                        'PnL': float(pnl),
                        'Reason': 'Manual Exit (Zerodha)',
                        'Mode': 'LIVE',
                        'OrderID': pos.get('order_id')
                    })
                    del st.session_state.open_positions[sym]
                    st.success(f"Synced manual exit for {sym}. PnL: ₹{pnl:.2f}")

    except Exception as e:
        st.error(f"Zerodha position sync failed: {e}")

# ------------------ Exit Check Function (Auto Live + Paper) ------------------
def check_exits_with_price(sym, price, live_trading=False):
    """
    MODIFIED: Check open positions for this symbol and exit if price hits stop-loss or target.
    Now includes logic for trailing stop-loss.
    """
    if 'open_positions' not in st.session_state:
        st.session_state.open_positions = {}

    positions = st.session_state.open_positions
    if sym not in positions:
        return

    pos       = positions[sym]
    side      = pos['side']
    entry     = pos['entry']
    stop      = pos['stop']
    target    = pos['target']
    qty       = pos['qty']
    mode      = pos.get('mode', 'PAPER')
    order_id = pos.get('order_id')
    tsl_pct   = pos.get('tsl_pct', 0.0) # NEW: Get the trailing stop percentage

    exit_reason = None
    exit_price  = None
    
    # NEW: Trailing Stop-Loss Logic
    # The new_tsl is calculated based on the current price
    if tsl_pct > 0:
        if side == 'BUY':
            new_tsl = price * (1 - tsl_pct / 100)
            if new_tsl > pos['stop']:
                # Update the stop-loss only if it's higher
                pos['stop'] = new_tsl
                st.session_state.open_positions[sym]['stop'] = new_tsl # Update the session state
                st.info(f"📈 TSL for {sym} updated to ₹{new_tsl:.2f}")
        elif side == 'SELL':
            new_tsl = price * (1 + tsl_pct / 100)
            if new_tsl < pos['stop']:
                # Update the stop-loss only if it's lower
                pos['stop'] = new_tsl
                st.session_state.open_positions[sym]['stop'] = new_tsl # Update the session state
                st.info(f"📉 TSL for {sym} updated to ₹{new_tsl:.2f}")

    # Now, check for stop-loss or target hit with the (potentially updated) stop price
    if side == 'BUY':
        if price <= pos['stop']:
            exit_price = pos['stop']
            exit_reason = 'Stop Loss'
        elif price >= target:
            exit_price = target
            exit_reason = 'Target Hit'
    elif side == 'SELL':
        if price >= pos['stop']:
            exit_price = pos['stop']
            exit_reason = 'Stop Loss'
        elif price <= target:
            exit_price = target
            exit_reason = 'Target Hit'

    if exit_reason and exit_price is not None:
        pnl = (exit_price - entry) * qty if side=='BUY' else (entry - exit_price) * qty

        # Record in trade history
        st.session_state.trade_history.append({
            'TradeID': pos.get('id', f"{sym}_{datetime.now().strftime('%H%M%S')}"),
            'Symbol': sym,
            'Side': side,
            'Qty': qty,
            'Entry': entry,
            'Exit': exit_price,
            'PnL': pnl,
            'Reason': exit_reason,
            'CloseTime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Mode': mode,
            'OrderID': order_id
        })

        # Place LIVE exit order if in live trading and a LIVE position
        if mode == 'LIVE' and 'kite' in st.session_state:
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
                st.info(f"LIVE automated exit order sent for {sym} | {side} | {exit_reason}")
            except Exception as e:
                st.error(f"Failed to exit {sym} LIVE: {e}")

        # Remove the position
        del st.session_state.open_positions[sym]
        st.success(f"Exited {sym} | {side} | {exit_reason} | PnL: ₹{pnl:.2f}")


# ------------------ Trade placement ------------------
def place_trade(symbol, side, entry_price, sl, tgt, qty, tsl_pct, live_trading=False):
    trade_id = next_trade_id()
    nowts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filled_price = float(entry_price)
    filled_qty = int(qty)
    order_id = None
    mode = 'PAPER'
    note = 'PAPER-FILLED'

    # LIVE
    if live_trading and ('kite' in st.session_state):
        try:
            order_details = place_real_order_with_zerodha(st.session_state.kite, symbol, side, qty)
            if order_details:
                filled_qty = order_details['filled_qty']
                filled_price = order_details['avg_price']
                order_id = order_details['order_id']
                mode = 'LIVE'
                note = f"ZERODHA-FILLED ({order_id})"
            else:
                st.error(f"Order for {symbol} could not be placed or filled. Not opening a position.")
                return # Exit the function if live order fails to fill
        except Exception as e:
            st.error(f"ERROR: {e}")
            return # Exit the function if live order fails

    st.session_state.open_positions[symbol] = {
        'id': trade_id,
        'time': nowts,
        'symbol': symbol,
        'side': side,
        'entry': float(filled_price),
        'stop': float(sl),
        'target': float(tgt),
        'qty': int(filled_qty),
        'order_id': order_id,
        'mode': mode,
        'note': note,
        'tsl_pct': float(tsl_pct) # NEW: Store the trailing stop percentage
    }

# ------------------ UI ------------------
ensure_state()
st.markdown("<h1 style='text-align:center; color:#0b5394;'>⚡ NIFTY50 Intraday Scanner — Merged Bot</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Scanner Settings")

    # Add a radio button to choose the mode
    scan_mode = st.radio(
        "Choose Scan Mode:",
        ["Auto Scan (Strong Stocks Only)", "Manual Select"]
    )

    # Use an if/else block to handle the selection logic
    if scan_mode == "Auto Scan (Strong Stocks Only)":
        selected = scan_strong_stocks(NIFTY50)
        st.markdown("---")
        st.subheader("Auto-Selected Stocks")
        if selected:
            st.write(selected)
        else:
            st.info("No stocks met the auto-scan criteria.")
    else:
        selected = st.multiselect(
            "Choose stocks to scan (NIFTY50 preloaded):",
            options=NIFTY50,
            default=["TCS", "INFY", "RELIANCE"]
        )
    st.markdown("---")
    interval = st.selectbox("Interval", ["5m", "15m", "30m", "60m"], index=1)
    period = st.selectbox("History period", ["7d", "14d", "30d", "60d"], index=2)
    voting = st.radio("Voting method", ["Strict (all must match)", "Majority (>=3 of 4)"], index=1)
    require_all = True if voting.startswith("Strict") else False
    vol_multiplier = st.number_input("Volume multiplier (× Vol MA20)", min_value=1.0, value=1.5, step=0.1)
    # NEW: AI Integration - Sentiment Analysis setting
    sentiment_threshold = st.number_input(
        "Sentiment Threshold",
        min_value=0.0, max_value=1.0, value=0.1, step=0.01,
        help="Sentiment Score (from -1 to +1) required to be a valid vote. 0.1 means slightly positive news is needed."
    )


    st.markdown("---")
    st.subheader("Risk / Money Mgmt")
    start_cap = st.number_input("Starting capital (₹)", min_value=20000.0, max_value=100000.0, value=100000.0, step=1000.0)
    st.session_state.starting_capital = start_cap
    risk_pct   = st.slider("Risk per trade (%)", 0.1, 5.0, 10.0, 0.1)
    stop_pct   = st.slider("Stop-loss (%)", 0.2, 5.0, 1.0, 0.1)
    target_pct = st.slider("Target (%)", 0.5, 10.0, 3.0, 0.5)
    # NEW: Trailing Stop-Loss setting
    tsl_pct = st.number_input("Trailing Stop-Loss (%)", min_value=0.0, max_value=5.0, value=0.0, step=0.1, help="If > 0, fixed SL is ignored and stop trails price by this percentage.")

    st.caption(f"Position sizing from ₹{int(start_cap)} @ {risk_pct:.1f}% risk, SL {stop_pct:.1f}%, Target {target_pct:.1f}%")

    st.markdown("---")
    auto_sim = st.checkbox("Auto place trades (paper/live)", value=False)
    close_weak = st.checkbox("Ignore weak signals (only strong BUY/SELL)", value=True)
# Mobile UI improvements
    st.markdown('''
    <style>
      .stButton>button {
          padding: 14px 18px; 
          border-radius: 12px; 
          font-size: 16px;
      }
      .stMetric {
          padding: 8px 8px;
      }
      div[data-testid="stSidebar"] {
          width: 280px;
      }
      @media (max-width: 768px){
         .block-container {
             padding-top: 1rem; 
             padding-left: .6rem; 
             padding-right: .6rem;
         }
      }
    </style>
''', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📰 News & Alerts")
    news_api_key = st.text_input("NewsAPI.org Key (optional)", type="password")
    use_sentiment = st.checkbox("Use sentiment weighting", value=False)
    telegram_token = st.text_input("Telegram Bot Token", type="password")
    telegram_chat_id = st.text_input("Telegram Chat ID")

    st.markdown("---")
    st.subheader("🔴 LIVE Zerodha Connection")
    live_trading = st.checkbox("Enable LIVE trading", value=False)

if live_trading:
    st.markdown("""
    **Steps to connect Zerodha LIVE:**
    1. API Key & Secret are securely stored in Streamlit Secrets.
    2. Access Token will auto-refresh if expired (requires entering Request Token once daily).
    3. Click 'Create Access Token' to manually refresh if needed.
    """)

    try:
        from kiteconnect import KiteConnect
        import os, json
        from datetime import datetime, date   # ✅ fixed import

        # --- Load from Streamlit Secrets ---
        API_KEY     = st.secrets.get("API_KEY", "")
        API_SECRET  = st.secrets.get("API_SECRET", "")

        # Local token file (persistent across restarts)
        TOKEN_FILE = "zerodha_token.json"

        # --------------------------
        # Helper functions
        # --------------------------
        def save_local_token(access_token):
            data = {
                "access_token": access_token,
                "date": date.today().isoformat()  # ✅ use date class directly
            }
            with open(TOKEN_FILE, "w") as f:
                json.dump(data, f)

        def load_local_token():
            if os.path.exists(TOKEN_FILE):
                with open(TOKEN_FILE, "r") as f:
                    return json.load(f)
            return None

        def token_valid(token_data):
            if not token_data:
                return False
            # Token is valid only for today
            return token_data.get("date") == date.today().isoformat()  # ✅

        kite = None
        token_data = load_local_token()
        access_token_to_use = token_data["access_token"] if token_valid(token_data) else None

        # Auto-connect if valid token exists
        if access_token_to_use:
            try:
                kite = KiteConnect(api_key=API_KEY)
                kite.set_access_token(access_token_to_use)
                st.session_state.kite = kite
                st.success("✅ Auto-connected using saved Access Token!")
            except Exception as e:
                st.warning(f"Saved access token invalid/expired: {e}")
                access_token_to_use = None

        # Request Token input (needed once daily)
        REQUEST_TOKEN = st.text_input("Request Token (daily, only if needed)")

        # Manual refresh button
        if st.button("Create Access Token") or not access_token_to_use:
            if REQUEST_TOKEN:
                try:
                    kite = KiteConnect(api_key=API_KEY)
                    data = kite.generate_session(REQUEST_TOKEN, api_secret=API_SECRET)
                    kite.set_access_token(data["access_token"])
                    st.session_state.kite = kite

                    # Save token locally for persistence
                    save_local_token(data["access_token"])
                    st.success("✅ Zerodha connected! Access token generated.")
                    st.info("⚠️ Token saved locally and auto-used for today.")
                except Exception as e:
                    st.error(f"Login failed: {e}")
            else:
                st.warning("Please enter the Request Token to generate Access Token.")

        if 'kite' in st.session_state:
            st.success("Connected to Zerodha ✅")
        else:
            st.warning("Not connected yet.")

    except Exception as e:
        st.info(f"Install KiteConnect first: pip install kiteconnect. Error: {e}")


    # NEW: Manual sync button
    st.markdown("---")
    st.subheader("Sync & Status")
    if st.button("🔄 Sync with Zerodha Live"):
        sync_zerodha_positions()

    # Shared Clipboard
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Shared Clipboard")
    CLIPBOARD_FILE = "shared_clipboard.txt"
    shared_text = ""
    if os.path.exists(CLIPBOARD_FILE):
        with open(CLIPBOARD_FILE, "r") as f:
            shared_text = f.read()
    new_text = st.sidebar.text_input("Paste or type here:", shared_text)
    if new_text != shared_text:
        with open(CLIPBOARD_FILE, "w") as f:
            f.write(new_text)
        shared_text = new_text
    st.sidebar.write("🔄 Current text:", shared_text)

else:
    st.info("📊 Running in **Paper Trade Mode** (Zerodha not connected).")
    if "signals" in locals():
        for sig in signals:
            st.write(f"💡 Paper Trade Signal: {sig}")

# ------------------ Main Loop (with Candle Patterns) ------------------
col_main, col_side = st.columns([3, 1])
last_prices = {}
summaries   = []
detailed_df = None

with col_main:
    if not selected:
        st.info("Select one or more stocks in the sidebar to begin scanning.")
    else:
        stop_loss_pct   = stop_pct
        target_pct_val  = target_pct  # avoid shadowing

        # Check for open LIVE positions and update their price/status
        if live_trading and 'kite' in st.session_state and is_market_open:
            st.markdown("---")
            st.subheader("Live Position Updates")
            live_symbols_to_check = list(st.session_state.open_positions.keys())
            for sym in live_symbols_to_check:
                pos = st.session_state.open_positions[sym]
                if pos['mode'] == 'LIVE':
                    live_price = get_live_price(st.session_state.kite, sym)
                    if live_price:
                        # NEW: Update LTP in the session state for display
                        st.session_state.open_positions[sym]['LTP'] = live_price
                        # Check for automated exit based on live price
                        check_exits_with_price(sym, live_price, live_trading=True)
                    else:
                        st.error(f"Could not get live price for {sym}.")

        for sym in selected:
            df, err = fetch_intraday(sym, period, interval)
            if err:
                summaries.append((sym, np.nan, "ERROR", str(err), np.nan, np.nan, np.nan, np.nan, False, np.nan, np.nan, 0.0, ""))
                continue
            if df.empty:
                summaries.append((sym, np.nan, "NO DATA", "", np.nan, np.nan, np.nan, np.nan, False, np.nan, np.nan, 0.0, ""))
                continue

            df  = compute_features(df, vol_multiplier, sma_short=20, sma_long=50)
            
            # NEW: Get daily sentiment score for AI integration
            daily_sentiment = get_sentiment_for_stock(sym) if 'get_sentiment_for_stock' in globals() else 0.0
            
            sig = vote_signal(df, require_all=require_all, vol_multiplier=vol_multiplier, daily_sentiment=daily_sentiment, sentiment_threshold=sentiment_threshold)

            sig_for_entry = sig

            price = df['Close'].iloc[-1]
            last_prices[sym] = float(price)
            sma_s = df['SMA_short'].iloc[-1] if pd.notna(df['SMA_short'].iloc[-1]) else np.nan
            sma_l = df['SMA_long'].iloc[-1]  if pd.notna(df['SMA_long'].iloc[-1])  else np.nan
            rsi_v = df['RSI14'].iloc[-1]      if pd.notna(df['RSI14'].iloc[-1])      else np.nan
            macd_hist = df['MACD_HIST'].iloc[-1] if pd.notna(df['MACD_HIST'].iloc[-1]) else np.nan
            vol_spike = bool(df['Vol_Spike'].iloc[-1]) if 'Vol_Spike' in df.columns else False

            # --- Candlestick pattern(s) on last bar ---
            candle_label = detect_last_candle_patterns(df)

            if isinstance(sig_for_entry, str) and sig_for_entry.startswith("BUY"):
                stop_loss = price * (1 - stop_loss_pct / 100)
                target    = price * (1 + target_pct_val / 100)
            elif isinstance(sig_for_entry, str) and sig_for_entry.startswith("SELL"):
                stop_loss = price * (1 + stop_loss_pct / 100)
                target    = price * (1 - target_pct_val / 100)
            else:
                stop_loss = np.nan
                target    = np.nan

            summaries.append((sym, price, sig_for_entry, "", sma_s, sma_l, rsi_v, macd_hist, vol_spike, stop_loss, target, daily_sentiment, candle_label))

            # push signal history
            if 'scanner_signals' not in st.session_state:
                st.session_state['scanner_signals'] = []
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state['scanner_signals'].append((
                ts, sym, sig_for_entry, float(price),
                float(stop_loss) if pd.notna(stop_loss) else np.nan,
                float(target)    if pd.notna(target)    else np.nan
            ))
            st.session_state['scanner_signals'] = st.session_state['scanner_signals'][-200:]

            # Decide entry/placement
            strong = (sig_for_entry in ('BUY', 'SELL')) or (isinstance(sig_for_entry, str) and sig_for_entry.startswith(('BUY', 'SELL')))
            if (not strong) and close_weak:
                sig_display = "HOLD"
            else:
                sig_display = sig_for_entry
                if auto_sim and strong and (sym not in st.session_state.open_positions):
                    if sig_for_entry.startswith('BUY'):
                        sl = price * (1 - stop_loss_pct/100.0)
                        tgt = price * (1 + target_pct_val/100.0)
                    else:
                        # Will rarely happen if long_only=True; allowed if long_only=False
                        sl = price * (1 + stop_loss_pct/100.0)
                        tgt = price * (1 - target_pct_val/100.0)

                    # Risk-based position sizing
                    risk_amount     = st.session_state.starting_capital * (risk_pct/100.0)
                    per_share_risk  = abs(price - sl)
                    qty = int(np.floor(risk_amount / per_share_risk)) if per_share_risk > 0 else 0
                    max_qty_by_cap  = int(np.floor(st.session_state.starting_capital * 0.25 / price)) # Max 35% of capital per trade
                    qty = min(qty, max_qty_by_cap) # Ensure position size doesn't exceed max allocation
                    
                    if qty > 0 and per_share_risk > 0 and (sig_for_entry.startswith('BUY') or sig_for_entry.startswith('SELL')):
                        # NEW: Pass the trailing stop percentage to the trade placement function
                        place_trade(sym, sig_for_entry, price, sl, tgt, qty, tsl_pct, live_trading=live_trading)

        # Display scanner summary table
        if summaries:
            summary_df = pd.DataFrame(summaries, columns=[
                "Symbol", "Price", "Signal", "Error", "SMA20", "SMA50", "RSI14",
                "MACD_HIST", "Vol_Spike", "Stop Loss", "Target", "Sentiment", "Candle Pattern"
            ])
            st.subheader("📊 Live Scanner Results")
            st.markdown(f"**Current Time (IST):** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.dataframe(summary_df.set_index("Symbol"), use_container_width=True)
            
            # Display signals
            buy_signals = summary_df[summary_df['Signal'].str.startswith("BUY", na=False)]
            sell_signals = summary_df[summary_df['Signal'].str.startswith("SELL", na=False)]
            if not buy_signals.empty:
                st.success("🟢 Strong BUY Signals detected!")
                st.dataframe(buy_signals, use_container_width=True)
            if not sell_signals.empty:
                st.error("🔴 Strong SELL Signals detected!")
                st.dataframe(sell_signals, use_container_width=True)

        # Display a sample of detailed data for the first stock
        if detailed_df is None and not df.empty:
            detailed_df = df.tail(15)

with col_side:
    st.subheader("🤖 Bot Status")
    total_pnl = sum([t['PnL'] for t in st.session_state.trade_history])
    pnl_color = 'green' if total_pnl >= 0 else 'red'
    st.markdown(f"**Total PnL:** <span style='color:{pnl_color};'>₹{total_pnl:.2f}</span>", unsafe_allow_html=True)
    st.markdown(f"**Total Trades:** {len(st.session_state.trade_history)}")
    
    current_cap = st.session_state.starting_capital + total_pnl
    st.markdown(f"**Current Capital:** ₹{current_cap:.2f}")

    # Display news/headlines
    st.markdown("---")
    st.subheader("Headlines")
    for s in selected:
        headlines = scrape_moneycontrol_news(s)
        if headlines:
            st.markdown(f"**{s} News:**")
            for h in headlines:
                st.write(f"- {h}")
    
    # Check for daily loss limit
    daily_loss_limit = st.session_state.starting_capital * 0.02 # 2% loss limit
    if total_pnl < -daily_loss_limit:
        st.error("🚫 Daily loss limit reached. Trading stopped.")
        # Optionally, clear open positions if this is an automated bot
        # st.session_state.open_positions = {}
    elif total_pnl > 0:
        st.success("🎉 Good performance today!")

    # Display open positions
    st.markdown("---")
    st.subheader("✅ Open Positions")
    if st.session_state.open_positions:
        pos_data = []
        for sym, pos in st.session_state.open_positions.items():
            current_pnl = 0
            # Use live price for display if available, otherwise use a placeholder
            live_price = pos.get('LTP')
            if live_price:
                current_pnl = (live_price - pos['entry']) * pos['qty'] if pos['side'] == 'BUY' else (pos['entry'] - live_price) * pos['qty']
            
            pos_data.append({
                "Symbol": sym,
                "Side": pos['side'],
                "Entry": pos['entry'],
                "Qty": pos['qty'],
                "LTP": live_price if live_price else "N/A",
                "PnL": current_pnl,
                "Stop": pos['stop'],
                "Target": pos['target'],
                "Mode": pos['mode'],
                "TSL %": pos.get('tsl_pct', 0.0) # NEW: Display the trailing stop percentage
            })
        st.dataframe(pd.DataFrame(pos_data).set_index("Symbol"), use_container_width=True)
    else:
        st.info("No open positions.")
    
    # Display trade history
    st.markdown("---")
    st.subheader("📜 Trade History")
    if st.session_state.trade_history:
        history_df = pd.DataFrame(st.session_state.trade_history)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No trade history yet.")
