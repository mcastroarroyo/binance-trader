cat > trade_agent/agent.py << 'PY'
import os
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from typing import Dict, Any

# ---- trading core (minimal, adapted from your 4h bot) ----
import math, csv, numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
from binance.spot import Spot

SYMBOL       = os.getenv("SYMBOL", "BTCUSDT")
API_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://testnet.binance.vision")

DONCHIAN_N   = int(os.getenv("DONCHIAN_N", "60"))
ATR_N        = int(os.getenv("ATR_N", "14"))
EMA_N        = int(os.getenv("EMA_N", "100"))
ATR_PCT_TH   = float(os.getenv("ATR_PCT_TH", "0.004"))  # 0.4%
ATR_STOP     = float(os.getenv("ATR_STOP", "2.0"))
TIME_STOP    = int(os.getenv("TIME_STOP", "48"))
VOL_TARGET   = float(os.getenv("VOL_TARGET", "0.15"))
BARS_PER_YEAR = 6*365  # 4h bars

FEE_BPS      = int(os.getenv("FEE_BPS", "5"))
SLIP_BPS     = int(os.getenv("SLIP_BPS", "2"))
PRICE_NUDGE  = float(os.getenv("PRICE_NUDGE", "0.001"))

NOTIONAL_TARGET_USDT = float(os.getenv("NOTIONAL_TARGET_USDT", "300"))

API_KEY      = os.getenv("BINANCE_KEY")
API_SECRET   = os.getenv("BINANCE_SECRET")

def _client():
    return Spot(api_key=API_KEY, api_secret=API_SECRET, base_url=API_BASE_URL)

def _fetch_klines(limit=1200):
    k = _client().klines(SYMBOL, "4h", limit=limit)
    cols = ["open_time","open","high","low","close","volume","close_time","qav","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(k, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.set_index("open_time").sort_index()

def _ema(x, n): return x.ewm(span=n, adjust=False).mean()
def _tr(df):
    pc = df["close"].shift(1)
    return pd.concat([(df["high"]-df["low"]).abs(), (df["high"]-pc).abs(), (df["low"]-pc).abs()], axis=1).max(axis=1)
def _atr(df, n): return _tr(df).rolling(n).mean()
def _donchian(df, n):
    up = df["high"].rolling(n).max().shift(1)
    dn = df["low"].rolling(n).min().shift(1)
    return up, dn

def _last_closed_signal(df) -> Dict[str, Any]:
    if len(df) < 300: return {"ok": False, "reason": "not_enough_bars"}
    df = df.copy()
    df["ret1"] = df["close"].pct_change()
    up, dn = _donchian(df, DONCHIAN_N)
    xatr = _atr(df, ATR_N)
    df["ema"] = _ema(df["close"], EMA_N)
    i = -2
    close = df["close"].iloc[i]
    atrv  = xatr.iloc[i]
    emaok = close > df["ema"].iloc[i]
    volok = (atrv / close) > ATR_PCT_TH if pd.notna(atrv) else False
    long_raw = bool(pd.notna(up.iloc[i]) and close >= up.iloc[i])
    flat_raw = bool(pd.notna(dn.iloc[i]) and close <= dn.iloc[i])
    want_long = long_raw and emaok and volok
    want_flat = flat_raw
    ann_vol = df["ret1"].rolling(6).std().iloc[i] * math.sqrt(BARS_PER_YEAR)
    w = VOL_TARGET / (ann_vol if ann_vol and ann_vol>0 else np.nan)
    w = float(min(max(0.0, 0.0 if (np.isnan(w)) else w), 1.0))
    stop_px = float(close - ATR_STOP * (atrv if pd.notna(atrv) else 0.0))
    return {"ok": True, "want_long": want_long, "want_flat": want_flat, "w": w,
            "close": float(close), "atr": float(atrv or 0.0), "stop_px": stop_px}

def _inventory():
    acct = _client().account()
    bal = {a["asset"]: float(a["free"]) + float(a["locked"]) for a in acct["balances"]}
    base = SYMBOL.replace("USDT", "")
    return bal.get(base, 0.0), bal.get("USDT", 0.0)

def _place_limit(side, px, qty):
    price = round(px, 2)             # adjust filters as needed
    qty = float(f"{qty:.6f}")
    return _client().new_order(symbol=SYMBOL, side=side, type="LIMIT",
                               timeInForce="GTC", quantity=qty, price=f"{price:.2f}")

def _cancel_all():
    try: _client().cancel_open_orders(symbol=SYMBOL)
    except Exception: pass

def run_once() -> Dict[str, Any]:
    try:
        df = _fetch_klines()
        sig = _last_closed_signal(df)
        if not sig.get("ok"): return {"status":"skip","reason":sig.get("reason")}
        last_close_ts = df.index[-1]
        if (datetime.now(timezone.utc) - last_close_ts) > timedelta(hours=5):
            return {"status":"error","reason":"stale_data"}

        inv_qty, usdt = _inventory()
        ticker = float(_client().ticker_price(symbol=SYMBOL)["price"])
        target_notional = min(NOTIONAL_TARGET_USDT, usdt * 0.5)
        target_qty = max(0.0001, min(0.01, (target_notional / ticker) * sig["w"]))

        action = "HOLD"; order_id = None
        if sig["want_long"] and inv_qty < target_qty * 0.9:
            _cancel_all()
            buy_px = ticker * (1 - float(PRICE_NUDGE))
            placed = _place_limit("BUY", buy_px, max(0.0, target_qty - inv_qty))
            action, order_id = "BUY", placed.get("orderId")
        elif sig["want_flat"] and inv_qty > 0.0001:
            _cancel_all()
            sell_px = ticker * (1 + float(PRICE_NUDGE))
            placed = _place_limit("SELL", sell_px, inv_qty)
            action, order_id = "SELL", placed.get("orderId")

        return {
            "status":"ok","action":action,"symbol":SYMBOL,
            "price":ticker,"w":sig["w"],"close":sig["close"],
            "atr":sig["atr"],"stop_px":sig["stop_px"],"order_id":order_id
        }
    except Exception as e:
        return {"status":"error","error":str(e)}

# ---- ADK agent that exposes run_once as a tool ----
def run_trade_cycle() -> dict:
    """Execute one 4h trading cycle and return a summary JSON."""
    return run_once()

root_agent = Agent(
    name="binance_4h_bot",
    model="gemini-2.0-flash",  # any supported model route; ADK handles it
    instruction="Call run_trade_cycle and return the JSON result as-is.",
    tools=[FunctionTool(func=run_trade_cycle)],
)
PY
