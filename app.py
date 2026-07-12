import os, sys, time, threading, traceback, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

sys.path.insert(0, '/opt/kronosbtc')
from model import Kronos, KronosTokenizer, KronosPredictor

app = Flask(__name__, static_folder='/opt/kronosbtc/static')
CORS(app)

MODEL_NAME     = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
LOOKBACK       = 384
PRED_LEN       = 24
MONTE_CARLO_N  = 30
REFRESH_SECS   = 3600
CACHE_FILE     = "/opt/kronosbtc/cache.json"

COINS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "BNB": "BNBUSDT",
    "SOL": "SOLUSDT",
    "ADA": "ADAUSDT",
    "ZEC": "ZECUSDT",
    "TAO": "TAOUSDT",
}

predictor   = None
cache       = {}
cache_lock  = threading.Lock()
model_ready = False
model_error = ""
running     = {}


# ── Cache persistence ─────────────────────────────────────────────────────────
def save_cache():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
        print("[Cache] Saved to disk.", flush=True)
    except Exception as e:
        print(f"[Cache] Save failed: {e}", flush=True)

def load_cache():
    global cache
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
            print(f"[Cache] Loaded from disk: {list(cache.keys())}", flush=True)
    except Exception as e:
        print(f"[Cache] Load failed: {e}", flush=True)


# ── Technical indicators ──────────────────────────────────────────────────────
def compute_indicators(df):
    """Compute RSI, MACD, Bollinger Bands, Volume momentum."""
    close = df["close"].values

    # RSI (14)
    delta = np.diff(close)
    gain  = np.where(delta > 0, delta, 0)
    loss  = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean().values
    avg_loss = pd.Series(loss).rolling(14).mean().values
    rs  = np.where(avg_loss == 0, 100, avg_gain / (avg_loss + 1e-10))
    rsi = 100 - (100 / (1 + rs))
    rsi = np.append(np.nan, rsi)

    # MACD
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd  = ema12 - ema26
    signal= pd.Series(macd).ewm(span=9, adjust=False).mean().values
    macd_hist = macd - signal

    # Bollinger Bands (20)
    sma20 = pd.Series(close).rolling(20).mean().values
    std20 = pd.Series(close).rolling(20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_pct   = np.where(bb_upper - bb_lower == 0, 0.5,
                        (close - bb_lower) / (bb_upper - bb_lower + 1e-10))

    # Volume momentum
    vol = df["volume"].values
    vol_sma = pd.Series(vol).rolling(20).mean().values
    vol_ratio = vol / (vol_sma + 1e-10)

    latest = {
        "rsi":        round(float(rsi[-1]) if not np.isnan(rsi[-1]) else 50, 2),
        "macd":       round(float(macd[-1]), 4),
        "macd_signal":round(float(signal[-1]), 4),
        "macd_hist":  round(float(macd_hist[-1]), 4),
        "bb_pct":     round(float(np.clip(bb_pct[-1], 0, 1)), 4),
        "vol_ratio":  round(float(vol_ratio[-1]), 4),
        "sma20":      round(float(sma20[-1]), 2),
        "bb_upper":   round(float(bb_upper[-1]), 2),
        "bb_lower":   round(float(bb_lower[-1]), 2),
    }
    return latest


# ── Fear & Greed Index ────────────────────────────────────────────────────────
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        r.raise_for_status()
        data = r.json()
        val  = int(data["data"][0]["value"])
        label= data["data"][0]["value_classification"]
        return {"value": val, "label": label}
    except Exception as e:
        print(f"[F&G] Failed: {e}", flush=True)
        return {"value": 50, "label": "Neutral"}


# ── Data fetcher ──────────────────────────────────────────────────────────────
def fetch_candles(symbol):
    """
    FIX #1 (Time Travel Bug): Fetch only LOOKBACK candles.
    Generate future timestamps from the last known candle forward.
    This ensures last_price is the CURRENT price, not 24h old.
    """
    binance_symbol = COINS[symbol]
    urls = [
        "https://api1.binance.com/api/v3/klines",
        "https://api2.binance.com/api/v3/klines",
        "https://api3.binance.com/api/v3/klines",
        "https://api.binance.com/api/v3/klines",
    ]
    params = {"symbol": binance_symbol, "interval": "1h", "limit": LOOKBACK}
    for url in urls:
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            raw = r.json()
            df = pd.DataFrame(raw, columns=[
                "open_time","open","high","low","close","volume",
                "close_time","qv","trades","tbb","tbq","ignore"
            ])
            df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            for c in ["open","high","low","close","volume"]:
                df[c] = df[c].astype(float)
            df = df[["timestamps","open","high","low","close","volume"]].reset_index(drop=True)
            print(f"[Data] {symbol} Binance OK {len(df)} candles", flush=True)
            return df
        except Exception as e:
            print(f"[Data] {url} failed: {e}", flush=True)
    raise RuntimeError(f"Could not fetch candles for {symbol}")


# ── Core prediction ───────────────────────────────────────────────────────────
def run_prediction(symbol):
    global cache

    print(f"[Kronos] Fetching {symbol} candles...", flush=True)
    df = fetch_candles(symbol)

    # FIX #1: last_price is the ACTUAL current price (last candle in history)
    last_price = float(df["close"].iloc[-1])
    last_time  = df["timestamps"].iloc[-1]

    x_df        = df[["open","high","low","close","volume"]].copy()
    x_timestamp = df["timestamps"].copy().reset_index(drop=True)

    # Generate future timestamps from last known candle forward
    freq = pd.Timedelta(hours=1)
    future_times = pd.date_range(
        start=last_time + freq,
        periods=PRED_LEN,
        freq=freq,
        tz="UTC"
    )
    y_timestamp = pd.Series(future_times).reset_index(drop=True)

    print(f"[Kronos] {symbol} current price: ${last_price:,.2f}", flush=True)

    # Compute technical indicators
    indicators = compute_indicators(df)
    print(f"[Kronos] RSI={indicators['rsi']} MACD_hist={indicators['macd_hist']} BB%={indicators['bb_pct']}", flush=True)

    # Fetch Fear & Greed (BTC only — it's a market-wide index)
    fear_greed = fetch_fear_greed() if symbol == "BTC" else {"value": 50, "label": "N/A"}
    print(f"[Kronos] Fear&Greed={fear_greed['value']} ({fear_greed['label']})", flush=True)

    # FIX #3: Use sample_count=MONTE_CARLO_N for single-pass batched inference
    print(f"[Kronos] Running batched Monte Carlo N={MONTE_CARLO_N} for {symbol}...", flush=True)
    try:
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=PRED_LEN,
            T=0.7,
            top_p=0.9,
            sample_count=MONTE_CARLO_N
        )
        # Batched result: shape (N, PRED_LEN) or (PRED_LEN,) depending on version
        if isinstance(pred_df, pd.DataFrame):
            # Single result returned — fall back to loop
            raise ValueError("sample_count not supported, falling back to loop")
        all_closes = pred_df
        print(f"[Kronos] Batched inference successful!", flush=True)
    except Exception as e:
        print(f"[Kronos] Batched inference failed ({e}), using loop...", flush=True)
        all_closes = []
        all_highs  = []
        all_lows   = []
        for i in range(MONTE_CARLO_N):
            print(f"[Kronos] {symbol} MC {i+1}/{MONTE_CARLO_N}", flush=True)
            p = predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=PRED_LEN,
                T=0.7,
                top_p=0.9,
                sample_count=1
            )
            all_closes.append(p["close"].values)
            all_highs.append(p["high"].values)
            all_lows.append(p["low"].values)

        closes = np.array(all_closes)
        highs  = np.array(all_highs)
        lows   = np.array(all_lows)

        mean_close = closes.mean(axis=0)

        # FIX #2: Use percentiles instead of absolute max/min
        upper = np.percentile(closes, 90, axis=0)
        lower = np.percentile(closes, 10, axis=0)

        final_prices = closes[:, -1]

        # FIX #4: Volatility as percentage change (apples to apples)
        std_pct      = closes.std(axis=0) / last_price * 100
        hist_ret     = df["close"].pct_change().dropna()
        hist_vol_pct = float(hist_ret.std() * 100)
        vol_amp_prob = float((std_pct > hist_vol_pct).mean()) * 100

        # Confidence score: inverse of spread consistency
        spread_pct   = (upper - lower) / last_price * 100
        confidence   = round(max(0, 100 - spread_pct.mean() * 2), 1)

        upside_prob  = float((final_prices > last_price).mean()) * 100

        # Signal strength combining model + technicals + fear&greed
        tech_signal = 0
        if indicators["rsi"] < 30: tech_signal += 20    # oversold = bullish
        elif indicators["rsi"] > 70: tech_signal -= 20  # overbought = bearish
        if indicators["macd_hist"] > 0: tech_signal += 15
        else: tech_signal -= 15
        if indicators["bb_pct"] < 0.2: tech_signal += 15
        elif indicators["bb_pct"] > 0.8: tech_signal -= 15
        if indicators["vol_ratio"] > 1.5: tech_signal += 10
        fg_signal = (fear_greed["value"] - 50) * 0.3
        combined_signal = round(upside_prob + tech_signal * 0.3 + fg_signal, 1)
        combined_signal = round(max(0, min(100, combined_signal)), 1)

        result = {
            "updated_at":      datetime.now(timezone.utc).isoformat(),
            "symbol":          f"{symbol}/USDT",
            "coin":            symbol,
            "last_price":      last_price,
            "last_time":       str(last_time),
            "pred_len":        PRED_LEN,
            "lookback":        LOOKBACK,
            "monte_carlo_n":   MONTE_CARLO_N,
            "model":           MODEL_NAME,
            "upside_prob":     round(upside_prob, 1),
            "combined_signal": combined_signal,
            "vol_amp_prob":    round(vol_amp_prob, 1),
            "confidence":      confidence,
            "indicators":      indicators,
            "fear_greed":      fear_greed,
            "forecast": {
                "timestamps": [str(t) for t in future_times],
                "mean_close": [round(v, 4) for v in mean_close.tolist()],
                "upper":      [round(v, 4) for v in upper.tolist()],
                "lower":      [round(v, 4) for v in lower.tolist()],
            },
            "history": {
                "timestamps": [str(t) for t in df["timestamps"].tolist()],
                "close":      [round(v, 4) for v in df["close"].tolist()],
            }
        }

        with cache_lock:
            cache[symbol] = result
        save_cache()

        print(f"[Kronos] {symbol} done. Upside: {upside_prob:.1f}% | Signal: {combined_signal:.1f}% | Confidence: {confidence:.1f}%", flush=True)
        return result


def prediction_loop(symbol):
    global running
    running[symbol] = True
    try:
        run_prediction(symbol)
    except Exception as e:
        print(f"[Kronos] {symbol} failed: {e}", flush=True)
        traceback.print_exc()
    finally:
        running[symbol] = False
    print(f"[Kronos] {symbol} next refresh in 1h...", flush=True)
    time.sleep(REFRESH_SECS)
    threading.Thread(target=prediction_loop, args=(symbol,), daemon=True).start()


def load_model():
    global predictor, model_ready, model_error
    try:
        load_cache()  # Load saved predictions immediately on startup
        print("[Kronos] Loading tokenizer...", flush=True)
        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
        print("[Kronos] Loading Kronos-base...", flush=True)
        model = Kronos.from_pretrained(MODEL_NAME)
        model.eval()
        predictor = KronosPredictor(model, tokenizer, max_context=512)
        model_ready = True
        print("[Kronos] Model ready!", flush=True)
        threading.Thread(target=prediction_loop, args=("BTC",), daemon=True).start()
    except Exception as e:
        model_error = str(e)
        print(f"[Kronos] Load failed: {e}", flush=True)
        traceback.print_exc()


threading.Thread(target=load_model, daemon=True).start()


@app.route("/")
def index():
    return send_from_directory("/opt/kronosbtc/static", "index.html")

@app.route("/status")
def status():
    return jsonify({
        "model_ready": model_ready,
        "model_error": model_error,
        "model":       MODEL_NAME,
        "coins":       list(COINS.keys()),
        "cached":      list(cache.keys()),
        "running":     {k: v for k, v in running.items() if v},
    })

@app.route("/cache/<symbol>")
def get_cache(symbol):
    symbol = symbol.upper()
    if symbol not in COINS:
        return jsonify({"error": f"Unknown symbol {symbol}"}), 400
    if symbol not in cache:
        return jsonify({
            "error": f"No prediction yet for {symbol}.",
            "model_ready": model_ready,
            "is_running": running.get(symbol, False)
        }), 404
    result = dict(cache[symbol])
    result["is_running"] = running.get(symbol, False)
    return jsonify(result)

@app.route("/predict/<symbol>")
def predict(symbol):
    symbol = symbol.upper()
    if symbol not in COINS:
        return jsonify({"error": f"Unknown symbol {symbol}"}), 400
    if not model_ready:
        return jsonify({"error": "Model not loaded yet."}), 503
    if running.get(symbol, False):
        return jsonify({"status": "already_running"})
    threading.Thread(target=prediction_loop, args=(symbol,), daemon=True).start()
    return jsonify({"status": "started"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
