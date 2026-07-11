import os, sys, time, threading, traceback
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, send_from_directory, request
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
cache       = {}        # cache[symbol] = result
cache_lock  = threading.Lock()
model_ready = False
model_error = ""
running     = {}        # running[symbol] = True/False

def load_model():
    global predictor, model_ready, model_error
    try:
        print("[Kronos] Loading tokenizer...", flush=True)
        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
        print("[Kronos] Loading Kronos-base...", flush=True)
        model = Kronos.from_pretrained(MODEL_NAME)
        model.eval()
        predictor = KronosPredictor(model, tokenizer, max_context=512)
        model_ready = True
        print("[Kronos] Model ready!", flush=True)
        # Auto-start BTC prediction
        threading.Thread(target=prediction_loop, args=("BTC",), daemon=True).start()
    except Exception as e:
        model_error = str(e)
        print(f"[Kronos] Load failed: {e}", flush=True)
        traceback.print_exc()

def fetch_candles(symbol):
    # FIX: Only fetch LOOKBACK candles. We don't want past "future" data.
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
            df = pd.DataFrame(raw, columns=["open_time","open","high","low","close","volume","close_time","qv","trades","tbb","tbq","ignore"])
            df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            for c in ["open","high","low","close","volume"]:
                df[c] = df[c].astype(float)
            df = df[["timestamps","open","high","low","close","volume"]].reset_index(drop=True)
            print(f"[Data] {symbol} Binance OK {len(df)} candles", flush=True)
            return df
        except Exception as e:
            print(f"[Data] {url} failed: {e}", flush=True)
    raise RuntimeError(f"Could not fetch candles for {symbol}")

def run_prediction(symbol):
    global cache
    print(f"[Kronos] Fetching {symbol} candles...", flush=True)
    df = fetch_candles(symbol)
    
    # 1. Input data is simply the latest LOOKBACK candles
    x_df        = df[["open","high","low","close","volume"]]
    x_timestamp = df["timestamps"]
    
    # 2. Get the actual current price and time
    last_price  = float(df.iloc[-1]["close"])
    last_time   = df.iloc[-1]["timestamps"]
    
    # 3. Generate FUTURE timestamps for the forecast (1h after last candle, for 24 hours)
    y_timestamp = pd.date_range(start=last_time + timedelta(hours=1), periods=PRED_LEN, freq='1h')
    
    print(f"[Kronos] {symbol} last price: ${last_price:,.2f}", flush=True)
    print(f"[Kronos] Running optimized Monte Carlo (N={MONTE_CARLO_N}) for {symbol}...", flush=True)
    
    # 4. SPEED FIX: Try batch sampling first. If the model wrapper supports it, it's 30x faster.
    try:
        print(f"[Kronos] Attempting batch sampling (sample_count={MONTE_CARLO_N})...", flush=True)
        pred_df = predictor.predict(
            df=x_df, 
            x_timestamp=x_timestamp, 
            y_timestamp=y_timestamp, 
            pred_len=PRED_LEN, 
            T=0.7, 
            top_p=0.9, 
            sample_count=MONTE_CARLO_N
        )
        
        # The output structure depends on your specific Kronos wrapper. 
        # We try to stack it into numpy arrays.
        if isinstance(pred_df, list):
            all_closes = np.array([p["close"].values for p in pred_df])
            all_highs = np.array([p["high"].values for p in pred_df])
            all_lows = np.array([p["low"].values for p in pred_df])
        else:
            # If it returns a single dataframe with a multi-index, we raise an error to trigger the fallback
            raise ValueError("Predictor did not return a list of samples. Falling back to loop.")
            
    except Exception as e:
        # Fallback to the loop if batch sampling isn't supported by your specific Kronos wrapper
        print(f"[Kronos] Batch sampling failed ({e}). Falling back to loop...", flush=True)
        all_closes, all_highs, all_lows = [], [], []
        for i in range(MONTE_CARLO_N):
            print(f"[Kronos] {symbol} MC {i+1}/{MONTE_CARLO_N}", flush=True)
            pred_df = predictor.predict(df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp, pred_len=PRED_LEN, T=0.7, top_p=0.9, sample_count=1)
            all_closes.append(pred_df["close"].values)
            all_highs.append(pred_df["high"].values)
            all_lows.append(pred_df["low"].values)
        all_closes = np.array(all_closes)
        all_highs = np.array(all_highs)
        all_lows = np.array(all_lows)

    # Calculate mean
    mean_close = all_closes.mean(axis=0)
    
    # 5. PRECISION FIX: Use 90th and 10th percentiles instead of absolute max/min
    upper      = np.percentile(all_highs, 90, axis=0)
    lower      = np.percentile(all_lows, 10, axis=0)
    
    # 6. Calculate probabilities
    final_prices = all_closes[:, -1]
    upside_prob  = float((final_prices > last_price).mean()) * 100
    
    # 7. MATH FIX: Compare percentage volatility instead of absolute volatility
    hist_vol_pct = float(df["close"].pct_change().std()) # Historical percentage volatility
    pred_vol_pct = float(np.std(all_closes / last_price, axis=0).mean()) 
    vol_amp_prob = float((pred_vol_pct > hist_vol_pct)) * 100

    # Build the result dictionary
    result = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": f"{symbol}/USDT",
        "coin": symbol,
        "last_price": last_price,
        "last_time": str(last_time),
        "pred_len": PRED_LEN,
        "lookback": LOOKBACK,
        "monte_carlo_n": MONTE_CARLO_N,
        "model": MODEL_NAME,
        "upside_prob": round(upside_prob, 1),
        "vol_amp_prob": round(vol_amp_prob, 1),
        "forecast": {
            "timestamps": [str(t) for t in y_timestamp.tolist()],
            "mean_close": [round(v, 2) for v in mean_close.tolist()],
            "upper": [round(v, 2) for v in upper.tolist()],
            "lower": [round(v, 2) for v in lower.tolist()],
        },
        "history": {
            "timestamps": [str(t) for t in df["timestamps"].tolist()],
            "close": [round(v, 2) for v in df["close"].tolist()],
        }
    }
    with cache_lock:
        cache[symbol] = result
    print(f"[Kronos] {symbol} done. Upside: {upside_prob:.1f}%", flush=True)
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

threading.Thread(target=load_model, daemon=True).start()

@app.route("/")
def index():
    return send_from_directory("/opt/kronosbtc/static", "index.html")

@app.route("/status")
def status():
    return jsonify({
        "model_ready": model_ready,
        "model_error": model_error,
        "model": MODEL_NAME,
        "coins": list(COINS.keys()),
        "cached": list(cache.keys()),
        "running": {k: v for k, v in running.items() if v},
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
            "is_running": running.get(symbol, False),
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
        return jsonify({"status": "already_running", "message": f"{symbol} prediction already in progress."})
    threading.Thread(target=prediction_loop, args=(symbol,), daemon=True).start()
    return jsonify({"status": "started", "message": f"{symbol} prediction started."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
