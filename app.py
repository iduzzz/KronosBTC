"""
Kronos BTC Predictor — Render Deployment
=========================================
- Kronos-base 102M parameters
- 384 candles context (lookback) + 24 prediction (pred_len) = 408 total fetched
- Monte Carlo N=30
- Auto-refresh every 2 hours in background
- Result always cached — iPhone gets instant response
"""

import os
import sys
import time
import threading
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

# Kronos source is cloned into ./model by build.sh
sys.path.insert(0, os.path.dirname(__file__))
from model import Kronos, KronosTokenizer, KronosPredictor

app = Flask(__name__, static_folder="static")
CORS(app)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME     = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
LOOKBACK       = 384   # history candles fed to model
PRED_LEN       = 24    # forecast hours
MONTE_CARLO_N  = 30    # sample paths
REFRESH_SECS   = 2 * 3600  # auto-refresh every 2 hours

# ── State ─────────────────────────────────────────────────────────────────────
predictor   = None
cache       = {}
cache_lock  = threading.Lock()
model_ready = False
model_error = ""
is_running  = False


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model():
    global predictor, model_ready, model_error
    try:
        print("[Kronos] Loading tokenizer...", flush=True)
        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
        print("[Kronos] Loading Kronos-base (102M params)...", flush=True)
        model = Kronos.from_pretrained(MODEL_NAME)
        model.eval()
        predictor = KronosPredictor(model, tokenizer, max_context=512)
        model_ready = True
        print("[Kronos] Model ready. Starting first prediction...", flush=True)
        threading.Thread(target=prediction_loop, daemon=True).start()
    except Exception as e:
        model_error = str(e)
        print(f"[Kronos] Model load FAILED: {e}", flush=True)
        traceback.print_exc()


# ── Data fetcher ──────────────────────────────────────────────────────────────
def fetch_candles(total_candles):
    """
    Fetch total_candles of BTC/USDT 1h data.
    Returns a clean DataFrame with columns: timestamps, open, high, low, close, volume
    Rows are sorted oldest → newest, integer-indexed from 0.
    """
    binance_mirrors = [
        "https://api1.binance.com/api/v3/klines",
        "https://api2.binance.com/api/v3/klines",
        "https://api3.binance.com/api/v3/klines",
        "https://api4.binance.com/api/v3/klines",
        "https://api.binance.com/api/v3/klines",
    ]
    params = {"symbol": "BTCUSDT", "interval": "1h", "limit": total_candles}

    for url in binance_mirrors:
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
            print(f"[Data] Binance OK — {len(df)} candles from {url}", flush=True)
            return df
        except Exception as e:
            print(f"[Data] {url} failed: {e}", flush=True)

    # KuCoin fallback
    print("[Data] All Binance mirrors failed — trying KuCoin...", flush=True)
    end_ts   = int(time.time())
    start_ts = end_ts - (total_candles * 3600 * 2)
    r = requests.get(
        "https://api.kucoin.com/api/v1/market/candles",
        params={"symbol": "BTC-USDT", "type": "1hour",
                "startAt": start_ts, "endAt": end_ts},
        timeout=15
    )
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "200000":
        raise RuntimeError(f"KuCoin API error: {data}")
    rows = list(reversed(data["data"]))[-total_candles:]
    records = []
    for row in rows:
        records.append({
            "timestamps": pd.Timestamp(int(row[0]), unit="s", tz="UTC"),
            "open":   float(row[1]),
            "close":  float(row[2]),
            "high":   float(row[3]),
            "low":    float(row[4]),
            "volume": float(row[5]),
        })
    df = pd.DataFrame(records).reset_index(drop=True)
    print(f"[Data] KuCoin OK — {len(df)} candles", flush=True)
    return df


# ── Core prediction ───────────────────────────────────────────────────────────
def run_prediction():
    """
    Run full Kronos-base prediction with Monte Carlo N=30.
    Uses EXACT same pattern as official Kronos examples:
      - Fetch LOOKBACK + PRED_LEN candles in one dataframe
      - x = df.loc[0 : LOOKBACK-1]           (history)
      - y = df.loc[LOOKBACK : LOOKBACK+PRED_LEN-1]  (future timestamps)
    This guarantees the timestamps are internally consistent.
    """
    global cache

    total = LOOKBACK + PRED_LEN
    print(f"[Kronos] Fetching {total} candles...", flush=True)
    df = fetch_candles(total)

    if len(df) < total:
        raise RuntimeError(f"Not enough candles: got {len(df)}, need {total}")

    # Ensure exactly the right number, oldest→newest
    df = df.iloc[-total:].reset_index(drop=True)

    # Split exactly as official examples do
    x_df        = df.loc[:LOOKBACK-1, ["open","high","low","close","volume"]]
    x_timestamp = df.loc[:LOOKBACK-1, "timestamps"]
    y_timestamp = df.loc[LOOKBACK:LOOKBACK+PRED_LEN-1, "timestamps"]

    last_price = float(df.loc[LOOKBACK-1, "close"])
    last_time  = df.loc[LOOKBACK-1, "timestamps"]

    print(f"[Kronos] x={len(x_df)} candles | y={len(y_timestamp)} future hours", flush=True)
    print(f"[Kronos] Last known price: ${last_price:,.2f}", flush=True)
    print(f"[Kronos] Starting Monte Carlo N={MONTE_CARLO_N}...", flush=True)

    all_closes = []
    all_highs  = []
    all_lows   = []

    for i in range(MONTE_CARLO_N):
        print(f"[Kronos] MC {i+1}/{MONTE_CARLO_N}...", flush=True)
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=PRED_LEN,
            T=1.0,
            top_p=0.9,
            sample_count=1
        )
        all_closes.append(pred_df["close"].values)
        all_highs.append(pred_df["high"].values)
        all_lows.append(pred_df["low"].values)

    closes    = np.array(all_closes)   # shape (N, PRED_LEN)
    highs     = np.array(all_highs)
    lows      = np.array(all_lows)

    mean_close = closes.mean(axis=0)
    upper      = highs.max(axis=0)
    lower      = lows.min(axis=0)
    std_close  = closes.std(axis=0)

    final_prices   = closes[:, -1]
    upside_prob    = float((final_prices > last_price).mean()) * 100
    hist_vol       = float(df["close"].pct_change().std() * last_price)
    vol_amp_prob   = float((std_close > hist_vol).mean()) * 100

    result = {
        "updated_at":    datetime.now(timezone.utc).isoformat(),
        "symbol":        "BTC/USDT",
        "last_price":    last_price,
        "last_time":     str(last_time),
        "pred_len":      PRED_LEN,
        "lookback":      LOOKBACK,
        "monte_carlo_n": MONTE_CARLO_N,
        "model":         MODEL_NAME,
        "upside_prob":   round(upside_prob, 1),
        "vol_amp_prob":  round(vol_amp_prob, 1),
        "forecast": {
            "timestamps": [str(t) for t in y_timestamp.tolist()],
            "mean_close": [round(v, 2) for v in mean_close.tolist()],
            "upper":      [round(v, 2) for v in upper.tolist()],
            "lower":      [round(v, 2) for v in lower.tolist()],
        },
        "history": {
            "timestamps": [str(t) for t in df["timestamps"].tolist()],
            "close":      [round(v, 2) for v in df["close"].tolist()],
        }
    }

    with cache_lock:
        cache = result

    print(f"[Kronos] ✅ Done. Upside prob: {upside_prob:.1f}% | Vol amp: {vol_amp_prob:.1f}%", flush=True)
    return result


# ── Background loop ───────────────────────────────────────────────────────────
def prediction_loop():
    """Runs forever: predict → wait 2h → predict → ..."""
    global is_running
    while True:
        is_running = True
        try:
            run_prediction()
        except Exception as e:
            print(f"[Kronos] Prediction failed: {e}", flush=True)
            traceback.print_exc()
        finally:
            is_running = False
        print(f"[Kronos] Next refresh in {REFRESH_SECS//3600}h...", flush=True)
        time.sleep(REFRESH_SECS)


# ── Start model loading immediately ──────────────────────────────────────────
threading.Thread(target=load_model, daemon=True).start()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/status")
def status():
    return jsonify({
        "model_ready": model_ready,
        "model_error": model_error,
        "has_cache":   bool(cache),
        "is_running":  is_running,
        "cache_time":  cache.get("updated_at"),
        "model":       MODEL_NAME,
        "lookback":    LOOKBACK,
        "monte_carlo_n": MONTE_CARLO_N,
    })


@app.route("/cache")
def get_cache():
    if not cache:
        return jsonify({
            "error":       "First prediction still computing.",
            "model_ready": model_ready,
            "is_running":  is_running,
        }), 404
    result = dict(cache)
    result["is_running"] = is_running
    return jsonify(result)


@app.route("/refresh")
def refresh():
    """Manually trigger a fresh prediction in background."""
    if not model_ready:
        return jsonify({"error": "Model not loaded yet."}), 503
    if is_running:
        return jsonify({"status": "already_running", "message": "Prediction already in progress."})
    threading.Thread(target=prediction_loop, daemon=True).start()
    return jsonify({"status": "started", "message": "Fresh prediction started in background."})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
