import os, sys, time, threading, traceback, json, queue
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import xml.etree.ElementTree as ET

sys.path.insert(0, '/opt/kronosbtc')
from model import Kronos, KronosTokenizer, KronosPredictor

app = Flask(__name__, static_folder='/opt/kronosbtc/static')
CORS(app)

MODEL_NAME     = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
LOOKBACK_1H    = 384
LOOKBACK_4H    = 384
PRED_LEN       = 24
MONTE_CARLO_N  = 30
REFRESH_SECS   = 3600
CACHE_FILE     = "/opt/kronosbtc/cache.json"

COINS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "BNB": "BNBUSDT", "SOL": "SOLUSDT",
    "ADA": "ADAUSDT", "ZEC": "ZECUSDT", "TAO": "TAOUSDT",
}

predictor  = None
cache      = {}
cache_lock = threading.Lock()
model_ready = False
model_error = ""
running      = {}   # symbol -> True/False
running_since = {}  # symbol -> start timestamp

# Safe queue worker — prevents thread leaks
task_queue = queue.Queue()

def worker():
    while True:
        symbol = task_queue.get()
        if symbol is None:
            break
        running[symbol] = True
        running_since[symbol] = time.time()
        try:
            run_prediction(symbol)
        except Exception as e:
            print(f"[Kronos] {symbol} failed: {e}", flush=True)
            traceback.print_exc()
        finally:
            running[symbol] = False
        print(f"[Kronos] {symbol} next refresh in 1h...", flush=True)
        time.sleep(REFRESH_SECS)
        task_queue.put(symbol)


# ── Cache persistence ──────────────────────────────────────────────────────────
def save_cache():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"[Cache] Save failed: {e}", flush=True)

def load_cache_from_disk():
    global cache
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
            print(f"[Cache] Loaded: {list(cache.keys())}", flush=True)
    except Exception as e:
        print(f"[Cache] Load failed: {e}", flush=True)


# ── Technical indicators ───────────────────────────────────────────────────────
def compute_indicators(df):
    close = df["close"].values

    # FIX #2: RSI with Wilder's Smoothing (EMA alpha=1/14), not simple rolling mean
    delta    = np.diff(close)
    gain     = np.where(delta > 0, delta, 0.0)
    loss     = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs       = np.where(avg_loss == 0, 100.0, avg_gain / (avg_loss + 1e-10))
    rsi      = 100 - (100 / (1 + rs))
    rsi      = np.append(np.nan, rsi)

    ema12     = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26     = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd      = ema12 - ema26
    sig       = pd.Series(macd).ewm(span=9, adjust=False).mean().values
    macd_hist = macd - sig

    sma20    = pd.Series(close).rolling(20).mean().values
    std20    = pd.Series(close).rolling(20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_pct   = np.where(bb_upper - bb_lower == 0, 0.5,
                        (close - bb_lower) / (bb_upper - bb_lower + 1e-10))

    vol       = df["volume"].values
    vol_sma   = pd.Series(vol).rolling(20).mean().values
    vol_ratio = vol / (vol_sma + 1e-10)

    return {
        "rsi":         round(float(rsi[-1]) if not np.isnan(rsi[-1]) else 50, 2),
        "macd":        round(float(macd[-1]), 4),
        "macd_signal": round(float(sig[-1]), 4),
        "macd_hist":   round(float(macd_hist[-1]), 4),
        "bb_pct":      round(float(np.clip(bb_pct[-1], 0, 1)), 4),
        "vol_ratio":   round(float(vol_ratio[-1]), 4),
        "sma20":       round(float(sma20[-1]), 2),
        "bb_upper":    round(float(bb_upper[-1]), 2),
        "bb_lower":    round(float(bb_lower[-1]), 2),
    }


# ── Fear & Greed ───────────────────────────────────────────────────────────────
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        r.raise_for_status()
        data = r.json()
        return {"value": int(data["data"][0]["value"]),
                "label": data["data"][0]["value_classification"]}
    except:
        return {"value": 50, "label": "Neutral"}


# ── On-chain data (BTC only) ───────────────────────────────────────────────────
def fetch_onchain():
    result = {"mempool_size": 0, "tx_count": 0, "mempool_label": "Normal"}
    try:
        r = requests.get("https://mempool.space/api/mempool", timeout=10)
        if r.ok:
            m = r.json()
            result["mempool_size"] = m.get("count", 0)
            # No directional signal — just display as context
            if result["mempool_size"] > 100000:
                result["mempool_label"] = "Very High"
            elif result["mempool_size"] > 50000:
                result["mempool_label"] = "High"
            elif result["mempool_size"] < 5000:
                result["mempool_label"] = "Low"
            else:
                result["mempool_label"] = "Normal"
    except Exception as e:
        print(f"[OnChain] Mempool failed: {e}", flush=True)
    try:
        time.sleep(1)
        r2 = requests.get("https://blockchain.info/q/24hrtransactioncount", timeout=10)
        if r2.ok:
            result["tx_count"] = int(r2.text.strip())
    except Exception as e:
        print(f"[OnChain] TxCount failed: {e}", flush=True)
    print(f"[OnChain] mempool={result['mempool_size']} ({result['mempool_label']}) tx24h={result['tx_count']}", flush=True)
    return result


# ── News sentiment ─────────────────────────────────────────────────────────────
POSITIVE_WORDS = ["bull", "surge", "rally", "gain", "rise", "high", "record", "pump",
                  "adoption", "approval", "etf", "institutional", "buy", "breakout",
                  "growth", "positive", "increase", "profit", "win", "soar"]
NEGATIVE_WORDS = ["bear", "crash", "drop", "fall", "low", "hack", "ban", "sell",
                  "dump", "fear", "loss", "decline", "negative", "decrease",
                  "scam", "fraud", "fine", "lawsuit", "plunge", "collapse"]

def fetch_news_sentiment(symbol):
    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    ]
    headlines = []
    for url in feeds:
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            if not r.ok:
                continue
            root = ET.fromstring(r.content)
            for item in root.findall(".//item")[:10]:
                title = item.findtext("title", "")
                desc  = item.findtext("description", "")
                headlines.append((title + " " + desc).lower())
            time.sleep(1)
        except Exception as e:
            print(f"[News] {url} failed: {e}", flush=True)

    if not headlines:
        return {"score": 0, "label": "Neutral", "headline_count": 0}

    coin_map = {"BTC": "bitcoin", "ETH": "ethereum", "BNB": "binance",
                "SOL": "solana", "ADA": "cardano", "ZEC": "zcash", "TAO": "bittensor"}
    coin_full = coin_map.get(symbol, symbol.lower())
    coin_name = symbol.lower()

    relevant = [h for h in headlines if coin_name in h or coin_full in h or "crypto" in h]
    if not relevant:
        relevant = headlines

    pos   = sum(sum(1 for w in POSITIVE_WORDS if w in h) for h in relevant)
    neg   = sum(sum(1 for w in NEGATIVE_WORDS if w in h) for h in relevant)
    total = pos + neg
    score = round((pos - neg) / max(total, 1) * 100, 1)
    label = "Positive" if score > 10 else "Negative" if score < -10 else "Neutral"

    print(f"[News] {symbol} sentiment={score} ({label}) articles={len(relevant)}", flush=True)
    return {"score": score, "label": label, "headline_count": len(relevant)}


# ── BTC dominance ──────────────────────────────────────────────────────────────
def fetch_btc_dominance():
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        if r.ok:
            return round(r.json()["data"]["market_cap_percentage"]["btc"], 2)
    except Exception as e:
        print(f"[Dominance] Failed: {e}", flush=True)
    return None


# ── Data fetcher ───────────────────────────────────────────────────────────────
def fetch_candles(symbol, interval="1h", limit=384):
    binance_symbol = COINS[symbol]
    urls = [
        "https://api1.binance.com/api/v3/klines",
        "https://api2.binance.com/api/v3/klines",
        "https://api3.binance.com/api/v3/klines",
        "https://api.binance.com/api/v3/klines",
    ]
    params = {"symbol": binance_symbol, "interval": interval, "limit": limit}
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
            print(f"[Data] {symbol} {interval} OK {len(df)} candles", flush=True)
            return df
        except Exception as e:
            print(f"[Data] {url} failed: {e}", flush=True)
    raise RuntimeError(f"Could not fetch {interval} candles for {symbol}")


# ── Kronos prediction ──────────────────────────────────────────────────────────
def kronos_predict(df, pred_len=24):
    last_time   = df["timestamps"].iloc[-1]
    x_df        = df[["open","high","low","close","volume"]].copy()
    x_timestamp = df["timestamps"].copy().reset_index(drop=True)
    future_times = pd.date_range(
        start=last_time + pd.Timedelta(hours=1),
        periods=pred_len, freq="1h", tz="UTC"
    )
    y_timestamp = pd.Series(future_times).reset_index(drop=True)

    all_closes, all_highs, all_lows = [], [], []
    for i in range(MONTE_CARLO_N):
        p = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=pred_len, T=0.7, top_p=0.9, sample_count=1
        )
        all_closes.append(p["close"].values)
        all_highs.append(p["high"].values)
        all_lows.append(p["low"].values)

    closes = np.array(all_closes)
    return {
        "mean":        closes.mean(axis=0),
        "upper":       np.percentile(closes, 90, axis=0),
        "lower":       np.percentile(closes, 10, axis=0),
        "std":         closes.std(axis=0),
        "closes":      closes,
        "future_times": future_times,
    }


# ── Signal interpretation (FIX #1) ────────────────────────────────────────────
def interpret_signals(upside_prob, indicators, fear_greed, news, onchain, btc_dominance, symbol):
    """
    FIX #1: Kronos upside_prob is the PRIMARY signal.
    Auxiliary signals are FILTERS and CONTEXT — they do not mathematically
    alter the probability. Instead they generate qualitative warnings/confirmations.
    """
    rsi      = indicators["rsi"]
    macd_h   = indicators["macd_hist"]
    bb       = indicators["bb_pct"]
    vol_r    = indicators["vol_ratio"]
    fg       = fear_greed["value"]
    news_s   = news["score"]

    # Kronos direction
    kronos_bullish = upside_prob >= 50

    # Auxiliary confirmations (do they agree with Kronos?)
    confirmations = []
    warnings      = []

    # RSI
    if rsi < 30:
        confirmations.append("RSI oversold — potential bounce")
    elif rsi > 70:
        if kronos_bullish:
            warnings.append("RSI overbought — Kronos bullish but may be stretched")
        else:
            confirmations.append("RSI overbought — confirms bearish pressure")

    # MACD
    if macd_h > 0 and kronos_bullish:
        confirmations.append("MACD bullish — confirms Kronos signal")
    elif macd_h < 0 and not kronos_bullish:
        confirmations.append("MACD bearish — confirms Kronos signal")
    elif macd_h > 0 and not kronos_bullish:
        warnings.append("MACD bullish but Kronos bearish — mixed signal")
    elif macd_h < 0 and kronos_bullish:
        warnings.append("MACD bearish but Kronos bullish — mixed signal")

    # Bollinger Bands
    if bb < 0.2:
        confirmations.append("Price near lower BB — oversold zone")
    elif bb > 0.8:
        warnings.append("Price near upper BB — overbought zone")

    # Fear & Greed
    if fg <= 25:
        confirmations.append(f"Extreme Fear ({fg}) — historically good buy zone")
    elif fg >= 75:
        warnings.append(f"Extreme Greed ({fg}) — historically risky zone")

    # News
    if news_s > 10:
        confirmations.append(f"News sentiment positive ({news_s:.0f})")
    elif news_s < -10:
        warnings.append(f"News sentiment negative ({news_s:.0f})")

    # BTC dominance for altcoins
    if btc_dominance and symbol != "BTC":
        if btc_dominance > 57:
            warnings.append(f"BTC dominance high ({btc_dominance}%) — unfavorable for alts")
        elif btc_dominance < 45:
            confirmations.append(f"BTC dominance low ({btc_dominance}%) — alt season favorable")

    # Overall context label
    n_conf = len(confirmations)
    n_warn = len(warnings)
    if n_conf >= 3 and n_warn == 0:
        context = "Strong confirmation"
    elif n_conf > n_warn:
        context = "Mostly confirmed"
    elif n_warn > n_conf:
        context = "Caution — mixed signals"
    else:
        context = "Neutral context"

    return {
        "confirmations": confirmations,
        "warnings":      warnings,
        "context":       context,
        "n_confirm":     n_conf,
        "n_warn":        n_warn,
    }


# ── Core prediction ────────────────────────────────────────────────────────────
def run_prediction(symbol):
    global cache

    df_1h      = fetch_candles(symbol, "1h", LOOKBACK_1H)
    last_price = float(df_1h["close"].iloc[-1])
    last_time  = df_1h["timestamps"].iloc[-1]

    indicators    = compute_indicators(df_1h)
    fear_greed    = fetch_fear_greed() if symbol == "BTC" else {"value": 50, "label": "N/A"}
    onchain       = fetch_onchain() if symbol == "BTC" else {}
    news          = fetch_news_sentiment(symbol)
    btc_dominance = fetch_btc_dominance() if symbol != "BTC" else None

    print(f"[Kronos] Running 1h MC N={MONTE_CARLO_N} for {symbol}...", flush=True)
    pred_1h = kronos_predict(df_1h, pred_len=PRED_LEN)

    print(f"[Kronos] Running 4h MC N={MONTE_CARLO_N} for {symbol}...", flush=True)
    has_4h = False
    mean_4h_resampled = pred_1h["mean"]
    try:
        df_4h   = fetch_candles(symbol, "4h", LOOKBACK_4H)
        pred_4h = kronos_predict(df_4h, pred_len=6)
        # Smooth interpolation (not step function)
        x_old             = np.arange(6) * 4
        x_new             = np.arange(24)
        mean_4h_resampled = np.interp(x_new, x_old, pred_4h["mean"])
        has_4h            = True
        print(f"[Kronos] 4h done.", flush=True)
    except Exception as e:
        print(f"[Kronos] 4h failed: {e}", flush=True)

    # Ensemble mean (60% 1h, 40% 4h)
    mean_ensemble = (0.6 * pred_1h["mean"] + 0.4 * mean_4h_resampled)

    # ── PRIMARY SIGNAL: Kronos upside probability ──────────────────────────────
    final_prices = pred_1h["closes"][:, -1]
    upside_prob  = float((final_prices > last_price).mean()) * 100

    # Spread-based confidence
    spread_pct = (pred_1h["upper"] - pred_1h["lower"]) / last_price * 100
    confidence = round(max(0, 100 - spread_pct.mean() * 2), 1)

    # Volatility
    std_pct      = pred_1h["std"] / last_price * 100
    hist_vol_pct = float(df_1h["close"].pct_change().dropna().std() * 100)
    vol_amp_prob = float((std_pct > hist_vol_pct).mean()) * 100

    # FIX #1: Signal interpretation — qualitative, not additive
    signal_context = interpret_signals(
        upside_prob, indicators, fear_greed, news, onchain, btc_dominance, symbol
    )

    result = {
        "updated_at":      datetime.now(timezone.utc).isoformat(),
        "symbol":          f"{symbol}/USDT",
        "coin":            symbol,
        "last_price":      last_price,
        "last_time":       str(last_time),
        "pred_len":        PRED_LEN,
        "lookback":        LOOKBACK_1H,
        "monte_carlo_n":   MONTE_CARLO_N,
        "model":           MODEL_NAME,
        # PRIMARY SIGNAL — Kronos only, pure probabilistic output
        "upside_prob":     round(upside_prob, 1),
        "confidence":      confidence,
        "vol_amp_prob":    round(vol_amp_prob, 1),
        "has_4h":          has_4h,
        # CONTEXT — auxiliary signals as qualitative filters
        "signal_context":  signal_context,
        "indicators":      indicators,
        "fear_greed":      fear_greed,
        "news":            news,
        "onchain":         onchain,
        "btc_dominance":   btc_dominance,
        "forecast": {
            "timestamps": [str(t) for t in pred_1h["future_times"]],
            "mean_close": [round(v, 4) for v in mean_ensemble.tolist()],
            "mean_1h":    [round(v, 4) for v in pred_1h["mean"].tolist()],
            "upper":      [round(v, 4) for v in pred_1h["upper"].tolist()],
            "lower":      [round(v, 4) for v in pred_1h["lower"].tolist()],
        },
        "history": {
            "timestamps": [str(t) for t in df_1h["timestamps"].tolist()],
            "close":      [round(v, 4) for v in df_1h["close"].tolist()],
        }
    }

    # Convert all numpy types to native Python for JSON serialization
    result = json.loads(json.dumps(result, default=lambda x: float(x) if hasattr(x, 'item') else str(x)))

    with cache_lock:
        cache[symbol] = result
    save_cache()

    print(f"[Kronos] {symbol} DONE. Upside={upside_prob:.1f}% Conf={confidence:.1f}% Context={signal_context['context']}", flush=True)
    return result


def load_model():
    global predictor, model_ready, model_error
    try:
        load_cache_from_disk()
        print("[Kronos] Loading tokenizer...", flush=True)
        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
        print("[Kronos] Loading Kronos-base...", flush=True)
        model = Kronos.from_pretrained(MODEL_NAME)
        model.eval()
        predictor   = KronosPredictor(model, tokenizer, max_context=512)
        model_ready = True
        print("[Kronos] Model ready!", flush=True)
        threading.Thread(target=worker, daemon=True).start()
        task_queue.put("BTC")
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
        "running_since": {k: v for k, v in running_since.items() if running.get(k)},
    })

@app.route("/cache/<symbol>")
def get_cache(symbol):
    symbol = symbol.upper()
    if symbol not in COINS:
        return jsonify({"error": f"Unknown symbol {symbol}"}), 400
    if symbol not in cache:
        return jsonify({
            "error":      f"No prediction yet for {symbol}.",
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
    if running.get(symbol, False) or symbol in list(task_queue.queue):
        return jsonify({"status": "already_running_or_queued"})
    task_queue.put(symbol)
    return jsonify({"status": "queued"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
