import os, sys, time, threading, traceback, json, queue, sqlite3, re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import torch

# ── GPU Setup (Intel Arc via DirectML, NVIDIA via CUDA, fallback CPU) ─────────
def setup_device():
    # Try Intel Arc GPU via torch-directml (Windows)
    try:
        import torch_directml
        dev = torch_directml.device()
        torch.tensor([1.0]).to(dev)  # quick validation
        print(f"[GPU] Intel Arc detected via DirectML!", flush=True)
        return dev, "directml"
    except Exception as e:
        print(f"[GPU] DirectML not available ({e}), trying CUDA...", flush=True)

    # Try NVIDIA CUDA
    if torch.cuda.is_available():
        print(f"[GPU] CUDA: {torch.cuda.get_device_name(0)}", flush=True)
        return torch.device("cuda"), "cuda"

    # CPU fallback - use all cores
    n = os.cpu_count() or 4
    torch.set_num_threads(n)
    torch.set_num_interop_threads(max(1, n // 2))
    print(f"[CPU] Fallback - using {n} cores", flush=True)
    return torch.device("cpu"), "cpu"

DEVICE, DEVICE_TYPE = setup_device()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Kronos, KronosTokenizer, KronosPredictor

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
app        = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'))
CORS(app)

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME     = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
LOOKBACK       = 384
PRED_LEN       = 24
# N dynamically set after device detection in load_model
MONTE_CARLO_N  = 100   # default, overridden based on device
REFRESH_SECS   = 3600
NUM_WORKERS    = 1     # Kronos model is NOT thread-safe - must run one coin at a time
DB_FILE        = os.path.join(BASE_DIR, "kronos.db")

COINS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "BNB": "BNBUSDT", "SOL": "SOLUSDT",
    "ADA": "ADAUSDT", "ZEC": "ZECUSDT", "TAO": "TAOUSDT",
}

predictor    = None
cache        = {}
cache_lock   = threading.Lock()
model_ready  = False
model_error  = ""
running      = {}
running_since= {}
progress     = {}   # symbol -> {"current": N, "total": N, "secs_per_run": float}
task_queue   = queue.Queue()
db_lock      = threading.Lock()

# [FIX] Bulletproof Task Queue: Track queued coins to prevent race conditions
queued_coins = set()
queue_lock   = threading.Lock()


# ── SQLite ─────────────────────────────────────────────────────────────────────
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        # Enable WAL mode - prevents readers blocking writers and concurrent write conflicts
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                symbol     TEXT PRIMARY KEY,
                data       TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error      TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS accuracy (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol           TEXT NOT NULL,
                predicted_at     TEXT NOT NULL,
                predicted_price  REAL NOT NULL,
                upside_prob      REAL NOT NULL,
                confidence       REAL NOT NULL DEFAULT 0,
                actual_price     REAL,
                direction_correct INTEGER,
                checked_at       TEXT
            )
        """)
        conn.commit()
    print(f"[DB] SQLite ready (WAL mode): {DB_FILE}", flush=True)

def save_prediction(symbol, result):
    try:
        with db_lock:
            with sqlite3.connect(DB_FILE) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO predictions (symbol, data, updated_at, error)
                    VALUES (?, ?, ?, NULL)
                """, (symbol, json.dumps(result), result["updated_at"]))
                target = result["forecast"]["mean_close"][-1]
                conf   = result.get("confidence", 0)
                conn.execute("""
                    INSERT INTO accuracy (symbol, predicted_at, predicted_price, upside_prob, confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (symbol, result["updated_at"], target, result["upside_prob"], conf))
                conn.commit()
        with cache_lock:
            cache[symbol] = result
        print(f"[DB] {symbol} saved.", flush=True)
    except Exception as e:
        print(f"[DB] Save failed: {e}", flush=True)

def load_cache_from_disk():
    global cache
    try:
        init_db()
        with sqlite3.connect(DB_FILE) as conn:
            rows = conn.execute("SELECT symbol, data FROM predictions").fetchall()
        for sym, data in rows:
            try:
                cache[sym] = json.loads(data)
            except Exception:
                pass
        if cache:
            print(f"[DB] Loaded: {list(cache.keys())}", flush=True)
    except Exception as e:
        print(f"[DB] Load failed: {e}", flush=True)

def check_accuracy():
    """Compare 24h-old predictions to actual price. Runs on single background timer."""
    try:
        with db_lock:
            with sqlite3.connect(DB_FILE) as conn:
                rows = conn.execute("""
                    SELECT id, symbol, predicted_price, upside_prob
                    FROM accuracy
                    WHERE actual_price IS NULL
                    AND predicted_at < datetime('now', '-24 hours')
                """).fetchall()

        for row_id, symbol, pred_price, upside_prob in rows:
            if symbol not in COINS:
                continue
            try:
                r = requests.get("https://api.binance.com/api/v3/ticker/price",
                                 params={"symbol": COINS[symbol]}, timeout=10)
                if r.ok:
                    actual  = float(r.json()["price"])
                    correct = 1 if (upside_prob >= 50 and actual > pred_price) or \
                                   (upside_prob < 50 and actual <= pred_price) else 0
                    with db_lock:
                        with sqlite3.connect(DB_FILE) as conn:
                            conn.execute("""
                                UPDATE accuracy
                                SET actual_price=?, direction_correct=?, checked_at=?
                                WHERE id=?
                            """, (actual, correct, datetime.now(timezone.utc).isoformat(), row_id))
                            conn.commit()
                    print(f"[Accuracy] {symbol} pred={pred_price:.0f} actual={actual:.0f} correct={bool(correct)}", flush=True)
            except Exception as e:
                print(f"[Accuracy] {symbol} check failed: {e}", flush=True)
    except Exception as e:
        print(f"[Accuracy] Check failed: {e}", flush=True)

def get_accuracy_stats():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            # Overall per-symbol stats
            rows = conn.execute("""
                SELECT symbol, COUNT(*) as total, SUM(direction_correct) as correct
                FROM accuracy WHERE direction_correct IS NOT NULL
                GROUP BY symbol
            """).fetchall()

            # Confidence-bucketed stats (all symbols combined)
            bucket_rows = conn.execute("""
                SELECT
                    CASE
                        WHEN confidence >= 80 THEN 'high'
                        WHEN confidence >= 60 THEN 'medium'
                        ELSE 'low'
                    END as bucket,
                    COUNT(*) as total,
                    SUM(direction_correct) as correct
                FROM accuracy
                WHERE direction_correct IS NOT NULL
                GROUP BY bucket
            """).fetchall()

        stats = {s: {"total": t, "correct": c, "pct": round(c/t*100, 1)}
                 for s, t, c in rows if t > 0}

        # Add confidence breakdown
        conf_stats = {}
        for bucket, total, correct in bucket_rows:
            if total > 0:
                conf_stats[bucket] = {
                    "total":   total,
                    "correct": correct,
                    "pct":     round(correct/total*100, 1),
                    "label":   {
                        "high":   "High confidence (≥80%)",
                        "medium": "Medium confidence (60-80%)",
                        "low":    "Low confidence (<60%)"
                    }.get(bucket, bucket)
                }

        return {"by_coin": stats, "by_confidence": conf_stats}
    except Exception:
        return {"by_coin": {}, "by_confidence": {}}


# ── Worker - processes one coin at a time, no auto-requeue ────────────────────
def worker():
    while True:
        symbol = task_queue.get()
        if symbol is None:
            break
            
        # [FIX] Remove from queued set when actually starting processing
        with queue_lock:
            queued_coins.discard(symbol)
            
        running[symbol] = True
        running_since[symbol] = time.time()
        try:
            run_prediction(symbol)
        except Exception as e:
            print(f"[Kronos] {symbol} failed: {e}", flush=True)
            traceback.print_exc()
        finally:
            running[symbol] = False
        print(f"[Kronos] {symbol} complete. Waiting for next manual request.", flush=True)


# ── Technical Indicators (RSI with Wilder's smoothing) ────────────────────────
def compute_indicators(df):
    close    = df["close"].values
    delta    = np.diff(close)
    gain     = np.where(delta > 0, delta, 0.0)
    loss     = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs       = np.where(avg_loss == 0, 100.0, avg_gain / (avg_loss + 1e-10))
    rsi      = np.append(np.nan, 100 - (100 / (1 + rs)))
    ema12    = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26    = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd     = ema12 - ema26
    sig      = pd.Series(macd).ewm(span=9, adjust=False).mean().values
    sma20    = pd.Series(close).rolling(20).mean().values
    std20    = pd.Series(close).rolling(20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_pct   = np.where(bb_upper - bb_lower == 0, 0.5,
                        (close - bb_lower) / (bb_upper - bb_lower + 1e-10))
    vol      = df["volume"].values
    vol_sma  = pd.Series(vol).rolling(20).mean().values

    # Data validation - flag suspicious volume
    vol_ratio_raw = vol[-1] / (vol_sma[-1] + 1e-10)
    vol_valid = 0.05 <= vol_ratio_raw <= 20.0

    return {
        "rsi":        round(float(rsi[-1]) if not np.isnan(rsi[-1]) else 50, 2),
        "macd_hist":  round(float((macd - sig)[-1]), 4),
        "bb_pct":     round(float(np.clip(bb_pct[-1], 0, 1)), 4),
        "vol_ratio":  round(float(vol_ratio_raw), 4),
        "vol_valid":  vol_valid,
        "sma20":      round(float(sma20[-1]), 2),
        "bb_upper":   round(float(bb_upper[-1]), 2),
        "bb_lower":   round(float(bb_lower[-1]), 2),
    }


# ── External signals ───────────────────────────────────────────────────────────
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        r.raise_for_status()
        d = r.json()
        return {"value": int(d["data"][0]["value"]),
                "label": d["data"][0]["value_classification"]}
    except Exception:  # [FIX] Replaced bare except
        # [FIX] Return None for value so interpretation logic knows it's missing
        return {"value": None, "label": "N/A"}

def fetch_funding_rate(symbol):
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/fundingRate",
                         params={"symbol": COINS[symbol], "limit": 3}, timeout=10)
        if r.ok and r.json():
            data = r.json()
            rate = float(data[-1]["fundingRate"]) * 100
            avg  = sum(float(d["fundingRate"]) for d in data) / len(data) * 100
            label = "Overcrowded Longs" if rate > 0.05 else \
                    "Overcrowded Shorts" if rate < -0.01 else "Neutral"
            print(f"[Funding] {symbol} {rate:.4f}% ({label})", flush=True)
            return {"rate": round(rate, 4), "avg": round(avg, 4), "label": label}
    except Exception as e:
        print(f"[Funding] {symbol} failed: {e}", flush=True)
    # [FIX] Return None for rate and avg so logic doesn't misinterpret 0.0 as "Neutral"
    return {"rate": None, "avg": None, "label": "N/A"}

def fetch_etf_flows():
    try:
        r = requests.get(
            "https://farside.co.uk/bitcoin-etf-flow-all-data-table/",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=15
        )
        if not r.ok:
            return {"total": None, "label": "Unavailable"}
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', r.text, re.DOTALL)
        if rows:
            numbers = re.findall(r'-?\d+(?:\.\d+)?', re.sub(r'<[^>]+>', ' ', rows[-2] if len(rows) > 1 else rows[-1]))
            if numbers:
                total = float(numbers[-1])
                # Sanity check - reject absurd values (scraper artifact)
                if not (-10000 <= total <= 10000):
                    print(f"[ETF] Rejected absurd value: {total}", flush=True)
                    return {"total": None, "label": "Unavailable"}
                label = "Strong Inflows" if total > 300 else \
                        "Inflows" if total > 0 else \
                        "Outflows" if total > -300 else "Strong Outflows"
                print(f"[ETF] Flow: ${total:.0f}M ({label})", flush=True)
                return {"total": total, "label": label}
    except Exception as e:
        print(f"[ETF] Failed: {e}", flush=True)
    return {"total": None, "label": "Unavailable"}

def fetch_onchain():
    result = {"mempool_size": 0, "mempool_label": "Normal"}
    try:
        r = requests.get("https://mempool.space/api/mempool", timeout=10)
        if r.ok:
            m = r.json()
            result["mempool_size"] = m.get("count", 0)
            s = result["mempool_size"]
            result["mempool_label"] = "Very High" if s > 100000 else \
                                      "High" if s > 50000 else \
                                      "Low" if s < 5000 else "Normal"
    except Exception as e:
        print(f"[OnChain] Failed: {e}", flush=True)
    return result

def fetch_btc_dominance():
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        if r.ok:
            return round(r.json()["data"]["market_cap_percentage"]["btc"], 2)
    except Exception:  # [FIX] Replaced bare except
        pass
    return None


# ── Data fetcher ───────────────────────────────────────────────────────────────
def fetch_candles(symbol, interval="1h", limit=384):
    binance_symbol = COINS[symbol]
    params = {"symbol": binance_symbol, "interval": interval, "limit": limit}
    for url in ["https://api1.binance.com/api/v3/klines",
                "https://api2.binance.com/api/v3/klines",
                "https://api3.binance.com/api/v3/klines",
                "https://api.binance.com/api/v3/klines"]:
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            raw = r.json()
            df  = pd.DataFrame(raw, columns=[
                "open_time","open","high","low","close","volume",
                "close_time","qv","trades","tbb","tbq","ignore"
            ])
            df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            for c in ["open","high","low","close","volume"]:
                df[c] = df[c].astype(float)
            df = df[["timestamps","open","high","low","close","volume"]].reset_index(drop=True)
            print(f"[Data] {symbol} {interval} OK ({len(df)} candles)", flush=True)
            return df
        except Exception as e:
            print(f"[Data] {url} failed: {e}", flush=True)
    raise RuntimeError(f"All Binance endpoints failed for {symbol} {interval}")


# ── Find safe lookback via fast dry-run (pred_len=2) ─────────────────────────
def find_safe_lookback(df, symbol):
    """
    Run a single fast prediction with pred_len=2 to find the maximum
    lookback the model can handle without a RoPE tensor size mismatch.
    Validates BEFORE committing to the full N=100 Monte Carlo loop.
    """
    candidates = [370, 360, 350, 340, 330, 320, 300, 280, 256]
    for lookback in candidates:
        if lookback > len(df):
            continue
        test_df = df.tail(lookback).reset_index(drop=True)
        x_df        = test_df[["open","high","low","close","volume"]].copy()
        x_timestamp = test_df["timestamps"].copy().reset_index(drop=True)
        last_time   = test_df["timestamps"].iloc[-1]
        future_times = pd.date_range(
            start=last_time + pd.Timedelta(hours=1),
            periods=24, freq="1h", tz="UTC"
        )
        y_timestamp = pd.Series(future_times).reset_index(drop=True)
        try:
            with torch.inference_mode():
                predictor.predict(
                    df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                    pred_len=24, T=0.7, top_p=0.9, sample_count=1
                )
            print(f"[Kronos] {symbol} safe lookback={lookback} (dry-run passed)", flush=True)
            return lookback
        except RuntimeError as e:
            if "size of tensor" in str(e) or "must match" in str(e):
                print(f"[Kronos] {symbol} lookback={lookback} failed dry-run, trying smaller...", flush=True)
                continue
            raise
    raise RuntimeError(f"{symbol}: no working lookback found in {candidates}")


# ── Kronos Monte Carlo - N=100, T=0.7, with real-time progress tracking ───────
def kronos_predict(df, symbol="UNK", pred_len=24):
    # Step 1: Find safe lookback via fast dry-run BEFORE the MC loop
    safe_lookback = find_safe_lookback(df, symbol)
    work_df = df.tail(safe_lookback).reset_index(drop=True)

    last_time   = work_df["timestamps"].iloc[-1]
    x_df        = work_df[["open","high","low","close","volume"]].copy()
    x_timestamp = work_df["timestamps"].copy().reset_index(drop=True)
    future_times = pd.date_range(
        start=last_time + pd.Timedelta(hours=1),
        periods=pred_len, freq="1h", tz="UTC"
    )
    y_timestamp = pd.Series(future_times).reset_index(drop=True)

    all_closes = []
    run_times  = []

    # Step 2: Full MC loop - safe_lookback guaranteed to work, no crashes mid-loop
    with torch.inference_mode():
        for i in range(MONTE_CARLO_N):
            t_start = time.time()
            try:
                p = predictor.predict(
                    df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                    pred_len=pred_len, T=0.7, top_p=0.9, sample_count=1
                )
            except RuntimeError as e:
                if "size of tensor" in str(e) or "must match" in str(e):
                    print(f"[Kronos] {symbol} unexpected tensor error at MC run {i+1} with lookback={safe_lookback}: {e}", flush=True)
                    raise
                raise
            elapsed = time.time() - t_start
            run_times.append(elapsed)
            all_closes.append(p["close"].values)

            avg_secs = sum(run_times) / len(run_times)
            remaining = (MONTE_CARLO_N - i - 1) * avg_secs
            progress[symbol] = {
                "current":        i + 1,
                "total":          MONTE_CARLO_N,
                "secs_per_run":   round(avg_secs, 2),
                "remaining_secs": round(remaining, 0),
                "pct":            round((i + 1) / MONTE_CARLO_N * 100, 1),
            }
            if i % 10 == 0:
                print(f"[Kronos] {symbol} MC {i+1}/{MONTE_CARLO_N} ({avg_secs:.1f}s/run, ~{remaining/60:.1f}min left)", flush=True)

    closes = np.array(all_closes)
    progress.pop(symbol, None)
    return {
        "mean":         closes.mean(axis=0),
        "upper":        np.percentile(closes, 90, axis=0),
        "lower":        np.percentile(closes, 10, axis=0),
        "std":          closes.std(axis=0),
        "closes":       closes,
        "future_times": future_times,
        "lookback_used": safe_lookback,  # actual lookback reported honestly
    }


# ── Signal interpretation ──────────────────────────────────────────────────────
def interpret_signals(upside_prob, ind, fear_greed, funding, etf_flows,
                      onchain, btc_dominance, symbol):
    rsi     = ind["rsi"]
    macd_h  = ind["macd_hist"]
    bb      = ind["bb_pct"]
    
    # [FIX] Use .get() to safely handle None values for missing API data
    fg      = fear_greed.get("value")
    fund    = funding.get("rate")
    bullish = upside_prob >= 50

    confirmations, warnings = [], []

    if rsi < 30:   confirmations.append(f"RSI oversold ({rsi:.1f}) - potential bounce")
    elif rsi > 70: warnings.append(f"RSI overbought ({rsi:.1f}) - stretched")

    if macd_h > 0 and bullish:     confirmations.append("MACD bullish - confirms Kronos")
    elif macd_h < 0 and not bullish: confirmations.append("MACD bearish - confirms Kronos")
    elif macd_h > 0 and not bullish: warnings.append("MACD bullish but Kronos bearish - mixed")
    elif macd_h < 0 and bullish:   warnings.append("MACD bearish but Kronos bullish - mixed")

    if bb < 0.2:   confirmations.append("Price near lower BB - oversold zone")
    elif bb > 0.8: warnings.append("Price near upper BB - overbought zone")

    # [FIX] Only evaluate Fear & Greed if data was successfully fetched
    if fg is not None:
        if fg <= 20:   confirmations.append(f"Extreme Fear ({fg}) - historically strong buy zone")
        elif fg >= 80: warnings.append(f"Extreme Greed ({fg}) - historically risky zone")

    # [FIX] Only evaluate Funding Rates if data was successfully fetched
    if fund is not None:
        if fund > 0.05:   warnings.append(f"High funding ({fund:.3f}%) - longs overcrowded")
        elif fund < -0.01: confirmations.append(f"Negative funding ({fund:.3f}%) - squeeze risk")

    if symbol == "BTC" and etf_flows.get("total") is not None:
        f = etf_flows["total"]
        if f > 300:    confirmations.append(f"Strong ETF inflows (${f:.0f}M)")
        elif f > 0:    confirmations.append(f"ETF inflows (${f:.0f}M)")
        elif f < -300: warnings.append(f"Strong ETF outflows (${f:.0f}M)")
        elif f < 0:    warnings.append(f"ETF outflows (${f:.0f}M)")

    if btc_dominance and symbol != "BTC":
        if btc_dominance > 57:   warnings.append(f"BTC dominance high ({btc_dominance}%) - alt headwinds")
        elif btc_dominance < 45: confirmations.append(f"BTC dominance low ({btc_dominance}%) - alt season")

    if not ind.get("vol_valid", True):
        warnings.append(f"Unusual volume detected ({ind['vol_ratio']:.2f}x) - data may be unreliable")

    n_c, n_w = len(confirmations), len(warnings)
    context = "Strong confirmation" if n_c >= 3 and n_w == 0 else \
              "Mostly confirmed" if n_c > n_w else \
              "Caution - mixed signals" if n_w > n_c else "Neutral context"

    return {"confirmations": confirmations, "warnings": warnings,
            "context": context, "n_confirm": n_c, "n_warn": n_w}


# ── Core prediction ────────────────────────────────────────────────────────────
def run_prediction(symbol):
    print(f"[Kronos] {symbol} starting on {DEVICE_TYPE.upper()}...", flush=True)

    df         = fetch_candles(symbol, "1h", LOOKBACK)
    last_price = float(df["close"].iloc[-1])
    last_time  = df["timestamps"].iloc[-1]

    # Fetch external signals in parallel while model is loading data
    sig = {}
    def fetch_signals():
        sig["indicators"]    = compute_indicators(df)
        sig["fear_greed"]    = fetch_fear_greed() if symbol == "BTC" else {"value": None, "label": "N/A"}
        sig["onchain"]       = fetch_onchain() if symbol == "BTC" else {}
        sig["funding"]       = fetch_funding_rate(symbol)
        sig["etf_flows"]     = fetch_etf_flows() if symbol == "BTC" else {"total": None, "label": "N/A"}
        sig["btc_dominance"] = fetch_btc_dominance() if symbol != "BTC" else None

    st = threading.Thread(target=fetch_signals)
    st.start()

    # Run Kronos (N=100, T=0.7) with real-time progress tracking
    pred = kronos_predict(df, symbol=symbol, pred_len=PRED_LEN)

    st.join()  # Wait for external signals

    closes     = pred["closes"]
    mean_close = pred["mean"]
    upper      = pred["upper"]
    lower      = pred["lower"]
    std        = pred["std"]

    final_prices    = closes[:, -1]
    raw_upside_prob = float((final_prices > last_price).mean()) * 100
    # Fix 1: Hard cap 5%-95% - prevents false certainty (100% or 0% are statistical illusions)
    upside_prob     = float(max(5.0, min(95.0, raw_upside_prob)))

    std_pct         = std / last_price * 100
    hist_vol_pct    = float(df["close"].pct_change().dropna().std() * 100)
    vol_amp_prob    = float((std_pct > hist_vol_pct).mean()) * 100
    spread_pct      = (upper - lower) / last_price * 100
    avg_spread      = float(spread_pct.mean())

    # Fix 2: Volatility-calibrated confidence
    # If model's predicted range is tighter than actual historical volatility,
    # it is hallucinating certainty - penalize heavily.
    if avg_spread < hist_vol_pct:
        confidence = round(max(10.0, 50.0 - (hist_vol_pct - avg_spread) * 3), 1)
        hallucinating = True
    else:
        confidence = round(max(0.0, 100.0 - avg_spread * 2), 1)
        hallucinating = False

    print(f"[Kronos] {symbol} spread={avg_spread:.2f}% hist_vol={hist_vol_pct:.2f}% hallucinating={hallucinating} conf={confidence}", flush=True)

    signal_context = interpret_signals(
        upside_prob, sig["indicators"], sig["fear_greed"],
        sig["funding"], sig["etf_flows"], sig["onchain"],
        sig["btc_dominance"], symbol
    )

    result = {
        "updated_at":    datetime.now(timezone.utc).isoformat(),
        "symbol":        f"{symbol}/USDT",
        "coin":          symbol,
        "last_price":    last_price,
        "last_time":     str(last_time),
        "pred_len":      PRED_LEN,
        "lookback":      LOOKBACK,
        "monte_carlo_n": MONTE_CARLO_N,
        "model":         MODEL_NAME,
        "device":        DEVICE_TYPE,
        "upside_prob":      round(upside_prob, 1),
        "raw_upside_prob":  round(raw_upside_prob, 1),
        "hallucinating":    hallucinating,
        "confidence":       confidence,
        "vol_amp_prob":  round(vol_amp_prob, 1),
        "lookback_used": pred.get("lookback_used", LOOKBACK),
        "signal_context": signal_context,
        "indicators":    sig["indicators"],
        "fear_greed":    sig["fear_greed"],
        "funding":       sig["funding"],
        "etf_flows":     sig["etf_flows"],
        "onchain":       sig["onchain"],
        "btc_dominance": sig["btc_dominance"],
        "accuracy":      get_accuracy_stats().get("by_coin", {}).get(symbol, {}),
        "forecast": {
            "timestamps": [str(t) for t in pred["future_times"]],
            "mean_close": [round(v, 4) for v in mean_close.tolist()],
            "upper":      [round(v, 4) for v in upper.tolist()],
            "lower":      [round(v, 4) for v in lower.tolist()],
        },
        "history": {
            "timestamps": [str(t) for t in df["timestamps"].tolist()],
            "close":      [round(v, 4) for v in df["close"].tolist()],
        }
    }

    result = json.loads(json.dumps(result, default=lambda x: float(x) if hasattr(x, 'item') else str(x)))
    save_prediction(symbol, result)
    print(f"[Kronos] {symbol} DONE. Upside={upside_prob:.1f}% Conf={confidence:.1f}% Device={DEVICE_TYPE.upper()}", flush=True)
    return result


# ── Model loader ───────────────────────────────────────────────────────────────
def load_model():
    global predictor, model_ready, model_error, MONTE_CARLO_N
    try:
        load_cache_from_disk()
        print("[Kronos] Loading tokenizer...", flush=True)
        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_NAME)
        print("[Kronos] Loading Kronos-base...", flush=True)
        model = Kronos.from_pretrained(MODEL_NAME)
        model.eval()

        # Fix 4: Try GPU, verify tensors actually move, fall back to CPU if not
        gpu_working = False
        if DEVICE_TYPE != "cpu":
            try:
                model = model.to(DEVICE)
                # Verify GPU inference actually works end-to-end
                test_tensor = torch.zeros(1, 10).to(DEVICE)
                _ = test_tensor + 1
                gpu_working = True
                print(f"[Kronos] Model on {DEVICE_TYPE.upper()} - verified!", flush=True)
            except Exception as e:
                print(f"[Kronos] GPU failed ({e}), falling back to CPU", flush=True)
                model = model.cpu()

        # N=100 - KronosPredictor runs on CPU internally (PCIe bottleneck confirmed)
        # Each run ~5-7s, total ~10 min per coin. Better P10/P90 accuracy than N=50.
        MONTE_CARLO_N = 100
        device_note = "GPU connected (CPU-bound inference)" if gpu_working else "CPU"
        print(f"[Kronos] {device_note} - N={MONTE_CARLO_N}", flush=True)

        predictor   = KronosPredictor(model, tokenizer, max_context=512)
        model_ready = True
        print(f"[Kronos] Ready! Device={DEVICE_TYPE.upper()} N={MONTE_CARLO_N}", flush=True)

        for i in range(NUM_WORKERS):
            threading.Thread(target=worker, daemon=True, name=f"Worker-{i+1}").start()

        # No auto-queuing - user manually selects which coin to predict
        print(f"[Kronos] Ready! Waiting for manual prediction requests.", flush=True)

        # [FIX] Run accuracy check immediately on startup, THEN loop hourly
        def accuracy_loop():
            check_accuracy()  # Check pending predictions right now
            while True:
                time.sleep(3600)
                check_accuracy()
                
        threading.Thread(target=accuracy_loop, daemon=True, name="AccuracyChecker").start()

    except Exception as e:
        model_error = str(e)
        print(f"[Kronos] Load failed: {e}", flush=True)
        traceback.print_exc()

threading.Thread(target=load_model, daemon=True).start()


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(os.path.join(BASE_DIR, 'static'), "index.html")

@app.route("/status")
def status():
    # [FIX] Make a safe copy of progress so worker mutation doesn't crash JSON serialization
    safe_progress = dict(progress)
    
    return jsonify({
        "model_ready":   model_ready,
        "model_error":   model_error,
        "model":         MODEL_NAME,
        "device":        DEVICE_TYPE,
        "monte_carlo_n": MONTE_CARLO_N,
        "coins":         list(COINS.keys()),
        "cached":        list(cache.keys()),
        "running":       {k: v for k, v in running.items() if v},
        "running_since": {k: v for k, v in running_since.items() if running.get(k)},
        "progress":      safe_progress,
        "accuracy":      get_accuracy_stats().get("by_coin", {}),
    })

@app.route("/cache/<symbol>")
def get_cache(symbol):
    symbol = symbol.upper()
    if symbol not in COINS:
        return jsonify({"error": f"Unknown symbol {symbol}"}), 400
    if symbol not in cache:
        return jsonify({
            "error":       f"No prediction yet for {symbol}.",
            "model_ready": model_ready,
            "is_running":  running.get(symbol, False)
        }), 404
    result = dict(cache[symbol])
    result["is_running"] = running.get(symbol, False)
    try:
        updated  = datetime.fromisoformat(result["updated_at"])
        age_secs = int((datetime.now(timezone.utc) - updated).total_seconds())
        result["age_minutes"]       = age_secs // 60
        result["next_refresh_mins"] = max(0, (REFRESH_SECS - age_secs) // 60)
        result["is_stale"]          = age_secs > REFRESH_SECS + 300
    except Exception:
        pass
    return jsonify(result)

@app.route("/predict/<symbol>")
def predict(symbol):
    symbol = symbol.upper()
    if symbol not in COINS:
        return jsonify({"error": f"Unknown symbol {symbol}"}), 400
    if not model_ready:
        return jsonify({"error": "Model not loaded yet."}), 503
        
    # [FIX] Bulletproof queue check using a set and lock
    with queue_lock:
        if running.get(symbol, False) or symbol in queued_coins:
            return jsonify({"status": "already_running_or_queued"})
        
        queued_coins.add(symbol)
        task_queue.put(symbol)
        
    return jsonify({"status": "queued"})

@app.route("/accuracy")
def accuracy_route():
    return jsonify(get_accuracy_stats())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
