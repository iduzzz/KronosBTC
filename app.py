import os, sys, time, threading, traceback, json, queue, sqlite3, re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import torch

# ── GPU Setup ─────────────────────────────────────────────────────────────────
def setup_device():
    try:
        import torch_directml
        dev = torch_directml.device()
        torch.tensor([1.0]).to(dev)
        print(f"[GPU] Intel Arc detected via DirectML!", flush=True)
        return dev, "directml"
    except Exception as e:
        print(f"[GPU] DirectML not available ({e}), trying CUDA...", flush=True)
    if torch.cuda.is_available():
        print(f"[GPU] CUDA: {torch.cuda.get_device_name(0)}", flush=True)
        return torch.device("cuda"), "cuda"
    n = os.cpu_count() or 4
    torch.set_num_threads(n)
    torch.set_num_interop_threads(max(1, n // 2))
    print(f"[CPU] Fallback - using {n} cores", flush=True)
    return torch.device("cpu"), "cpu"

DEVICE, DEVICE_TYPE = setup_device()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Kronos, KronosTokenizer, KronosPredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app      = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'))
CORS(app)

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME     = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
LOOKBACK       = 384
PRED_LEN       = 24
MONTE_CARLO_N  = 100
REFRESH_SECS   = 3600
NUM_WORKERS    = 1
DB_FILE        = os.path.join(BASE_DIR, "kronos.db")

COINS = {
    "ZEC": "ZECUSDT", "BTC": "BTCUSDT", "TAO": "TAOUSDT", "ETH": "ETHUSDT",
}

predictor    = None
cache        = {}
cache_lock   = threading.Lock()
model_ready  = False
model_error  = ""
running      = {}
running_since= {}
progress     = {}
task_queue   = queue.Queue()
db_lock      = threading.Lock()
queued_coins = set()
queue_lock   = threading.Lock()


# ── SQLite ─────────────────────────────────────────────────────────────────────
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
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
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol              TEXT NOT NULL,
                predicted_at        TEXT NOT NULL,
                entry_price         REAL NOT NULL DEFAULT 0,
                predicted_price     REAL NOT NULL,
                upside_prob         REAL NOT NULL,
                confidence          REAL NOT NULL DEFAULT 0,
                p10_at_prediction   REAL,
                p90_at_prediction   REAL,
                momentum_direction  INTEGER,
                carry_direction     INTEGER,
                actual_price        REAL,
                direction_correct   INTEGER,
                inside_band         INTEGER,
                momentum_correct    INTEGER,
                carry_correct       INTEGER,
                random_walk_correct INTEGER,
                checked_at          TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS request_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol       TEXT NOT NULL,
                requested_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol       TEXT NOT NULL,
                predicted_at TEXT NOT NULL,
                direction    TEXT NOT NULL,
                entry_price  REAL NOT NULL,
                stop_price   REAL NOT NULL,
                target_price REAL NOT NULL,
                size_pct     REAL NOT NULL,
                risk_pct     REAL NOT NULL,
                closed       INTEGER DEFAULT 0,
                exit_price   REAL,
                exit_reason  TEXT,
                pnl_pct      REAL,
                closed_at    TEXT
            )
        """)
        conn.commit()

    # Migrations for existing databases
    migrations = [
        "ALTER TABLE accuracy ADD COLUMN entry_price REAL NOT NULL DEFAULT 0",
        "ALTER TABLE accuracy ADD COLUMN p10_at_prediction REAL",
        "ALTER TABLE accuracy ADD COLUMN p90_at_prediction REAL",
        "ALTER TABLE accuracy ADD COLUMN momentum_direction INTEGER",
        "ALTER TABLE accuracy ADD COLUMN carry_direction INTEGER",
        "ALTER TABLE accuracy ADD COLUMN inside_band INTEGER",
        "ALTER TABLE accuracy ADD COLUMN momentum_correct INTEGER",
        "ALTER TABLE accuracy ADD COLUMN carry_correct INTEGER",
        "ALTER TABLE accuracy ADD COLUMN random_walk_correct INTEGER",
    ]
    for sql in migrations:
        try:
            with sqlite3.connect(DB_FILE) as conn:
                conn.execute(sql)
                conn.commit()
        except Exception:
            pass  # Column already exists

    print(f"[DB] SQLite ready (WAL mode): {DB_FILE}", flush=True)


def save_prediction(symbol, result):
    try:
        with db_lock:
            with sqlite3.connect(DB_FILE) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO predictions (symbol, data, updated_at, error)
                    VALUES (?, ?, ?, NULL)
                """, (symbol, json.dumps(result), result["updated_at"]))

                fc     = result["forecast"]
                target = fc["mean_close"][-1]
                p10    = fc["lower"][-1]
                p90    = fc["upper"][-1]
                entry  = result["last_price"]
                conf   = result.get("confidence", 0)

                # Baseline directions
                mom_dir   = result.get("momentum_direction")
                carry_dir = result.get("carry_direction")

                conn.execute("""
                    INSERT INTO accuracy
                    (symbol, predicted_at, entry_price, predicted_price, upside_prob,
                     confidence, p10_at_prediction, p90_at_prediction,
                     momentum_direction, carry_direction)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, result["updated_at"], entry, target,
                      result["upside_prob"], conf, p10, p90, mom_dir, carry_dir))
                conn.commit()

        with cache_lock:
            cache[symbol] = result
        print(f"[DB] {symbol} saved.", flush=True)
        log_paper_trade(symbol, result)
    except Exception as e:
        print(f"[DB] Save failed: {e}", flush=True)


def log_paper_trade(symbol, result):
    """Log paper trade when signal is LONG or SHORT. Auto-closes after 24h."""
    try:
        ps  = result.get("position_size")
        sig = result.get("signal_context", {}).get("trade_signal")
        if sig not in ("LONG", "SHORT") or not ps:
            return
        with db_lock:
            with sqlite3.connect(DB_FILE) as conn:
                conn.execute("""
                    INSERT INTO paper_trades
                    (symbol, predicted_at, direction, entry_price,
                     stop_price, target_price, size_pct, risk_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, result["updated_at"], sig, result["last_price"],
                      ps["stop_loss"], ps["take_profit"], ps["size_pct"], ps["risk_pct"]))
                conn.commit()
        print(f"[Paper] {symbol} {sig} trade logged. Entry={result['last_price']} Stop={ps['stop_loss']} Target={ps['take_profit']}", flush=True)
    except Exception as e:
        print(f"[Paper] Log failed: {e}", flush=True)


def close_paper_trades():
    """Close open paper trades: hit stop/target, or auto-close at 24h."""
    try:
        with db_lock:
            with sqlite3.connect(DB_FILE) as conn:
                open_trades = conn.execute("""
                    SELECT id, symbol, direction, entry_price, stop_price, target_price
                    FROM paper_trades WHERE closed=0
                    AND predicted_at < datetime('now', '-24 hours')
                """).fetchall()

        for trade in open_trades:
            tid, sym, direction, entry, stop, target = trade
            if sym not in COINS:
                continue
            try:
                r = requests.get("https://api.binance.com/api/v3/ticker/price",
                                 params={"symbol": COINS[sym]}, timeout=10)
                if not r.ok:
                    continue
                current = float(r.json()["price"])

                # Check stop and target
                if direction == "LONG":
                    if current <= stop:
                        exit_reason, exit_price = "STOP_HIT", stop
                    elif current >= target:
                        exit_reason, exit_price = "TARGET_HIT", target
                    else:
                        exit_reason, exit_price = "TIME_EXIT", current
                else:  # SHORT
                    if current >= stop:
                        exit_reason, exit_price = "STOP_HIT", stop
                    elif current <= target:
                        exit_reason, exit_price = "TARGET_HIT", target
                    else:
                        exit_reason, exit_price = "TIME_EXIT", current

                if direction == "LONG":
                    pnl_pct = (exit_price - entry) / entry * 100
                else:
                    pnl_pct = (entry - exit_price) / entry * 100

                with db_lock:
                    with sqlite3.connect(DB_FILE) as conn:
                        conn.execute("""
                            UPDATE paper_trades
                            SET closed=1, exit_price=?, exit_reason=?, pnl_pct=?, closed_at=?
                            WHERE id=?
                        """, (exit_price, exit_reason, round(pnl_pct, 3),
                              datetime.now(timezone.utc).isoformat(), tid))
                        conn.commit()
                print(f"[Paper] {sym} {direction} closed: {exit_reason} PnL={pnl_pct:.2f}%", flush=True)
            except Exception as e:
                print(f"[Paper] Close {tid} failed: {e}", flush=True)
    except Exception as e:
        print(f"[Paper] Close loop failed: {e}", flush=True)


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
    """Compare 24h-old predictions against actual prices."""
    try:
        with db_lock:
            with sqlite3.connect(DB_FILE) as conn:
                rows = conn.execute("""
                    SELECT id, symbol, entry_price, upside_prob,
                           p10_at_prediction, p90_at_prediction,
                           momentum_direction, carry_direction
                    FROM accuracy
                    WHERE actual_price IS NULL
                    AND predicted_at < datetime('now', '-24 hours')
                """).fetchall()

        for row in rows:
            (row_id, symbol, entry_price, upside_prob,
             p10, p90, mom_dir, carry_dir) = row
            if symbol not in COINS:
                continue
            try:
                r = requests.get("https://api.binance.com/api/v3/ticker/price",
                                 params={"symbol": COINS[symbol]}, timeout=10)
                if r.ok:
                    actual = float(r.json()["price"])

                    # Kronos direction accuracy
                    correct = 1 if (upside_prob >= 50 and actual > entry_price) or \
                                   (upside_prob < 50 and actual <= entry_price) else 0

                    # P10-P90 band hit
                    band_hit = None
                    if p10 is not None and p90 is not None:
                        band_hit = 1 if p10 <= actual <= p90 else 0

                    # Momentum baseline
                    mom_correct = None
                    if mom_dir is not None:
                        mom_correct = 1 if (mom_dir == 1 and actual > entry_price) or \
                                          (mom_dir == 0 and actual <= entry_price) else 0

                    # Carry baseline
                    carry_correct = None
                    if carry_dir is not None:
                        carry_correct = 1 if (carry_dir == 1 and actual > entry_price) or \
                                            (carry_dir == 0 and actual <= entry_price) else 0

                    # Random Walk baseline: predict price stays flat (correct if <1% move)
                    rw_correct = 1 if abs(actual - entry_price) / (entry_price + 1e-10) < 0.01 else 0

                    with db_lock:
                        with sqlite3.connect(DB_FILE) as conn:
                            conn.execute("""
                                UPDATE accuracy
                                SET actual_price=?, direction_correct=?, inside_band=?,
                                    momentum_correct=?, carry_correct=?, random_walk_correct=?, checked_at=?
                                WHERE id=?
                            """, (actual, correct, band_hit, mom_correct, carry_correct,
                                  rw_correct, datetime.now(timezone.utc).isoformat(), row_id))
                            conn.commit()

                    print(f"[Accuracy] {symbol} entry={entry_price:.2f} actual={actual:.2f} "
                          f"correct={bool(correct)} band={band_hit}", flush=True)
            except Exception as e:
                print(f"[Accuracy] {symbol} check failed: {e}", flush=True)
    except Exception as e:
        print(f"[Accuracy] Check failed: {e}", flush=True)


def get_accuracy_stats():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            # Per-symbol Kronos accuracy
            rows = conn.execute("""
                SELECT symbol, COUNT(*) as total, SUM(direction_correct) as correct
                FROM accuracy WHERE direction_correct IS NOT NULL
                GROUP BY symbol
            """).fetchall()

            # Confidence buckets
            bucket_rows = conn.execute("""
                SELECT
                    CASE
                        WHEN confidence >= 80 THEN 'high'
                        WHEN confidence >= 60 THEN 'medium'
                        ELSE 'low'
                    END as bucket,
                    COUNT(*) as total,
                    SUM(direction_correct) as correct
                FROM accuracy WHERE direction_correct IS NOT NULL
                GROUP BY bucket
            """).fetchall()

            # Baseline comparison
            base_row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(direction_correct) as kronos_correct,
                    SUM(momentum_correct) as mom_correct,
                    SUM(carry_correct) as carry_correct,
                    SUM(random_walk_correct) as rw_correct
                FROM accuracy
                WHERE direction_correct IS NOT NULL
            """).fetchone()

            # P10-P90 band hit rate
            band_row = conn.execute("""
                SELECT COUNT(*) as total, SUM(inside_band) as hits
                FROM accuracy WHERE inside_band IS NOT NULL
            """).fetchone()

        stats = {s: {"total": t, "correct": c, "pct": round(c/t*100, 1)}
                 for s, t, c in rows if t > 0}

        conf_stats = {}
        for bucket, total, correct in bucket_rows:
            if total > 0:
                conf_stats[bucket] = {
                    "total": total, "correct": correct,
                    "pct": round(correct/total*100, 1),
                    "label": {
                        "high": "High confidence (>=80%)",
                        "medium": "Medium (60-80%)",
                        "low": "Low (<60%)"
                    }.get(bucket, bucket)
                }

        baselines = {}
        if base_row and base_row[0] > 0:
            total = base_row[0]
            baselines = {
                "total": total,
                "kronos_pct":      round((base_row[1] or 0) / total * 100, 1),
                "momentum_pct":    round((base_row[2] or 0) / total * 100, 1) if base_row[2] is not None else None,
                "carry_pct":       round((base_row[3] or 0) / total * 100, 1) if base_row[3] is not None else None,
                "random_walk_pct": round((base_row[4] or 0) / total * 100, 1) if base_row[4] is not None else None,
            }

        band_stats = {}
        if band_row and band_row[0] > 0:
            band_stats = {
                "total": band_row[0],
                "hits": band_row[1] or 0,
                "pct": round((band_row[1] or 0) / band_row[0] * 100, 1)
            }

        return {
            "by_coin": stats,
            "by_confidence": conf_stats,
            "baselines": baselines,
            "band_stats": band_stats,
        }
    except Exception:
        return {"by_coin": {}, "by_confidence": {}, "baselines": {}, "band_stats": {}}


# ── Worker ─────────────────────────────────────────────────────────────────────
def worker():
    while True:
        symbol = task_queue.get()
        if symbol is None:
            break
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


# ── Technical Indicators ───────────────────────────────────────────────────────
def compute_indicators(df):
    close    = df["close"].values
    high     = df["high"].values
    low      = df["low"].values

    # RSI (Wilder's smoothing)
    delta    = np.diff(close)
    gain     = np.where(delta > 0, delta, 0.0)
    loss     = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs       = np.where(avg_loss == 0, 100.0, avg_gain / (avg_loss + 1e-10))
    rsi      = np.append(np.nan, 100 - (100 / (1 + rs)))

    # MACD
    ema12    = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26    = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd     = ema12 - ema26
    sig      = pd.Series(macd).ewm(span=9, adjust=False).mean().values

    # Bollinger Bands
    sma20    = pd.Series(close).rolling(20).mean().values
    std20    = pd.Series(close).rolling(20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_pct   = np.where(bb_upper - bb_lower == 0, 0.5,
                        (close - bb_lower) / (bb_upper - bb_lower + 1e-10))

    # Volume
    vol      = df["volume"].values
    vol_sma  = pd.Series(vol).rolling(20).mean().values
    vol_ratio_raw = vol[-1] / (vol_sma[-1] + 1e-10)
    vol_valid = 0.05 <= vol_ratio_raw <= 20.0

    # ADX (#8 - Market regime)
    try:
        prev_close = close[:-1]
        curr_high  = high[1:]
        curr_low   = low[1:]
        curr_close = close[1:]

        tr = np.maximum(curr_high - curr_low,
             np.maximum(np.abs(curr_high - prev_close),
                        np.abs(curr_low  - prev_close)))

        up_move   = curr_high - high[:-1]
        down_move = low[:-1]  - curr_low

        plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        period = 14
        atr     = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / (atr + 1e-10)
        minus_di= 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / (atr + 1e-10)
        dx      = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx     = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values

        adx_val    = round(float(adx[-1]), 1)
        plus_di_val = round(float(plus_di[-1]), 1)
        minus_di_val= round(float(minus_di[-1]), 1)

        if adx_val >= 25:
            regime = "Trending"
        elif adx_val >= 20:
            regime = "Weak Trend"
        else:
            regime = "Choppy"
    except Exception:
        adx_val, plus_di_val, minus_di_val, regime = 0.0, 0.0, 0.0, "Unknown"

    return {
        "rsi":        round(float(rsi[-1]) if not np.isnan(rsi[-1]) else 50, 2),
        "macd_hist":  round(float((macd - sig)[-1]), 4),
        "bb_pct":     round(float(np.clip(bb_pct[-1], 0, 1)), 4),
        "vol_ratio":  round(float(vol_ratio_raw), 4),
        "vol_valid":  vol_valid,
        "sma20":      round(float(sma20[-1]), 2),
        "bb_upper":   round(float(bb_upper[-1]), 2),
        "bb_lower":   round(float(bb_lower[-1]), 2),
        "adx":        adx_val,
        "plus_di":    plus_di_val,
        "minus_di":   minus_di_val,
        "regime":     regime,
    }


# ── External signals ───────────────────────────────────────────────────────────
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        r.raise_for_status()
        d = r.json()
        return {"value": int(d["data"][0]["value"]),
                "label": d["data"][0]["value_classification"]}
    except Exception:
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
            return {"rate": round(rate, 4), "avg": round(avg, 4), "label": label}
    except Exception as e:
        print(f"[Funding] {symbol} failed: {e}", flush=True)
    return {"rate": None, "avg": None, "label": "N/A"}

def fetch_etf_flows():
    try:
        r = requests.get("https://farside.co.uk/bitcoin-etf-flow-all-data-table/",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if not r.ok:
            return {"total": None, "label": "Unavailable"}
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', r.text, re.DOTALL)
        if rows:
            numbers = re.findall(r'-?\d+(?:\.\d+)?',
                                 re.sub(r'<[^>]+>', ' ', rows[-2] if len(rows) > 1 else rows[-1]))
            if numbers:
                total = float(numbers[-1])
                if not (-10000 <= total <= 10000):
                    return {"total": None, "label": "Unavailable"}
                label = "Strong Inflows" if total > 300 else \
                        "Inflows" if total > 0 else \
                        "Outflows" if total > -300 else "Strong Outflows"
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
    except Exception:
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

            # #2: Drop incomplete last candle
            now = datetime.now(timezone.utc)
            last_candle_time = df["timestamps"].iloc[-1]
            candle_age = (now - last_candle_time).total_seconds()
            if candle_age < 59 * 60:  # less than 59 minutes old = still forming
                df = df.iloc[:-1].reset_index(drop=True)
                print(f"[Data] {symbol} dropped incomplete last candle ({candle_age/60:.1f}min old)", flush=True)

            print(f"[Data] {symbol} {interval} OK ({len(df)} candles)", flush=True)
            return df
        except Exception as e:
            print(f"[Data] {url} failed: {e}", flush=True)
    raise RuntimeError(f"All Binance endpoints failed for {symbol} {interval}")


# ── Find safe lookback ─────────────────────────────────────────────────────────
def find_safe_lookback(df, symbol):
    candidates = [384, 370, 360, 350, 340, 330, 320, 300, 280, 256]
    for lookback in candidates:
        if lookback > len(df):
            continue
        test_df = df.tail(lookback).reset_index(drop=True)
        x_df        = test_df[["open","high","low","close","volume"]].copy()
        x_timestamp = test_df["timestamps"].copy().reset_index(drop=True)
        last_time   = test_df["timestamps"].iloc[-1]
        future_times = pd.date_range(
            start=last_time + pd.Timedelta(hours=1),
            periods=2, freq="1h", tz="UTC"
        )
        y_timestamp = pd.Series(future_times).reset_index(drop=True)
        try:
            with torch.inference_mode():
                predictor.predict(
                    df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                    pred_len=2, T=0.7, top_p=0.9, sample_count=1
                )
            print(f"[Kronos] {symbol} safe lookback={lookback} (dry-run passed)", flush=True)
            return lookback
        except RuntimeError as e:
            if "size of tensor" in str(e) or "must match" in str(e):
                print(f"[Kronos] {symbol} lookback={lookback} failed dry-run, trying smaller...", flush=True)
                continue
            raise
    raise RuntimeError(f"{symbol}: no working lookback found in {candidates}")


# ── Kronos Monte Carlo ─────────────────────────────────────────────────────────
def kronos_predict(df, symbol="UNK", pred_len=24):
    safe_lookback = find_safe_lookback(df, symbol)
    work_df = df.tail(safe_lookback).reset_index(drop=True)

    last_time    = work_df["timestamps"].iloc[-1]
    x_df         = work_df[["open","high","low","close","volume"]].copy()
    x_timestamp  = work_df["timestamps"].copy().reset_index(drop=True)
    future_times = pd.date_range(
        start=last_time + pd.Timedelta(hours=1),
        periods=pred_len, freq="1h", tz="UTC"
    )
    y_timestamp = pd.Series(future_times).reset_index(drop=True)

    # #7: One greedy path (T=0.1, near-deterministic) for mean forecast
    greedy_close = None
    try:
        with torch.inference_mode():
            p_greedy = predictor.predict(
                df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                pred_len=pred_len, T=0.1, top_p=1.0, sample_count=1
            )
        greedy_close = p_greedy["close"].values
        print(f"[Kronos] {symbol} greedy path done.", flush=True)
    except Exception as e:
        print(f"[Kronos] {symbol} greedy path failed ({e}), using MC mean.", flush=True)

    # N=100 MC paths at T=0.7 for uncertainty (P10/P90)
    all_closes = []
    run_times  = []
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
                    print(f"[Kronos] {symbol} tensor error at MC {i+1}, lookback={safe_lookback}: {e}", flush=True)
                    raise
                raise
            elapsed = time.time() - t_start
            run_times.append(elapsed)
            all_closes.append(p["close"].values)

            avg_secs  = sum(run_times) / len(run_times)
            remaining = (MONTE_CARLO_N - i - 1) * avg_secs
            progress[symbol] = {
                "current": i + 1, "total": MONTE_CARLO_N,
                "secs_per_run": round(avg_secs, 2),
                "remaining_secs": round(remaining, 0),
                "pct": round((i + 1) / MONTE_CARLO_N * 100, 1),
            }
            if i % 10 == 0:
                print(f"[Kronos] {symbol} MC {i+1}/{MONTE_CARLO_N} "
                      f"({avg_secs:.1f}s/run, ~{remaining/60:.1f}min left)", flush=True)

    closes = np.array(all_closes)
    progress.pop(symbol, None)

    # Use greedy path as mean if available, else MC mean
    mean_close = greedy_close if greedy_close is not None else closes.mean(axis=0)

    return {
        "mean":         mean_close,
        "upper":        np.percentile(closes, 90, axis=0),
        "lower":        np.percentile(closes, 10, axis=0),
        "std":          closes.std(axis=0),
        "closes":       closes,
        "future_times": future_times,
        "lookback_used": safe_lookback,
    }


# ── Circuit Breaker ────────────────────────────────────────────────────────────
def circuit_breaker_check(symbol):
    """
    Returns (True, "OK") if safe to trade.
    Returns (False, reason) if circuit is broken.
    Requires minimum 20 overall / 10 per-coin predictions before activating.
    Also checks last 7 predictions for sudden cold streaks.
    """
    try:
        stats     = get_accuracy_stats()
        baselines = stats.get("baselines", {})
        by_coin   = stats.get("by_coin", {})

        # Overall circuit: if Kronos < 50% accuracy over 20+ predictions
        total = baselines.get("total", 0)
        if total >= 20:
            kronos_pct = baselines.get("kronos_pct", 50)
            if kronos_pct < 50:
                return False, f"Circuit broken: overall accuracy {kronos_pct}% over {total} predictions"

        # Per-coin circuit: if this coin < 40% over 10+ predictions
        coin_data = by_coin.get(symbol, {})
        if coin_data.get("total", 0) >= 10:
            if coin_data.get("pct", 50) < 40:
                return False, f"Circuit broken for {symbol}: accuracy {coin_data['pct']}% over {coin_data['total']} predictions"

        # Recency check: last 7 predictions — catches sudden cold streaks fast
        with sqlite3.connect(DB_FILE) as conn:
            recent = conn.execute("""
                SELECT direction_correct FROM accuracy
                WHERE symbol=? AND direction_correct IS NOT NULL
                ORDER BY predicted_at DESC LIMIT 7
            """, (symbol,)).fetchall()

        if len(recent) >= 5:
            recent_pct = sum(r[0] for r in recent) / len(recent) * 100
            if recent_pct < 30:
                return False, f"Cold streak: {recent_pct:.0f}% over last {len(recent)} predictions for {symbol}"

    except Exception as e:
        print(f"[Circuit] Check failed: {e}", flush=True)

    return True, "OK"


# ── Position Sizing (advisory only — never a command) ─────────────────────────
def compute_position_size(upside_prob, confidence, regime, last_price, p10, p90, capital=10000):
    """
    Advisory position sizing using simplified Kelly criterion.
    This is DISPLAY ONLY — never a trade command.
    Capital default = 10000 (adjust in UI per your actual capital).
    """
    try:
        edge = abs(upside_prob - 50) / 50.0  # 0 to 1

        regime_mult = {"Trending": 1.0, "Weak Trend": 0.6, "Choppy": 0.0}.get(regime, 0.5)
        conf_mult   = max(0.1, confidence / 100.0)

        avg_win  = abs(p90 - last_price) / last_price if last_price > 0 else 0.05
        avg_loss = abs(last_price - p10) / last_price if last_price > 0 else 0.05
        avg_loss = max(avg_loss, 0.001)

        kelly = (edge * avg_win - (1 - edge) * avg_loss) / avg_win
        kelly = max(0.0, min(kelly, 0.25))  # cap at quarter-Kelly

        size     = kelly * regime_mult * conf_mult
        size     = min(size, 0.10)  # hard cap: never more than 10% of capital

        direction = "LONG" if upside_prob >= 50 else "SHORT"
        stop      = p10 if direction == "LONG" else p90
        risk_dist = abs(last_price - stop)
        target    = last_price + (1.5 * risk_dist) if direction == "LONG" else last_price - (1.5 * risk_dist)
        risk_pct  = size * (risk_dist / last_price) if last_price > 0 else 0

        return {
            "direction":       direction,
            "size_pct":        round(size * 100, 2),
            "stop_loss":       round(stop, 4),
            "take_profit":     round(target, 4),
            "risk_pct":        round(risk_pct * 100, 2),
            "max_position_usd": round(capital * size, 2),
            "capital_assumed": capital,
        }
    except Exception as e:
        print(f"[PositionSize] Failed: {e}", flush=True)
        return None


# ── Signal interpretation ──────────────────────────────────────────────────────
def interpret_signals(upside_prob, ind, fear_greed, funding, etf_flows,
                      onchain, btc_dominance, symbol):
    rsi    = ind["rsi"]
    macd_h = ind["macd_hist"]
    bb     = ind["bb_pct"]
    fg     = fear_greed.get("value")
    fund   = funding.get("rate")
    regime = ind.get("regime", "Unknown")
    adx    = ind.get("adx", 0)
    bullish = upside_prob >= 50

    confirmations, warnings = [], []

    if rsi < 30:   confirmations.append(f"RSI oversold ({rsi:.1f}) - potential bounce")
    elif rsi > 70: warnings.append(f"RSI overbought ({rsi:.1f}) - stretched")

    if macd_h > 0 and bullish:       confirmations.append("MACD bullish - confirms Kronos")
    elif macd_h < 0 and not bullish: confirmations.append("MACD bearish - confirms Kronos")
    elif macd_h > 0 and not bullish: warnings.append("MACD bullish but Kronos bearish - mixed")
    elif macd_h < 0 and bullish:     warnings.append("MACD bearish but Kronos bullish - mixed")

    if bb < 0.2:   confirmations.append("Price near lower BB - oversold zone")
    elif bb > 0.8: warnings.append("Price near upper BB - overbought zone")

    if fg is not None:
        if fg <= 20:   confirmations.append(f"Extreme Fear ({fg}) - historically strong buy zone")
        elif fg >= 80: warnings.append(f"Extreme Greed ({fg}) - historically risky zone")

    if fund is not None:
        if fund > 0.05:    warnings.append(f"High funding ({fund:.3f}%) - longs overcrowded")
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

    # #8: Hard regime veto — choppy + neutral funding = NO_TRADE regardless of Kronos
    choppy_veto = (adx < 20) and (fund is None or abs(fund) < 0.01)
    if choppy_veto:
        warnings.append(f"REGIME VETO: Market choppy (ADX={adx:.0f}) + neutral funding - trade signal forced to NO_TRADE")

    n_c, n_w = len(confirmations), len(warnings)

    # Circuit breaker check
    circuit_ok, circuit_msg = circuit_breaker_check(symbol)

    # Trade signal — hard veto overrides everything
    if not circuit_ok:
        trade_signal = "CIRCUIT_BROKEN"
    elif choppy_veto:
        trade_signal = "NO_TRADE"
    elif upside_prob > 60 and n_c >= 2 and n_w == 0:
        trade_signal = "LONG"
    elif upside_prob < 40 and n_c >= 2 and n_w == 0:
        trade_signal = "SHORT"
    else:
        trade_signal = "NO_TRADE"

    context = "Strong confirmation" if n_c >= 3 and n_w == 0 else \
              "Mostly confirmed" if n_c > n_w else \
              "Caution - mixed signals" if n_w > n_c else "Neutral context"

    return {
        "confirmations": confirmations, "warnings": warnings,
        "context": context, "n_confirm": n_c, "n_warn": n_w,
        "trade_signal": trade_signal, "circuit_msg": circuit_msg,
        "regime": regime, "adx": adx,
    }


# ── Core prediction ────────────────────────────────────────────────────────────
def run_prediction(symbol):
    print(f"[Kronos] {symbol} starting on {DEVICE_TYPE.upper()}...", flush=True)

    df         = fetch_candles(symbol, "1h", LOOKBACK)
    last_price = float(df["close"].iloc[-1])
    last_time  = df["timestamps"].iloc[-1]

    # #5: Compute baseline directions before running model
    # Momentum: last 24h trend continues
    try:
        price_24h_ago  = float(df["close"].iloc[-25]) if len(df) >= 25 else float(df["close"].iloc[0])
        mom_direction  = 1 if last_price > price_24h_ago else 0
    except Exception:
        mom_direction = None

    # Carry: funding rate direction (negative funding = bullish squeeze)
    # Will be computed after fetch_signals

    # Fetch external signals in parallel
    sig = {}
    def fetch_signals():
        try:
            sig["indicators"]    = compute_indicators(df)
        except Exception as e:
            print(f"[Indicators] Failed: {e}", flush=True)
            sig["indicators"] = {"rsi": 50, "macd_hist": 0, "bb_pct": 0.5,
                                  "vol_ratio": 1.0, "vol_valid": True,
                                  "sma20": last_price, "bb_upper": last_price,
                                  "bb_lower": last_price, "adx": 0,
                                  "plus_di": 0, "minus_di": 0, "regime": "Unknown"}
        sig["fear_greed"]    = fetch_fear_greed() if symbol == "BTC" else {"value": None, "label": "N/A"}
        sig["onchain"]       = fetch_onchain() if symbol == "BTC" else {}
        sig["funding"]       = fetch_funding_rate(symbol)
        sig["etf_flows"]     = fetch_etf_flows() if symbol == "BTC" else {"total": None, "label": "N/A"}
        sig["btc_dominance"] = fetch_btc_dominance() if symbol != "BTC" else None

    st = threading.Thread(target=fetch_signals)
    st.start()

    pred = kronos_predict(df, symbol=symbol, pred_len=PRED_LEN)

    st.join()

    # Carry direction from funding rate
    carry_direction = None
    fund_rate = sig["funding"].get("rate")
    if fund_rate is not None:
        # Negative funding = shorts paying longs = bullish squeeze expected
        # Positive funding = longs paying shorts = bearish correction expected
        if fund_rate < -0.01:
            carry_direction = 1  # bullish
        elif fund_rate > 0.05:
            carry_direction = 0  # bearish
        else:
            carry_direction = None  # neutral, no carry signal

    closes     = pred["closes"]
    mean_close = pred["mean"]
    upper      = pred["upper"]
    lower      = pred["lower"]
    std        = pred["std"]

    final_prices    = closes[:, -1]
    raw_upside_prob = float((final_prices > last_price).mean()) * 100
    upside_prob     = float(max(5.0, min(95.0, raw_upside_prob)))

    std_pct      = std / last_price * 100
    hist_vol_pct = float(df["close"].pct_change().dropna().std() * 100)
    hist_vol_24h = hist_vol_pct * float(np.sqrt(PRED_LEN))
    vol_amp_prob = float((std_pct > hist_vol_pct).mean()) * 100
    spread_pct   = (upper - lower) / last_price * 100
    avg_spread   = float(spread_pct.mean())

    if avg_spread < hist_vol_24h:
        confidence    = round(max(10.0, 50.0 - (hist_vol_24h - avg_spread) * 3), 1)
        hallucinating = True
    else:
        confidence    = round(max(0.0, 100.0 - avg_spread * 2), 1)
        hallucinating = False

    print(f"[Kronos] {symbol} spread={avg_spread:.2f}% hist_vol_24h={hist_vol_24h:.2f}% "
          f"hallucinating={hallucinating} conf={confidence}", flush=True)

    signal_context = interpret_signals(
        upside_prob, sig["indicators"], sig["fear_greed"],
        sig["funding"], sig["etf_flows"], sig["onchain"],
        sig["btc_dominance"], symbol
    )

    result = {
        "updated_at":         datetime.now(timezone.utc).isoformat(),
        "symbol":             f"{symbol}/USDT",
        "coin":               symbol,
        "last_price":         last_price,
        "last_time":          str(last_time),
        "pred_len":           PRED_LEN,
        "lookback":           LOOKBACK,
        "monte_carlo_n":      MONTE_CARLO_N,
        "model":              MODEL_NAME,
        "device":             DEVICE_TYPE,
        "upside_prob":        round(upside_prob, 1),
        "raw_upside_prob":    round(raw_upside_prob, 1),
        "hallucinating":      hallucinating,
        "confidence":         confidence,
        "vol_amp_prob":       round(vol_amp_prob, 1),
        "lookback_used":      pred.get("lookback_used", LOOKBACK),
        "momentum_direction": mom_direction,
        "carry_direction":    carry_direction,
        "signal_context":     signal_context,
        "indicators":         sig["indicators"],
        "fear_greed":         sig["fear_greed"],
        "funding":            sig["funding"],
        "etf_flows":          sig["etf_flows"],
        "onchain":            sig["onchain"],
        "btc_dominance":      sig["btc_dominance"],
        "accuracy":           get_accuracy_stats().get("by_coin", {}).get(symbol, {}),
        "position_size":      compute_position_size(
                                  upside_prob, confidence,
                                  signal_context.get("regime", "Unknown"),
                                  last_price, float(lower[-1]), float(upper[-1])
                              ),
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
    print(f"[Kronos] {symbol} DONE. Upside={upside_prob:.1f}% Conf={confidence:.1f}% "
          f"Signal={signal_context['trade_signal']} Regime={signal_context['regime']}", flush=True)
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

        gpu_working = False
        if DEVICE_TYPE != "cpu":
            try:
                model = model.to(DEVICE)
                test_tensor = torch.zeros(1, 10).to(DEVICE)
                _ = test_tensor + 1
                gpu_working = True
                print(f"[Kronos] Model on {DEVICE_TYPE.upper()} - verified!", flush=True)
            except Exception as e:
                print(f"[Kronos] GPU failed ({e}), falling back to CPU", flush=True)
                model = model.cpu()

        MONTE_CARLO_N = 100
        device_note = "GPU connected (CPU-bound inference)" if gpu_working else "CPU"
        print(f"[Kronos] {device_note} - N={MONTE_CARLO_N}", flush=True)

        predictor   = KronosPredictor(model, tokenizer, max_context=512)
        model_ready = True
        print(f"[Kronos] Ready! Device={DEVICE_TYPE.upper()} N={MONTE_CARLO_N}", flush=True)

        for i in range(NUM_WORKERS):
            threading.Thread(target=worker, daemon=True, name=f"Worker-{i+1}").start()

        print(f"[Kronos] Ready! Waiting for manual prediction requests.", flush=True)

        def accuracy_loop():
            check_accuracy()
            close_paper_trades()
            while True:
                time.sleep(3600)
                check_accuracy()
                close_paper_trades()
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
    safe_progress = dict(progress)
    coin_ages = {}
    for sym, data in cache.items():
        try:
            updated  = datetime.fromisoformat(data["updated_at"])
            age_mins = int((datetime.now(timezone.utc) - updated).total_seconds() // 60)
            coin_ages[sym] = str(age_mins) + 'm ago' if age_mins < 60 else str(age_mins // 60) + 'h ago'
        except Exception:
            pass
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
        "requests_24h":  get_request_stats(),
        "coin_ages":     coin_ages,
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

def get_request_stats():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            rows = conn.execute("""
                SELECT symbol, COUNT(*) as cnt FROM request_log
                WHERE requested_at > datetime('now', '-24 hours')
                GROUP BY symbol ORDER BY cnt DESC
            """).fetchall()
            total = conn.execute("""
                SELECT COUNT(*) FROM request_log
                WHERE requested_at > datetime('now', '-24 hours')
            """).fetchone()[0]
        return {"total_24h": total, "by_coin": {s: c for s, c in rows}}
    except Exception:
        return {"total_24h": 0, "by_coin": {}}

@app.route("/predict/<symbol>")
def predict(symbol):
    symbol = symbol.upper()
    if symbol not in COINS:
        return jsonify({"error": f"Unknown symbol {symbol}"}), 400
    if not model_ready:
        return jsonify({"error": "Model not loaded yet."}), 503
    with queue_lock:
        if running.get(symbol, False) or symbol in queued_coins:
            return jsonify({"status": "already_running_or_queued"})
        queued_coins.add(symbol)
        task_queue.put(symbol)
    try:
        with db_lock:
            with sqlite3.connect(DB_FILE) as conn:
                conn.execute("INSERT INTO request_log (symbol, requested_at) VALUES (?, ?)",
                             (symbol, datetime.now(timezone.utc).isoformat()))
                conn.commit()
    except Exception:
        pass
    return jsonify({"status": "queued"})

@app.route("/accuracy")
def accuracy_route():
    return jsonify(get_accuracy_stats())

@app.route("/paper_pnl")
def paper_pnl():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            open_trades = conn.execute("""
                SELECT symbol, direction, entry_price, stop_price, target_price,
                       size_pct, risk_pct, predicted_at
                FROM paper_trades WHERE closed=0
                ORDER BY predicted_at DESC
            """).fetchall()

            closed_trades = conn.execute("""
                SELECT symbol, direction, entry_price, exit_price,
                       exit_reason, pnl_pct, size_pct, closed_at
                FROM paper_trades WHERE closed=1
                ORDER BY closed_at DESC LIMIT 50
            """).fetchall()

            totals = conn.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
                       SUM(pnl_pct) as total_pnl,
                       AVG(pnl_pct) as avg_pnl
                FROM paper_trades WHERE closed=1
            """).fetchone()

        return jsonify({
            "open": [{"symbol": r[0], "direction": r[1], "entry": r[2],
                      "stop": r[3], "target": r[4], "size_pct": r[5],
                      "risk_pct": r[6], "predicted_at": r[7]} for r in open_trades],
            "closed": [{"symbol": r[0], "direction": r[1], "entry": r[2],
                        "exit": r[3], "reason": r[4], "pnl_pct": r[5],
                        "size_pct": r[6], "closed_at": r[7]} for r in closed_trades],
            "summary": {
                "total_trades": totals[0] or 0,
                "wins": totals[1] or 0,
                "total_pnl_pct": round(totals[2] or 0, 2),
                "avg_pnl_pct": round(totals[3] or 0, 2),
                "win_rate": round((totals[1] or 0) / (totals[0] or 1) * 100, 1),
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
