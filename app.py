import os, sys, time, threading, traceback, json, queue, sqlite3, re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, send_from_directory, request
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

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME     = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
LOOKBACK       = 384
PRED_LEN       = 24
MONTE_CARLO_N  = max(1, min(int(os.environ.get("KRONOS_MONTE_CARLO_N", "100")), 1000))
REFRESH_SECS   = 3600
NUM_WORKERS    = 1
DB_FILE        = os.path.join(BASE_DIR, "kronos.db")

# This application is a local research tool.  It deliberately has no exchange
# credentials or order-submission code.  Paper mode must be explicitly enabled.
APP_MODE = os.environ.get("KRONOS_APP_MODE", "research").lower()
PAPER_TRADING_ENABLED = APP_MODE == "paper"
# Keep the execution accounting internally consistent.  Spot is the default
# because the app's candles come from Binance spot.  Set KRONOS_EXECUTION_VENUE
# to "futures" only when you deliberately want both candles and funding to use
# Binance USD-M futures.
EXECUTION_VENUE = os.environ.get("KRONOS_EXECUTION_VENUE", "spot").strip().lower()
if EXECUTION_VENUE not in {"spot", "futures"}:
    EXECUTION_VENUE = "spot"
TAKER_FEE_PCT = float(os.environ.get("KRONOS_TAKER_FEE_PCT", "0.10"))
SLIPPAGE_PCT = float(os.environ.get("KRONOS_SLIPPAGE_PCT", "0.05"))
ROUND_TRIP_COST_PCT = 2 * (TAKER_FEE_PCT + SLIPPAGE_PCT)
FUNDING_INTERVALS_PER_HORIZON = PRED_LEN / 8
MIN_VALIDATION_OUTCOMES = 200

COINS = {
    "ZEC": "ZECUSDT", "BTC": "BTCUSDT", "TAO": "TAOUSDT", "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
}

predictor    = None
cache        = {}
cache_lock   = threading.Lock()
state_lock   = threading.Lock()
model_ready  = False
model_error  = ""
running      = {}
running_since= {}
progress     = {}
task_queue   = queue.Queue()
db_lock      = threading.Lock()
queued_coins = set()
queue_lock   = threading.Lock()
stats_cache  = {"data": None, "updated_monotonic": 0.0}
stats_lock   = threading.Lock()
lookback_cache = {}
lookback_lock  = threading.Lock()


def _request_with_retry(url, *, params=None, headers=None, timeout=(3, 12), retries=2):
    """Small, bounded retry helper for read-only external data requests."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            if response.ok:
                return response
            last_error = RuntimeError(f"HTTP {response.status_code}")
        except requests.RequestException as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(0.75 * (attempt + 1))
    raise RuntimeError(f"Request failed for {url}: {last_error}")


def _invalidate_stats_cache():
    with stats_lock:
        stats_cache["data"] = None
        stats_cache["updated_monotonic"] = 0.0


def _json_safe(value):
    """Convert known NumPy/Pandas values without lossy JSON double-encoding."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


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
                target_timestamp     TEXT,
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
                flat_correct INTEGER,
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
        conn.execute("""
            CREATE TABLE IF NOT EXISTS real_trades (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol         TEXT NOT NULL,
                direction      TEXT NOT NULL,
                entry_price    REAL NOT NULL,
                entry_time     TEXT NOT NULL,
                amount_usd     REAL NOT NULL,
                quantity       REAL NOT NULL,
                app_upside_prob REAL,
                app_predicted_price REAL,
                app_signal     TEXT,
                exit_price     REAL,
                exit_time      TEXT,
                pnl_usd        REAL,
                pnl_pct        REAL,
                notes          TEXT,
                closed         INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_failures (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol    TEXT NOT NULL,
                failed_at TEXT NOT NULL,
                error     TEXT
            )
        """)
        conn.commit()

    # Migrations for existing databases
    migrations = [
        "ALTER TABLE accuracy ADD COLUMN target_timestamp TEXT",
        "ALTER TABLE accuracy ADD COLUMN entry_price REAL NOT NULL DEFAULT 0",
        "ALTER TABLE accuracy ADD COLUMN p10_at_prediction REAL",
        "ALTER TABLE accuracy ADD COLUMN p90_at_prediction REAL",
        "ALTER TABLE accuracy ADD COLUMN momentum_direction INTEGER",
        "ALTER TABLE accuracy ADD COLUMN carry_direction INTEGER",
        "ALTER TABLE accuracy ADD COLUMN inside_band INTEGER",
        "ALTER TABLE accuracy ADD COLUMN momentum_correct INTEGER",
        "ALTER TABLE accuracy ADD COLUMN carry_correct INTEGER",
        "ALTER TABLE accuracy ADD COLUMN flat_correct INTEGER",
        "ALTER TABLE accuracy ADD COLUMN target_at TEXT",
        "ALTER TABLE accuracy ADD COLUMN raw_upside_prob REAL",
        "ALTER TABLE accuracy ADD COLUMN candidate_signal TEXT",
        "ALTER TABLE accuracy ADD COLUMN actual_move_pct REAL",
        "ALTER TABLE accuracy ADD COLUMN gross_return_pct REAL",
        "ALTER TABLE accuracy ADD COLUMN net_return_pct REAL",
        "ALTER TABLE accuracy ADD COLUMN estimated_cost_pct REAL",
        "ALTER TABLE accuracy ADD COLUMN estimated_funding_cost_pct REAL",
        "ALTER TABLE paper_trades ADD COLUMN target_at TEXT",
        "ALTER TABLE paper_trades ADD COLUMN gross_pnl_pct REAL",
        "ALTER TABLE paper_trades ADD COLUMN net_pnl_pct REAL",
        "ALTER TABLE paper_trades ADD COLUMN estimated_cost_pct REAL",
        "ALTER TABLE paper_trades ADD COLUMN estimated_funding_cost_pct REAL",
    ]
    for sql in migrations:
        try:
            with sqlite3.connect(DB_FILE) as conn:
                conn.execute(sql)
                conn.commit()
        except Exception:
            pass  # Column already exists

    print(f"[DB] SQLite ready (WAL mode): {DB_FILE}", flush=True)


def _deprecated_save_prediction(symbol, result):
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

                # Exact target timestamp: 24h after prediction, rounded to hour
                pred_time = datetime.fromisoformat(result["updated_at"])
                target_ts = (pred_time + pd.Timedelta(hours=24)).replace(
                    minute=0, second=0, microsecond=0).isoformat()

                conn.execute("""
                    INSERT INTO accuracy
                    (symbol, predicted_at, target_timestamp, entry_price, predicted_price, upside_prob,
                     confidence, p10_at_prediction, p90_at_prediction,
                     momentum_direction, carry_direction)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, result["updated_at"], target_ts, entry, target,
                      result["upside_prob"], conf, p10, p90, mom_dir, carry_dir))
                conn.commit()

        with cache_lock:
            cache[symbol] = result
        print(f"[DB] {symbol} saved.", flush=True)
        log_paper_trade(symbol, result)
    except Exception as e:
        print(f"[DB] Save failed: {e}", flush=True)


def _deprecated_log_paper_trade(symbol, result):
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


def _deprecated_close_paper_trades():
    """Check ALL open trades every hour for stop/target hits.
    TIME_EXIT only after 24h. Stops/targets evaluated immediately."""
    try:
        with db_lock:
            with sqlite3.connect(DB_FILE) as conn:
                all_open = conn.execute("""
                    SELECT id, symbol, direction, entry_price, stop_price, target_price, predicted_at
                    FROM paper_trades WHERE closed=0
                """).fetchall()

        for trade in all_open:
            tid, sym, direction, entry, stop, target, predicted_at = trade
            if sym not in COINS:
                continue
            try:
                # Check 24h age for TIME_EXIT eligibility
                try:
                    pred_time = datetime.fromisoformat(predicted_at)
                    past_24h  = (datetime.now(timezone.utc) - pred_time).total_seconds() >= 86400
                except Exception:
                    past_24h = False

                # Fetch candles SINCE the trade opened — prevents pre-trade wicks triggering false exits
                try:
                    pred_ts = int(datetime.fromisoformat(predicted_at).timestamp() * 1000)
                except Exception:
                    pred_ts = None

                klines_params = {"symbol": COINS[sym], "interval": "1h", "limit": 1000}
                if pred_ts:
                    klines_params["startTime"] = pred_ts

                r = requests.get("https://api.binance.com/api/v3/klines",
                                 params=klines_params, timeout=(3, 10))
                if not r.ok:
                    continue
                candles  = r.json()
                highs    = [float(c[2]) for c in candles]
                lows     = [float(c[3]) for c in candles]
                current  = float(candles[-1][4])
                max_high = max(highs)
                min_low  = min(lows)

                # Determine exit — stops/targets immediate, time exit only after 24h
                exit_reason = exit_price = None
                if direction == "LONG":
                    if min_low <= stop:
                        exit_reason, exit_price = "STOP_HIT", stop
                    elif max_high >= target:
                        exit_reason, exit_price = "TARGET_HIT", target
                    elif past_24h:
                        exit_reason, exit_price = "TIME_EXIT", current
                    else:
                        continue  # still open, no trigger yet
                else:  # SHORT
                    if max_high >= stop:
                        exit_reason, exit_price = "STOP_HIT", stop
                    elif min_low <= target:
                        exit_reason, exit_price = "TARGET_HIT", target
                    elif past_24h:
                        exit_reason, exit_price = "TIME_EXIT", current
                    else:
                        continue  # still open, no trigger yet

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


def _deprecated_check_accuracy():
    """Compare 24h-old predictions against actual prices."""
    try:
        with db_lock:
            with sqlite3.connect(DB_FILE) as conn:
                rows = conn.execute("""
                    SELECT id, symbol, entry_price, upside_prob,
                           p10_at_prediction, p90_at_prediction,
                           momentum_direction, carry_direction, target_timestamp
                    FROM accuracy
                    WHERE actual_price IS NULL
                    AND datetime(predicted_at) < datetime('now', '-24 hours')
                """).fetchall()

        for row in rows:
            (row_id, symbol, entry_price, upside_prob,
             p10, p90, mom_dir, carry_dir) = row[:8]
            target_ts = row[8] if len(row) > 8 else None
            if symbol not in COINS:
                continue
            try:
                actual = None
                # Use exact target candle close if timestamp stored
                if target_ts:
                    try:
                        ts_dt = datetime.fromisoformat(target_ts)
                        ts_ms = int(ts_dt.timestamp() * 1000)
                        rk = requests.get("https://api.binance.com/api/v3/klines",
                                          params={"symbol": COINS[symbol], "interval": "1h",
                                                  "startTime": ts_ms, "limit": 1},
                                          timeout=(3, 10))
                        if rk.ok and rk.json():
                            actual = float(rk.json()[0][4])
                            print(f"[Accuracy] {symbol} using exact candle close at {target_ts}", flush=True)
                    except Exception as e:
                        print(f"[Accuracy] {symbol} candle fetch failed: {e}, falling back to ticker", flush=True)

                # Never fall back to a live ticker for a historical label.
                # Leave the row pending until the exact closed candle is available.

                if actual is not None:

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
                    flat_correct = 1 if abs(actual - entry_price) / (entry_price + 1e-10) < 0.01 else 0

                    with db_lock:
                        with sqlite3.connect(DB_FILE) as conn:
                            conn.execute("""
                                UPDATE accuracy
                                SET actual_price=?, direction_correct=?, inside_band=?,
                                    momentum_correct=?, carry_correct=?, flat_correct=?, checked_at=?
                                WHERE id=?
                            """, (actual, correct, band_hit, mom_correct, carry_correct,
                                  flat_correct, datetime.now(timezone.utc).isoformat(), row_id))
                            conn.commit()

                    print(f"[Accuracy] {symbol} entry={entry_price:.2f} actual={actual:.2f} "
                          f"correct={bool(correct)} band={band_hit}", flush=True)
            except Exception as e:
                print(f"[Accuracy] {symbol} check failed: {e}", flush=True)
    except Exception as e:
        print(f"[Accuracy] Check failed: {e}", flush=True)


def _deprecated_get_accuracy_stats():
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
                    SUM(flat_correct) as flat_correct
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
                "flat_pct": round((base_row[4] or 0) / total * 100, 1) if base_row[4] is not None else None,
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
        with state_lock:
            running[symbol] = True
            running_since[symbol] = time.time()
        try:
            run_prediction(symbol)
        except Exception as e:
            print(f"[Kronos] {symbol} failed: {e}", flush=True)
            traceback.print_exc()
            # Log the failure to dedicated table — prevents survivorship bias in accuracy stats
            try:
                with db_lock:
                    with sqlite3.connect(DB_FILE) as conn:
                        conn.execute("""
                            INSERT INTO prediction_failures (symbol, failed_at, error)
                            VALUES (?, ?, ?)
                        """, (symbol, datetime.now(timezone.utc).isoformat(), str(e)[:500]))
                        conn.commit()
            except Exception as db_e:
                print(f"[DB] Failed to log prediction failure: {db_e}", flush=True)
        finally:
            with state_lock:
                running[symbol] = False
                running_since.pop(symbol, None)
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
        r = _request_with_retry("https://api.alternative.me/fng/?limit=1")
        d = r.json()
        return {"value": int(d["data"][0]["value"]),
                "label": d["data"][0]["value_classification"]}
    except Exception:
        return {"value": None, "label": "N/A"}

def fetch_funding_rate(symbol):
    try:
        r = _request_with_retry("https://fapi.binance.com/fapi/v1/fundingRate",
                                params={"symbol": COINS[symbol], "limit": 3})
        if r.json():
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
        r = _request_with_retry("https://farside.co.uk/bitcoin-etf-flow-all-data-table/",
                                headers={"User-Agent": "Mozilla/5.0"}, timeout=(3, 15))
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
        r = _request_with_retry("https://mempool.space/api/mempool")
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
        r = _request_with_retry("https://api.coingecko.com/api/v3/global")
        return round(r.json()["data"]["market_cap_percentage"]["btc"], 2)
    except Exception:
        pass
    return None


# ── Data fetcher ───────────────────────────────────────────────────────────────
def fetch_candles(symbol, interval="1h", limit=384):
    binance_symbol = COINS[symbol]
    params = {"symbol": binance_symbol, "interval": interval, "limit": limit}
    if EXECUTION_VENUE == "futures":
        urls = ["https://fapi.binance.com/fapi/v1/klines"]
    else:
        urls = ["https://api1.binance.com/api/v3/klines",
                "https://api2.binance.com/api/v3/klines",
                "https://api3.binance.com/api/v3/klines",
                "https://api.binance.com/api/v3/klines"]
    for url in urls:
        try:
            r = _request_with_retry(url, params=params, timeout=(3, 15), retries=1)
            raw = r.json()
            df  = pd.DataFrame(raw, columns=[
                "open_time","open","high","low","close","volume",
                "close_time","qv","trades","tbb","tbq","ignore"
            ])
            df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            for c in ["open","high","low","close","volume"]:
                df[c] = df[c].astype(float)
            df = df[["timestamps","open","high","low","close","volume"]].reset_index(drop=True)

            # Exclude an open candle even during the final minute of its hour.
            now = datetime.now(timezone.utc)
            last_close_time = pd.to_datetime(int(raw[-1][6]), unit="ms", utc=True)
            if last_close_time.to_pydatetime() > now:
                df = df.iloc[:-1].reset_index(drop=True)
                print(f"[Data] {symbol} dropped incomplete last candle", flush=True)

            print(f"[Data] {symbol} {interval} OK ({len(df)} candles)", flush=True)
            return df
        except Exception as e:
            print(f"[Data] {url} failed: {e}", flush=True)
    raise RuntimeError(f"All Binance endpoints failed for {symbol} {interval}")


# ── Find safe lookback ─────────────────────────────────────────────────────────
def find_safe_lookback(df, symbol):
    with lookback_lock:
        cached = lookback_cache.get((symbol, DEVICE_TYPE))
    if cached is not None and cached <= len(df):
        return cached
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
            with lookback_lock:
                lookback_cache[(symbol, DEVICE_TYPE)] = lookback
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
            with state_lock:
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
    with state_lock:
        progress.pop(symbol, None)

    # The displayed central forecast must match the uncertainty bands: both
    # are derived from the same Monte Carlo population.  A separate greedy
    # path can be useful for diagnostics, but is not a statistical mean.
    mean_close = closes.mean(axis=0)

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
        with db_lock:
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
def _deprecated_compute_position_size(upside_prob, confidence, regime, last_price, p10, p90, capital=10000):
    """
    Position sizing DISABLED — model edge not yet validated.
    Returns advisory warning only.
    """
    direction = "LONG" if upside_prob >= 50 else "SHORT"
    stop      = p10 if direction == "LONG" else p90
    target    = p90 if direction == "LONG" else p10

    return {
        "direction":        direction,
        "size_pct":         0,
        "stop_loss":        round(float(stop), 4) if stop else 0,
        "take_profit":      round(float(target), 4) if target else 0,
        "risk_pct":         0,
        "max_position_usd": 0,
        "capital_assumed":  capital,
        "disabled":         True,
        "reason":           "NO POSITION SIZE — model edge not validated. Do not risk real capital.",
    }


# ── Signal interpretation ──────────────────────────────────────────────────────
def _deprecated_interpret_signals(upside_prob, ind, fear_greed, funding, etf_flows,
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

    # Hard regime veto — choppy + near-neutral funding = NO_TRADE regardless of Kronos
    choppy_veto = (adx < 20) and (fund is None or abs(fund) < 0.05)
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
    upside_prob     = raw_upside_prob  # sampled paths are not calibrated probabilities

    # The forecast band is descriptive only.  It is not a confidence score or
    # calibrated probability until the stored outcomes demonstrate calibration.
    forecast_band_width_pct = float((upper[-1] - lower[-1]) / last_price * 100)

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
        "probability_status": "uncalibrated_research_only",
        "forecast_band_width_pct": round(forecast_band_width_pct, 3),
        "execution_venue":    EXECUTION_VENUE,
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
        "position_size":      None,
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

    result = _json_safe(result)
    save_prediction(symbol, result)
    print(f"[Kronos] {symbol} DONE. SampledUpside={upside_prob:.1f}% "
          f"Candidate={signal_context['candidate_signal']} Regime={signal_context['regime']}", flush=True)
    return result


# ── Research-grade evaluation ─────────────────────────────────────────────────
# The active persistence and evaluation functions below are the only runtime
# implementations.  Older pre-validation routines are deliberately named
# ``_deprecated_*`` above so they cannot silently replace this research path.


def _as_utc(value):
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def fetch_exact_hourly_close(symbol, target_at):
    """Return only a fully closed, exact target-hour candle; never a live price."""
    target = _as_utc(target_at)
    start_ms = int(target.timestamp() * 1000)
    endpoint = "https://fapi.binance.com/fapi/v1/klines" if EXECUTION_VENUE == "futures" else "https://api.binance.com/api/v3/klines"
    r = _request_with_retry(endpoint, params={
        "symbol": COINS[symbol], "interval": "1h", "startTime": start_ms, "limit": 1
    })
    candles = r.json()
    if not candles or int(candles[0][0]) != start_ms:
        return None
    close_time = pd.to_datetime(int(candles[0][6]), unit="ms", utc=True)
    if close_time >= pd.Timestamp.now(tz="UTC"):
        return None
    return float(candles[0][4])


def save_prediction(symbol, result):
    """Persist every forecast with its actual model target, before outcomes exist."""
    try:
        target_at = result["forecast"]["timestamps"][-1]
        candidate = result.get("signal_context", {}).get("candidate_signal", "NO_TRADE")
        rate = result.get("funding", {}).get("rate")
        funding_cost = 0.0
        if EXECUTION_VENUE == "futures" and rate is not None and candidate in ("LONG", "SHORT"):
            # Positive funding is paid by longs; this is an estimate recorded at entry.
            funding_cost = float(rate) * FUNDING_INTERVALS_PER_HORIZON * (1 if candidate == "LONG" else -1)
        estimated_cost = ROUND_TRIP_COST_PCT + funding_cost
        with db_lock:
            with sqlite3.connect(DB_FILE) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO predictions (symbol, data, updated_at, error)
                    VALUES (?, ?, ?, NULL)
                """, (symbol, json.dumps(result), result["updated_at"]))
                fc = result["forecast"]
                conn.execute("""
                    INSERT INTO accuracy
                    (symbol, predicted_at, target_timestamp, target_at, entry_price,
                     predicted_price, upside_prob, raw_upside_prob,
                     p10_at_prediction, p90_at_prediction, momentum_direction,
                     carry_direction, candidate_signal, estimated_cost_pct, estimated_funding_cost_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, result["updated_at"], target_at, target_at, result["last_price"],
                      fc["mean_close"][-1], result["upside_prob"], result.get("raw_upside_prob"),
                      fc["lower"][-1], fc["upper"][-1],
                      result.get("momentum_direction"), result.get("carry_direction"),
                      candidate, estimated_cost, funding_cost))
        with cache_lock:
            cache[symbol] = result
        _invalidate_stats_cache()
        log_paper_trade(symbol, result)
    except Exception as e:
        print(f"[DB] Save failed: {e}", flush=True)


def check_accuracy():
    """Label outcomes with the exact closed candle specified at forecast time."""
    try:
        with db_lock:
            with sqlite3.connect(DB_FILE) as conn:
                rows = conn.execute("""
                    SELECT id, symbol, entry_price, upside_prob, p10_at_prediction,
                           p90_at_prediction, momentum_direction, carry_direction,
                           target_at, candidate_signal, estimated_cost_pct
                    FROM accuracy
                    WHERE actual_price IS NULL AND target_at IS NOT NULL
                """).fetchall()
        for row in rows:
            (row_id, symbol, entry, upside, p10, p90, mom, carry, target_at,
             candidate, cost) = row
            if symbol not in COINS:
                continue
            actual = fetch_exact_hourly_close(symbol, target_at)
            if actual is None:
                continue
            move = (actual / entry - 1) * 100
            direction_correct = int((upside >= 50 and actual > entry) or (upside < 50 and actual <= entry))
            if candidate == "LONG":
                gross = move
            elif candidate == "SHORT":
                gross = -move
            else:
                gross = net = None
            if candidate in ("LONG", "SHORT"):
                net = gross - (cost if cost is not None else ROUND_TRIP_COST_PCT)
            band = int(p10 <= actual <= p90) if p10 is not None and p90 is not None else None
            mom_correct = int((mom == 1 and actual > entry) or (mom == 0 and actual <= entry)) if mom is not None else None
            carry_correct = int((carry == 1 and actual > entry) or (carry == 0 and actual <= entry)) if carry is not None else None
            flat_correct = int(abs(move) < 1.0)
            with db_lock:
                with sqlite3.connect(DB_FILE) as conn:
                    conn.execute("""
                        UPDATE accuracy SET actual_price=?, direction_correct=?, inside_band=?,
                          momentum_correct=?, carry_correct=?, flat_correct=?, actual_move_pct=?,
                          gross_return_pct=?, net_return_pct=?, checked_at=? WHERE id=?
                    """, (actual, direction_correct, band, mom_correct, carry_correct, flat_correct,
                          round(move, 4), None if gross is None else round(gross, 4),
                          None if net is None else round(net, 4),
                          datetime.now(timezone.utc).isoformat(), row_id))
            _invalidate_stats_cache()
    except Exception as e:
        print(f"[Accuracy] Exact-horizon check failed: {e}", flush=True)


def _non_overlapping(rows):
    chosen, next_allowed = [], {}
    for row in sorted(rows, key=lambda r: (r[0], r[1])):
        symbol, predicted_at = row[0], _as_utc(row[1])
        if predicted_at >= next_allowed.get(symbol, pd.Timestamp("1970-01-01", tz="UTC")):
            chosen.append(row)
            next_allowed[symbol] = predicted_at + pd.Timedelta(hours=PRED_LEN)
    return chosen


def _block_bootstrap_lower_mean(values, block_size=5, draws=2000):
    """Conservative 5th-percentile mean using contiguous return blocks."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    blocks = [values[i:i + block_size] for i in range(0, len(values), block_size)]
    rng = np.random.default_rng(20260818)
    means = []
    for _ in range(draws):
        sampled = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])[:len(values)]
        means.append(sampled.mean())
    return float(np.percentile(means, 5))


def _get_accuracy_stats_uncached():
    """Return descriptive accuracy plus cost-aware strategy metrics, never a claim of edge."""
    try:
        with sqlite3.connect(DB_FILE) as conn:
            all_rows = conn.execute("""
                SELECT symbol, COUNT(*), SUM(direction_correct)
                FROM accuracy WHERE direction_correct IS NOT NULL AND target_at IS NOT NULL GROUP BY symbol
            """).fetchall()
            strategy_rows = conn.execute("""
                SELECT symbol, predicted_at, net_return_pct, gross_return_pct, actual_move_pct
                FROM accuracy WHERE net_return_pct IS NOT NULL
            """).fetchall()
            baseline = conn.execute("""
                SELECT COUNT(*), SUM(direction_correct), SUM(momentum_correct), SUM(carry_correct), SUM(flat_correct)
                FROM accuracy WHERE direction_correct IS NOT NULL AND target_at IS NOT NULL
            """).fetchone()
            band = conn.execute("SELECT COUNT(*), SUM(inside_band) FROM accuracy WHERE inside_band IS NOT NULL AND target_at IS NOT NULL").fetchone()
        by_coin = {s: {"total": n, "correct": c or 0, "pct": round((c or 0) / n * 100, 1)} for s, n, c in all_rows}
        non_overlapping = _non_overlapping(strategy_rows)
        returns = [r[2] for r in non_overlapping]
        gross_returns = [r[3] for r in non_overlapping]
        moves = [abs(r[4]) for r in non_overlapping]
        lower_mean = _block_bootstrap_lower_mean(returns)
        wins = sum(1 for x in returns if x > 0)
        profit_factor = (sum(x for x in returns if x > 0) / abs(sum(x for x in returns if x < 0))
                         if any(x < 0 for x in returns) else None)
        equity, peak, max_drawdown = 1.0, 1.0, 0.0
        for value in returns:
            equity *= 1 + value / 100
            peak = max(peak, equity)
            max_drawdown = min(max_drawdown, (equity / peak - 1) * 100)
        strategy = {
            "total": len(returns), "wins": wins,
            "win_rate": round(wins / len(returns) * 100, 1) if returns else None,
            "avg_net_return_pct": round(float(np.mean(returns)), 3) if returns else None,
            "median_net_return_pct": round(float(np.median(returns)), 3) if returns else None,
            "avg_gross_return_pct": round(float(np.mean(gross_returns)), 3) if gross_returns else None,
            "avg_abs_move_pct": round(float(np.mean(moves)), 3) if moves else None,
            "profit_factor": round(float(profit_factor), 2) if profit_factor is not None else None,
            "max_drawdown_pct": round(max_drawdown, 3),
            "lower_net_return_bound_pct": round(lower_mean, 3) if lower_mean is not None else None,
            "cost_pct": ROUND_TRIP_COST_PCT,
            "non_overlapping": True,
            "validated": len(returns) >= MIN_VALIDATION_OUTCOMES and lower_mean is not None and lower_mean > 0,
        }
        b = baseline or (0, 0, 0, 0, 0)
        baselines = {"total": b[0] or 0, "kronos_pct": round((b[1] or 0) / b[0] * 100, 1) if b[0] else None,
                     "momentum_pct": round((b[2] or 0) / b[0] * 100, 1) if b[0] else None,
                     "carry_pct": round((b[3] or 0) / b[0] * 100, 1) if b[0] else None,
                     "flat_pct": round((b[4] or 0) / b[0] * 100, 1) if b[0] else None}
        return {"by_coin": by_coin, "baselines": baselines,
                "band_stats": {"total": band[0] or 0, "hits": band[1] or 0,
                               "pct": round((band[1] or 0) / band[0] * 100, 1) if band[0] else None},
                "strategy": strategy}
    except Exception as e:
        print(f"[Stats] failed: {e}", flush=True)
        return {"by_coin": {}, "baselines": {}, "band_stats": {}, "strategy": {}}


def get_accuracy_stats():
    """Cache CPU-intensive evaluation metrics briefly; outcomes invalidate it."""
    now = time.monotonic()
    with stats_lock:
        if stats_cache["data"] is not None and now - stats_cache["updated_monotonic"] < 300:
            return stats_cache["data"]
    data = _get_accuracy_stats_uncached()
    with stats_lock:
        stats_cache["data"] = data
        stats_cache["updated_monotonic"] = now
    return data


def log_paper_trade(symbol, result):
    if not PAPER_TRADING_ENABLED:
        return
    candidate = result.get("signal_context", {}).get("candidate_signal")
    if candidate not in ("LONG", "SHORT"):
        return
    try:
        target_at = result["forecast"]["timestamps"][-1]
        rate = result.get("funding", {}).get("rate")
        funding_cost = (float(rate) * FUNDING_INTERVALS_PER_HORIZON * (1 if candidate == "LONG" else -1)
                        if EXECUTION_VENUE == "futures" and rate is not None else 0.0)
        with db_lock:
            with sqlite3.connect(DB_FILE) as conn:
                if conn.execute("SELECT 1 FROM paper_trades WHERE symbol=? AND closed=0", (symbol,)).fetchone():
                    return
                conn.execute("""
                    INSERT INTO paper_trades (symbol, predicted_at, target_at, direction, entry_price,
                      stop_price, target_price, size_pct, risk_pct, estimated_cost_pct, estimated_funding_cost_pct)
                    VALUES (?, ?, ?, ?, ?, 0, 0, 0, 0, ?, ?)
                """, (symbol, result["updated_at"], target_at, candidate, result["last_price"],
                      ROUND_TRIP_COST_PCT + funding_cost, funding_cost))
    except Exception as e:
        print(f"[Paper] log failed: {e}", flush=True)


def close_paper_trades():
    try:
        with db_lock:
            with sqlite3.connect(DB_FILE) as conn:
                rows = conn.execute("SELECT id, symbol, direction, entry_price, target_at, estimated_cost_pct FROM paper_trades WHERE closed=0 AND target_at IS NOT NULL").fetchall()
        for trade_id, symbol, direction, entry, target_at, cost in rows:
            exit_price = fetch_exact_hourly_close(symbol, target_at)
            if exit_price is None:
                continue
            gross = ((exit_price / entry - 1) * 100) if direction == "LONG" else ((1 - exit_price / entry) * 100)
            net = gross - (cost if cost is not None else ROUND_TRIP_COST_PCT)
            with db_lock:
                with sqlite3.connect(DB_FILE) as conn:
                    conn.execute("""
                      UPDATE paper_trades SET closed=1, exit_price=?, exit_reason='HORIZON_CLOSE',
                        gross_pnl_pct=?, net_pnl_pct=?, pnl_pct=?, closed_at=? WHERE id=?
                    """, (exit_price, round(gross, 4), round(net, 4), round(net, 4),
                          datetime.now(timezone.utc).isoformat(), trade_id))
            _invalidate_stats_cache()
    except Exception as e:
        print(f"[Paper] close failed: {e}", flush=True)


def compute_position_size(*_args, **_kwargs):
    return None


def interpret_signals(upside_prob, ind, fear_greed, funding, etf_flows, onchain, btc_dominance, symbol):
    # This preserves the existing heuristic as a research candidate only.
    candidate = "NO_TRADE"
    adx, fund = ind.get("adx", 0), funding.get("rate")
    if fund is not None and adx >= 20:
        if upside_prob > 60 and ind.get("macd_hist", 0) > 0 and ind.get("rsi", 50) < 70:
            candidate = "LONG"
        elif upside_prob < 40 and ind.get("macd_hist", 0) < 0 and ind.get("rsi", 50) > 30:
            candidate = "SHORT"
    return {"confirmations": [], "warnings": [], "context": "Research candidate; not a trade instruction",
            "n_confirm": 0, "n_warn": 0, "candidate_signal": candidate,
            "trade_signal": candidate if PAPER_TRADING_ENABLED else "RESEARCH_ONLY",
            "circuit_msg": "Signals remain research-only until independent net-return validation.",
            "regime": ind.get("regime", "Unknown"), "adx": adx}


# ── Model loader ───────────────────────────────────────────────────────────────
def load_model():
    global predictor, model_ready, model_error
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
    with cache_lock:
        safe_cache = dict(cache)
    with state_lock:
        safe_progress = dict(progress)
        safe_running = dict(running)
        safe_running_since = dict(running_since)
    coin_ages = {}
    for sym, data in safe_cache.items():
        try:
            updated  = datetime.fromisoformat(data["updated_at"])
            age_mins = int((datetime.now(timezone.utc) - updated).total_seconds() // 60)
            coin_ages[sym] = str(age_mins) + 'm ago' if age_mins < 60 else str(age_mins // 60) + 'h ago'
        except Exception:
            pass
    return jsonify({
        "app_mode":      APP_MODE,
        "paper_trading_enabled": PAPER_TRADING_ENABLED,
        "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
        "model_ready":   model_ready,
        "model_error":   model_error,
        "model":         MODEL_NAME,
        "device":        DEVICE_TYPE,
        "monte_carlo_n": MONTE_CARLO_N,
        "execution_venue": EXECUTION_VENUE,
        "coins":         list(COINS.keys()),
        "cached":        list(safe_cache.keys()),
        "running":       {k: v for k, v in safe_running.items() if v},
        "running_since": {k: v for k, v in safe_running_since.items() if safe_running.get(k)},
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
    with cache_lock:
        cached_result = cache.get(symbol)
    with state_lock:
        is_running = running.get(symbol, False)
    if cached_result is None:
        return jsonify({
            "error": f"No prediction yet for {symbol}.",
            "model_ready": model_ready,
            "is_running":  is_running
        }), 404
    result = dict(cached_result)
    result["is_running"] = is_running
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
                WHERE datetime(requested_at) > datetime('now', '-24 hours')
                GROUP BY symbol ORDER BY cnt DESC
            """).fetchall()
            total = conn.execute("""
                SELECT COUNT(*) FROM request_log
                WHERE datetime(requested_at) > datetime('now', '-24 hours')
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
        with state_lock:
            is_running = running.get(symbol, False)
        if is_running or symbol in queued_coins:
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

@app.route("/real_trades", methods=["GET"])
def get_real_trades():
    try:
        with sqlite3.connect(DB_FILE) as conn:
            open_trades = conn.execute("""
                SELECT id, symbol, direction, entry_price, entry_time,
                       amount_usd, quantity, app_upside_prob, app_predicted_price,
                       app_signal, notes
                FROM real_trades WHERE closed=0
                ORDER BY entry_time DESC
            """).fetchall()
            closed_trades = conn.execute("""
                SELECT id, symbol, direction, entry_price, entry_time,
                       exit_price, exit_time, amount_usd, pnl_usd, pnl_pct, notes
                FROM real_trades WHERE closed=1
                ORDER BY exit_time DESC LIMIT 50
            """).fetchall()
            summary = conn.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END) as wins,
                       SUM(pnl_usd) as total_pnl,
                       SUM(amount_usd) as total_invested
                FROM real_trades WHERE closed=1
            """).fetchone()
        return jsonify({
            "open": [{"id":r[0],"symbol":r[1],"direction":r[2],"entry_price":r[3],
                      "entry_time":r[4],"amount_usd":r[5],"quantity":r[6],
                      "app_upside_prob":r[7],"app_predicted_price":r[8],
                      "app_signal":r[9],"notes":r[10]} for r in open_trades],
            "closed": [{"id":r[0],"symbol":r[1],"direction":r[2],"entry_price":r[3],
                        "entry_time":r[4],"exit_price":r[5],"exit_time":r[6],
                        "amount_usd":r[7],"pnl_usd":r[8],"pnl_pct":r[9],"notes":r[10]} for r in closed_trades],
            "summary": {
                "total_trades": summary[0] or 0,
                "wins": summary[1] or 0,
                "total_pnl_usd": round(summary[2] or 0, 2),
                "total_invested": round(summary[3] or 0, 2),
                "win_rate": round((summary[1] or 0) / max(summary[0] or 1, 1) * 100, 1)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/real_trades/open", methods=["POST"])
def open_real_trade():
    try:
        data       = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON"}), 400
        symbol     = data["symbol"].upper()
        direction  = data["direction"].upper()
        entry_price= float(data["entry_price"])
        amount_usd = float(data["amount_usd"])
        if symbol not in COINS or direction not in ("LONG", "SHORT"):
            return jsonify({"error": "Invalid symbol or direction"}), 400
        if not np.isfinite(entry_price) or not np.isfinite(amount_usd) or entry_price <= 0 or amount_usd <= 0:
            return jsonify({"error": "Entry price and amount must be finite positive numbers"}), 400
        quantity   = amount_usd / entry_price
        entry_time = data.get("entry_time", datetime.now(timezone.utc).isoformat())
        notes      = data.get("notes", "")

        # Capture current app prediction if available
        app_upside_prob = None
        app_predicted_price = None
        app_signal = None
        with cache_lock:
            c = cache.get(symbol)
        if c:
            app_upside_prob = c.get("upside_prob")
            fc = c.get("forecast", {})
            if fc.get("mean_close"):
                app_predicted_price = fc["mean_close"][-1]
            app_signal = c.get("signal_context", {}).get("trade_signal")

        with db_lock:
            with sqlite3.connect(DB_FILE) as conn:
                conn.execute("""
                    INSERT INTO real_trades
                    (symbol, direction, entry_price, entry_time, amount_usd, quantity,
                     app_upside_prob, app_predicted_price, app_signal, notes)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (symbol, direction, entry_price, entry_time, amount_usd, quantity,
                      app_upside_prob, app_predicted_price, app_signal, notes))
                conn.commit()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/real_trades/close/<int:trade_id>", methods=["POST"])
def close_real_trade(trade_id):
    try:
        data       = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON"}), 400
        exit_price = float(data["exit_price"])
        if not np.isfinite(exit_price) or exit_price <= 0:
            return jsonify({"error": "Exit price must be a finite positive number"}), 400
        exit_time  = data.get("exit_time", datetime.now(timezone.utc).isoformat())

        with db_lock:
            with sqlite3.connect(DB_FILE) as conn:
                trade = conn.execute(
                    "SELECT direction, entry_price, amount_usd, quantity FROM real_trades WHERE id=?",
                    (trade_id,)).fetchone()
                if not trade:
                    return jsonify({"error": "Trade not found"}), 404

                direction, entry_price, amount_usd, quantity = trade
                if direction == "LONG":
                    pnl_usd = (exit_price - entry_price) * quantity
                else:
                    pnl_usd = (entry_price - exit_price) * quantity
                pnl_pct = pnl_usd / amount_usd * 100

                conn.execute("""
                    UPDATE real_trades
                    SET exit_price=?, exit_time=?, pnl_usd=?, pnl_pct=?, closed=1
                    WHERE id=?
                """, (exit_price, exit_time, round(pnl_usd, 2), round(pnl_pct, 3), trade_id))
                conn.commit()
        return jsonify({"status": "ok", "pnl_usd": round(pnl_usd, 2), "pnl_pct": round(pnl_pct, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/real_trades/delete/<int:trade_id>", methods=["POST"])
def delete_real_trade(trade_id):
    try:
        with db_lock:
            with sqlite3.connect(DB_FILE) as conn:
                conn.execute("DELETE FROM real_trades WHERE id=?", (trade_id,))
                conn.commit()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
