import os, time, requests, math
from datetime import datetime, timezone, timedelta, date
from zoneinfo import ZoneInfo
from fastapi import FastAPI, Query, Response

API = "https://finnhub.io/api/v1"
KEY = os.getenv("FINNHUB_API_KEY")  # set env var

app = FastAPI(title="Finnhub Sidecar")

CACHE = {}
def cache_get(k, ttl): 
    v = CACHE.get(k)
    return v["data"] if v and (time.time()-v["ts"]<ttl) else None
def cache_put(k, data): CACHE[k] = {"data": data, "ts": time.time()}

def pct(c, pc):
    try:
        if c is None or pc in (0, None): return None
        return float(c)/float(pc)-1.0
    except: return None

def phase_chi(now=None):
    tz = ZoneInfo("America/Chicago")
    dt = (now or datetime.now(timezone.utc)).astimezone(tz)
    if dt.weekday()>=5: return "closed"
    m = dt.hour*60+dt.minute
    if   3*60 <= m < 8*60+30: return "premarket"
    elif 8*60+30 <= m < 15*60: return "regular"
    elif 15*60 <= m < 19*60:   return "afterhours"
    else: return "closed"

def fh(path, params):
    params = dict(params or {})
    params["token"] = KEY
    r = requests.get(f"{API}/{path}", params=params, timeout=3.0)
    r.raise_for_status()
    return r.json()

# --- FUTURES SNAPSHOT (ETF proxies) ---
TAPE = {"es_proxy":"SPY","nq_proxy":"QQQ","ym_proxy":"DIA","rty_proxy":"IWM","vix":"CBOE:VIX"}

@app.get("/gpt/futures-snapshot")
def futures_snapshot(resp: Response):
    if not KEY:
        resp.status_code = 503; return {"code":"NO_KEY","message":"Finnhub key missing"}
    cached = cache_get("fut_snap", 180)
    if cached: return cached
    out = {}
    for k,sym in TAPE.items():
        try:
            q = fh("quote", {"symbol": sym})
            out[k] = {
                "symbol": sym,
                "last": q.get("c"),
                "percent": pct(q.get("c"), q.get("pc")),
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "stale": False
            }
        except Exception:
            out[k] = {"symbol": sym, "last": None, "percent": None, "time_utc": datetime.now(timezone.utc).isoformat(), "stale": True}
    out["market_phase"] = phase_chi()
    out["stale_seconds"] = 0
    cache_put("fut_snap", out)
    return out

# --- NEWS + SENTIMENT ---
@app.get("/gpt/news/sentiment")
def news_sentiment(symbol: str, window: str = "24h", limit: int = 12):
    if not KEY: return {"items": [], "sentiment": None}
    # window → date range (UTC)
    now = datetime.now(timezone.utc)
    delta = {"12h":12, "24h":24, "7d":168}.get(window, 24)
    start = (now - timedelta(hours=delta)).date().isoformat()
    end   = (now + timedelta(days=1)).date().isoformat()
    key = f"news:{symbol}:{window}:{limit}"
    cached = cache_get(key, 600)
    if cached: return cached
    items = []
    try:
        news = fh("company-news", {"symbol": symbol, "from": start, "to": end})[:200]
        for n in news:
            items.append({
                "source": n.get("source"),
                "title": n.get("headline"),
                "url": n.get("url"),
                "published": datetime.fromtimestamp(n.get("datetime", 0), tz=timezone.utc).isoformat(),
                "symbols": [symbol],
                "sectors": [],
                "summary": n.get("summary"),
                "sentiment": None,   # headline-only sentiment optional; keep None for now
                "buzz_window": window
            })
        items = sorted(items, key=lambda x: x["published"], reverse=True)[:limit]
    except Exception:
        items = []
    sent = None
    try:
        s = fh("news-sentiment", {"symbol": symbol})
        sc = s.get("companyNewsScore")
        bias = s.get("sectorAverageBullishPercent")
        buzz = s.get("buzz", {})
        sent = {
            "symbol_sentiment": sc,              # 0..1
            "news_count_24h": buzz.get("articlesInLastWeek"),
            "news_bias_24h": bias,               # 0..1
            "social_sentiment": None,
            "headline_risk": "elevated" if (bias is not None and bias < 0.45 and (buzz.get("articlesInLastWeek",0) > (buzz.get("weeklyAverage",0) or 1))) else "normal"
        }
    except Exception:
        sent = None
    out = {"items": items, "sentiment": sent}
    cache_put(key, out)
    return out

# --- MACRO + EARNINGS ---
def next_macro_minutes():
    try:
        today = date.today().isoformat()
        to = (date.today() + timedelta(days=7)).isoformat()
        cal = fh("calendar/economic", {"from": today, "to": to})
        wanted = ("Consumer Price Index", "FOMC", "Fed Interest Rate", "Nonfarm Payrolls")
        now = datetime.now(timezone.utc)
        mins = {}
        for ev in cal.get("economicCalendar", []):
            name = ev.get("event")
            if not name: continue
            if not any(w in name for w in wanted): continue
            dt = ev.get("time") or ev.get("datetime")
            if not dt: continue
            t = datetime.fromisoformat(dt.replace("Z","+00:00"))
            diff = (t - now).total_seconds()/60.0
            if diff >= -30:  # ignore stale prints
                key = "cpi" if "Price Index" in name else ("fomc" if "FOMC" in name or "Interest Rate" in name else ("nfp" if "Payroll" in name else None))
                if key: mins[key] = math.floor(diff)
        label = "none"
        within = False
        mm = min([v for v in mins.values() if v is not None], default=None)
        if mm is not None:
            within = mm <= 120
            label = "risk" if mm <= 60 else "watch"
        return {
            "next_cpi_minutes": mins.get("cpi"),
            "next_fomc_minutes": mins.get("fomc"),
            "next_nfp_minutes": mins.get("nfp"),
            "within_event_window": within,
            "label": label
        }
    except Exception:
        return {"within_event_window": False, "label": "none"}

def next_earnings(symbol: str):
    try:
        fr = date.today().isoformat()
        to = (date.today() + timedelta(days=60)).isoformat()
        cal = fh("calendar/earnings", {"from": fr, "to": to})
        rows = [r for r in cal.get("earningsCalendar", []) if r.get("symbol")==symbol]
        if not rows: return None
        row = sorted(rows, key=lambda x: x.get("date"))[0]
        dtv = datetime.fromisoformat((row.get("date")+"T12:00:00+00:00"))
        # pre/post:
        tslot = (row.get("hour") or "").lower()
        prepost = "pre" if "bmo" in tslot or "pre" in tslot else ("post" if "amc" in tslot or "post" in tslot else "none")
        dte = (dtv.date() - date.today()).days
        flag = "today" if dte==0 else ("near" if dte<=7 else "none")
        return {
            "next_earnings_at": dtv.isoformat(),
            "dte_to_earnings": dte,
            "pre_or_post": prepost,
            "earnings_flag": flag,
            "expected_move_pct": None
        }
    except Exception:
        return None

@app.get("/gpt/events/earnings")
def events_earnings(symbol: str):
    events = next_macro_minutes()
    earn = next_earnings(symbol)
    return {"events": events, "earnings": earn}

