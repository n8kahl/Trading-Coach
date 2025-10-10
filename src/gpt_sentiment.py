"""YouTube sentiment endpoint for GPT usage.

Adds GET /gpt/sentiment which fetches the latest video from a channel handle
and derives a lightweight sentiment, tickers, and key levels summary.

Dependencies are optional at runtime; when unavailable the endpoint returns 503
instead of failing import-time. This keeps tests passing without extra wheels.
"""

from __future__ import annotations

import html
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

try:
    from dateutil import parser as dtp  # type: ignore
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    from youtube_transcript_api import (  # type: ignore
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
    )
    _DEPS_OK = True
except Exception:  # pragma: no cover - optional deps
    _DEPS_OK = False


router = APIRouter(prefix="/gpt", tags=["gpt"])


class KeyLevel(BaseModel):
    symbol: str
    level: float
    note: str | None = None


class SentimentResponse(BaseModel):
    channel: str
    video_id: str
    video_url: str
    published_at: datetime
    title: str
    summary: str
    sentiment_label: str
    sentiment_score: float
    tickers: List[str]
    key_levels: List[KeyLevel] = []
    risks: List[str] = []
    quotes: List[str] = []
    raw_excerpt: str | None = None


_CACHE: Dict[str, Any] = {"data": None, "ts": 0.0, "key": None}
_TTL = 60 * 15

YOUTUBE_USER_URL = "https://www.youtube.com/@{handle}"
YOUTUBE_RSS = "https://www.youtube.com/feeds/videos.xml?channel_id={cid}"
YOUTUBE_WATCH = "https://www.youtube.com/watch?v={vid}"

# Default headers to reduce upstream blocks
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


async def _resolve_channel_id(client: httpx.AsyncClient, handle: str) -> str:
    url = YOUTUBE_USER_URL.format(handle=handle.lstrip("@"))
    r = await client.get(url, headers=DEFAULT_HEADERS, timeout=20)
    r.raise_for_status()
    m = re.search(r"channel/(UC[\w-]{20,})", r.text)
    if not m:
        raise HTTPException(status_code=422, detail="Could not resolve channel_id from handle.")
    return m.group(1)


def _pick_sentiment(text: str) -> tuple[str, float]:
    if not _DEPS_OK:  # fallback neutral
        return "neutral", 0.0
    analyzer = SentimentIntensityAnalyzer()
    s = float(analyzer.polarity_scores(text).get("compound", 0.0))
    lbl = "neutral"
    if s >= 0.15:
        lbl = "bullish"
    elif s <= -0.15:
        lbl = "bearish"
    if re.search(r"\bbullish\b", text, re.I) and re.search(r"\bbearish\b", text, re.I):
        lbl = "mixed"
    return lbl, round(s, 3)


_TICKER_STOP = {
    "AND",
    "OR",
    "FOR",
    "WITH",
    "YOU",
    "THE",
    "ALL",
    "THIS",
    "FROM",
    "PRE",
    "POST",
    "OPEN",
    "HIGH",
    "LOW",
    "VWAP",
    "POC",
    "VAL",
    "VAH",
    "FOMC",
    "CPI",
    "NFP",
}


def _extract_tickers(text: str) -> List[str]:
    cands = set(re.findall(r"\b[A-Z]{2,5}\b", text))
    allow = {"SPY", "QQQ", "DIA", "IWM", "VIX"}
    out: List[str] = []
    for t in cands:
        if t in allow or (t.isupper() and t not in _TICKER_STOP and not t.isdigit()):
            out.append(t)
    return sorted(out)[:12]


def _extract_levels(text: str) -> List[KeyLevel]:
    pat = re.compile(r"\b([A-Z]{2,5})\b[^0-9$]{0,8}(\$?\d{2,4}(?:\.\d{1,2})?)")
    levels: List[KeyLevel] = []
    for sym, num in pat.findall(text):
        try:
            level = float(num.replace("$", ""))
            levels.append(KeyLevel(symbol=sym, level=level))
        except Exception:
            continue
    seen = set()
    uniq: List[KeyLevel] = []
    for kl in levels:
        k = (kl.symbol, round(kl.level, 2))
        if k not in seen:
            uniq.append(kl)
            seen.add(k)
    return uniq[:20]


async def _fetch_transcript(video_id: str) -> str:
    if not _DEPS_OK:
        return ""
    try:
        parts: List[Dict[str, Any]] | None = None
        # Preferred robust flow: list → find → fetch
        try:
            transcripts = YouTubeTranscriptApi.list_transcripts(video_id)  # type: ignore
            transcript = transcripts.find_transcript(["en", "en-US", "en-GB"])  # type: ignore
            parts = transcript.fetch()  # type: ignore
        except Exception:
            # Fallback to direct helper if available in the installed version
            try:
                parts = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])  # type: ignore
            except Exception:
                parts = None
        if not parts:
            return ""
        text = " ".join([str(p.get("text", "")) for p in parts])
        return text[:2000]
    except Exception:  # pragma: no cover - defensive
        return ""


async def _get_latest_video(client: httpx.AsyncClient, channel_id: str) -> Dict[str, Any] | None:
    rss = YOUTUBE_RSS.format(cid=channel_id)
    r = await client.get(rss, headers=DEFAULT_HEADERS, timeout=20)
    r.raise_for_status()
    entries = re.findall(r"<entry>(.*?)</entry>", r.text, re.S)
    best: Dict[str, Any] | None = None
    for e in entries:
        vid = re.search(r"videoId>([^<]+)</yt:", e)
        title = re.search(r"<title>([^<]+)</title>", e)
        when = re.search(r"<published>([^<]+)</published>", e)
        if not (vid and title and when):
            continue
        published = dtp.parse(when.group(1)) if _DEPS_OK else datetime.now(timezone.utc)
        item = {
            "video_id": vid.group(1),
            "title": html.unescape(title.group(1)),
            "published_at": published,
            "url": YOUTUBE_WATCH.format(vid=vid.group(1)),
        }
        if (best is None) or (item["published_at"] > best["published_at"]):
            best = item
    return best


@router.get("/sentiment", response_model=SentimentResponse, responses={204: {"description": "No Content"}})
async def gpt_sentiment(
    channel: str = Query("@BrettCorrigan"),
    window_hours: int = Query(36, ge=1, le=168),
    force: bool = Query(False),
):
    if not _DEPS_OK:
        raise HTTPException(status_code=503, detail="Sentiment dependencies not installed on server")

    now = time.time()
    key = f"{channel}|{window_hours}"
    if not force and _CACHE["data"] and _CACHE["key"] == key and (now - float(_CACHE["ts"])) < _TTL:
        return _CACHE["data"]

    async with httpx.AsyncClient(follow_redirects=True, headers=DEFAULT_HEADERS, timeout=20) as client:
        try:
            channel_id = await _resolve_channel_id(client, channel)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"YouTube handle fetch error: {exc}")
        latest = None
        try:
            latest = await _get_latest_video(client, channel_id)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"YouTube RSS error: {exc}")
        if not latest:
            raise HTTPException(status_code=204, detail="No recent videos")
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        if latest["published_at"].astimezone(timezone.utc) < cutoff:
            raise HTTPException(status_code=204, detail="No recent video in window")

        # Transcript is optional; failures yield empty string
        transcript = await _fetch_transcript(latest["video_id"])  # optional
        text_blob = f"{latest['title']}\n{transcript}"
        tickers = _extract_tickers(text_blob)
        key_levels = _extract_levels(text_blob)
        lbl, score = _pick_sentiment(text_blob)

        quotes: List[str] = []
        risks: List[str] = []
        for line in re.split(r"[\n\.]+", transcript)[:15]:
            if re.search(r"(watch|careful|risk|volatile|data|earnings|CPI|FOMC|NFP|jobs|rates)", line, re.I):
                risks.append(line.strip())
            if len(quotes) < 3 and 30 < len(line) < 140:
                quotes.append(line.strip())

        summary = (
            f"Sentiment {lbl} (score {score:+.2f}). "
            f"Tickers: {', '.join(tickers[:6]) or '—'}. "
            f"Key levels: "
            + (", ".join([f"{kl.symbol} {kl.level:.2f}" for kl in key_levels[:5]]) if key_levels else "—")
        )

        resp = SentimentResponse(
            channel=channel,
            video_id=latest["video_id"],
            video_url=latest["url"],
            published_at=latest["published_at"],
            title=latest["title"],
            summary=summary,
            sentiment_label=lbl,
            sentiment_score=score,
            tickers=tickers,
            key_levels=key_levels[:12],
            risks=risks[:5],
            quotes=quotes[:3],
            raw_excerpt=transcript[:500] if transcript else "",
        )

        _CACHE["data"], _CACHE["ts"], _CACHE["key"] = resp, now, key
        return resp
