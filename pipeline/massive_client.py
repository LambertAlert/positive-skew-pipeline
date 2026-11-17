"""
pipeline/massive_client.py
Thin Massive (ex-Polygon) REST client with disk caching.

Endpoints used
- /v2/aggs/grouped/locale/us/market/stocks/{date}   one call = every US ticker's daily bar (T,o,h,l,c,v,vw,n,t)
- /v3/reference/tickers                              type=CS / ADRC, active true and false (paginated)
- /v3/reference/tickers/{ticker}                     market_cap, weighted_shares_outstanding, sic_code

The key is read from MASSIVE_API_KEY (or POLYGON_API_KEY). Bars are split-adjusted (adjusted=true);
Massive does not adjust daily aggregates for dividends.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

from .config import Config


class MassiveError(RuntimeError):
    pass


class MassiveForbidden(MassiveError):
    """403: the plan doesn't include this data (e.g. dates older than the plan's history)."""


class MassiveClient:
    def __init__(self, cfg: Config, api_key: Optional[str] = None, session=None):
        self.cfg = cfg
        src = next((k for k in cfg.api_key_env if os.environ.get(k)), None)
        self.key_source = "argument" if api_key else (src or "none")
        self.key = (api_key or (os.environ[src] if src else "") or "").strip().strip('"').strip("'") or None
        if not self.key:
            raise MassiveError(f"No API key: set one of {cfg.api_key_env}")
        if session is None:
            if requests is None:
                raise MassiveError("pip install requests")
            session = requests.Session()
        self.s = session
        self.cache = Path(cfg.cache_dir)
        self._lock, self._last = threading.Lock(), 0.0
        (self.cache / "grouped").mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- http
    def _get(self, url: str, params: Optional[dict] = None, tries: int = 5) -> dict:
        params = dict(params or {})
        params["apiKey"] = self.key
        if not url.startswith("http"):
            url = self.cfg.base_url + url
        delay = 1.0
        for attempt in range(tries):
            if self.cfg.calls_per_minute:
                with self._lock:  # simple global pacing across threads
                    wait = self._last + 60.0 / self.cfg.calls_per_minute - time.time()
                    if wait > 0:
                        time.sleep(wait)
                    self._last = time.time()
            r = self.s.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            if r.status_code == 404:
                return {}
            if r.status_code == 401:
                hint = (f"the key in {self.key_source}" if self.key_source in self.cfg.api_key_env else "the key you entered")
                raise MassiveError(
                    f"Massive rejected the API key (401: {r.text[:80]}). It came from {hint} "
                    f"(starts '{self.key[:4]}…', {len(self.key)} chars). Check it on your Massive dashboard; "
                    f"to change a saved one in PowerShell: setx {self.key_source if self.key_source in self.cfg.api_key_env else 'MASSIVE_API_KEY'} \"<real key>\" "
                    f"and open a new terminal.")
            if r.status_code == 403:
                raise MassiveForbidden(f"403 {url}: {r.text[:160]}")
            raise MassiveError(f"{r.status_code} {url}: {r.text[:200]}")
        raise MassiveError(f"gave up after {tries} tries: {url}")

    def _paginate(self, path: str, params: dict) -> List[dict]:
        out, data = [], self._get(path, params)
        out.extend(data.get("results", []) or [])
        while data.get("next_url"):
            data = self._get(data["next_url"])  # next_url carries the cursor; apiKey re-added
            out.extend(data.get("results", []) or [])
        return out

    # ---------------------------------------------------------------- grouped daily bars
    def grouped_day(self, day: str) -> pd.DataFrame:
        """All US tickers' bar for one date, cached to parquet. Holidays cache as empty frames."""
        f = self.cache / "grouped" / f"{day}.parquet"
        if f.exists():
            return pd.read_parquet(f)
        try:
            data = self._get(f"/v2/aggs/grouped/locale/us/market/stocks/{day}", {"adjusted": "true"})
        except MassiveForbidden:  # outside the plan's (rolling) window: skip, don't cache
            return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume", "vwap", "trades"])
        rows = data.get("results", []) or []
        df = pd.DataFrame(rows)
        if len(df):
            df = df.rename(columns={"T": "ticker", "o": "open", "h": "high", "l": "low",
                                    "c": "close", "v": "volume", "vw": "vwap", "n": "trades"})
            df["date"] = pd.Timestamp(day)
            keep = ["date", "ticker", "open", "high", "low", "close", "volume", "vwap", "trades"]
            df = df[[c for c in keep if c in df.columns]]
        else:
            df = pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume", "vwap", "trades"])
        today = pd.Timestamp.today().normalize()
        if pd.Timestamp(day) < today:  # never cache a session that may still be in progress
            tmp = f.with_suffix(".tmp")
            df.to_parquet(tmp, index=False)
            os.replace(tmp, f)
        return df

    def earliest_available(self, start: str, end: str) -> str:
        """First business day in [start, end] the plan can serve grouped bars for (binary search,
        about a dozen calls, cached a week). Holidays answer 200 with no rows, so they count as served."""
        f = self.cache / "plan_history.json"
        if f.exists() and (time.time() - f.stat().st_mtime) < 7 * 86400:
            got = json.loads(f.read_text())
            if got.get("start", "") <= start <= got["earliest"] or got.get("start") == start:
                return max(got["earliest"], start)
        days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]

        def ok(day):
            if (self.cache / "grouped" / f"{day}.parquet").exists():
                return True
            try:
                self._get(f"/v2/aggs/grouped/locale/us/market/stocks/{day}", {"adjusted": "true"})
                return True
            except MassiveForbidden:
                return False

        lo, hi = 0, len(days) - 1
        if ok(days[lo]):
            earliest = days[lo]
        else:
            while lo < hi:
                mid = (lo + hi) // 2
                if ok(days[mid]):
                    hi = mid
                else:
                    lo = mid + 1
            earliest = days[lo]
        f.write_text(json.dumps({"start": start, "earliest": earliest}))
        return earliest

    def grouped_range(self, start: str, end: str, workers: int = 8,
                      tickers: Optional[Iterable[str]] = None) -> pd.DataFrame:
        if self.cfg.probe_history:
            start = max(start, self.earliest_available(start, end))
        days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]
        keep = set(tickers) if tickers is not None else None
        frames: Dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self.grouped_day, d): d for d in days}
            for fu in as_completed(futs):
                df = fu.result()
                if keep is not None and len(df):
                    df = df[df["ticker"].isin(keep)]
                frames[futs[fu]] = df
        out = pd.concat([frames[d] for d in days if len(frames[d])], ignore_index=True)
        return out.sort_values(["ticker", "date"]).reset_index(drop=True)

    # ---------------------------------------------------------------- reference
    def stock_set(self, max_age_days: int = 7) -> pd.DataFrame:
        """Common stocks + ADRs (active and, optionally, delisted). Cached a week."""
        f = self.cache / "stock_set.json"
        if f.exists() and (time.time() - f.stat().st_mtime) < max_age_days * 86400:
            return pd.DataFrame(json.loads(f.read_text()))
        rows = []
        actives = ["true", "false"] if self.cfg.include_delisted else ["true"]
        for typ in self.cfg.stock_types:
            for act in actives:
                rows += self._paginate("/v3/reference/tickers",
                                       {"market": "stocks", "type": typ, "active": act, "limit": 1000})
        df = pd.DataFrame(rows)
        cols = [c for c in ["ticker", "name", "type", "active", "primary_exchange", "delisted_utc", "cik"] if c in df.columns]
        df = df[cols].drop_duplicates("ticker", keep="first")
        tmp = f.with_suffix(".tmp")
        tmp.write_text(df.to_json(orient="records"))
        os.replace(tmp, f)
        return df

    def ticker_details(self, tickers: Iterable[str], asof: Optional[Dict[str, str]] = None,
                       workers: int = 8) -> pd.DataFrame:
        """Shares outstanding, market cap and SIC per ticker, cached on disk.

        asof maps ticker -> date (used for delisted names, whose current details 404).
        """
        f = self.cache / "ticker_details.json"
        cache = json.loads(f.read_text()) if f.exists() else {}
        todo = [t for t in tickers if t not in cache]

        def one(t):
            params = {"date": asof[t]} if asof and t in asof else {}
            res = self._get(f"/v3/reference/tickers/{t}", params).get("results") or {}
            return t, {
                "shares": res.get("weighted_shares_outstanding") or res.get("share_class_shares_outstanding"),
                "market_cap": res.get("market_cap"),
                "sic_code": res.get("sic_code"),
                "sic_description": res.get("sic_description"),
            }

        if todo:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for t, d in ex.map(one, todo):
                    cache[t] = d
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache))
            os.replace(tmp, f)
        return pd.DataFrame([{"ticker": t, **cache[t]} for t in tickers if t in cache])
