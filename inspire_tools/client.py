"""INSPIRE-HEP REST API client with rate limiting."""

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import requests

BASE_URL = "https://inspirehep.net/api"

# INSPIRE allows 15 requests per 5-second window.
# We use 14 to leave a small margin.
_RATE_LIMIT_MAX = 14
_RATE_LIMIT_WINDOW = 5.0  # seconds


class _RateLimiter:
    """Thread-safe sliding-window rate limiter."""

    def __init__(
        self,
        max_requests: int = _RATE_LIMIT_MAX,
        window: float = _RATE_LIMIT_WINDOW,
    ):
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()
        self._max = max_requests
        self._window = window

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while (
                    self._timestamps
                    and now - self._timestamps[0] > self._window
                ):
                    self._timestamps.popleft()
                if len(self._timestamps) < self._max:
                    self._timestamps.append(now)
                    return
                wait = self._window - (now - self._timestamps[0]) + 0.05
            time.sleep(max(wait, 0.05))


# Module-level shared rate limiter (per-IP limit from INSPIRE).
_global_limiter = _RateLimiter()


@dataclass
class InspireClient:
    """Thin wrapper around the INSPIRE-HEP REST API."""

    base_url: str = BASE_URL
    session: requests.Session = field(default_factory=requests.Session)
    _limiter: _RateLimiter = field(default_factory=lambda: _global_limiter, repr=False)

    def _get(
        self, endpoint: str, params: dict | None = None, _retries: int = 3,
    ) -> dict:
        self._limiter.acquire()
        url = f"{self.base_url}/{endpoint}"
        resp = self.session.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", 5))
            time.sleep(retry_after)
            return self._get(endpoint, params, _retries=_retries)
        if resp.status_code >= 500 and _retries > 0:
            time.sleep(2)
            return self._get(endpoint, params, _retries=_retries - 1)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Literature
    # ------------------------------------------------------------------

    def search_literature(
        self,
        query: str,
        size: int = 100,
        page: int = 1,
        sort: str = "mostrecent",
        fields: str | None = None,
    ) -> dict:
        """Search the literature endpoint.

        Returns the raw JSON response including ``hits.total`` and
        ``hits.hits`` with paper metadata.
        """
        params: dict = {"q": query, "size": size, "page": page, "sort": sort}
        if fields:
            params["fields"] = fields
        return self._get("literature", params)

    def search_literature_all(
        self,
        query: str,
        sort: str = "mostrecent",
        fields: str | None = None,
        max_results: int = 10_000,
    ) -> list[dict]:
        """Paginate through all results for *query* and return a flat list of hits."""
        page = 1
        size = min(250, max_results)
        hits: list[dict] = []
        while True:
            data = self.search_literature(
                query, size=size, page=page, sort=sort, fields=fields,
            )
            batch = data.get("hits", {}).get("hits", [])
            hits.extend(batch)
            total = data.get("hits", {}).get("total", 0)
            if not batch or len(hits) >= total or len(hits) >= max_results:
                break
            page += 1
        return hits[:max_results]

    def get_literature(self, recid: int | str) -> dict:
        """Fetch a single literature record by INSPIRE record ID."""
        return self._get(f"literature/{recid}")

    # ------------------------------------------------------------------
    # Authors
    # ------------------------------------------------------------------

    def search_authors(self, query: str, size: int = 10) -> dict:
        return self._get("authors", {"q": query, "size": size})

    def get_author(self, recid: int | str) -> dict:
        return self._get(f"authors/{recid}")

    def get_author_by_orcid(self, orcid: str) -> dict:
        return self._get(f"orcid/{orcid}")
