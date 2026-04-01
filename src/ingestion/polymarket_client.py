"""Polymarket API client for fetching markets, prices, and trade history."""
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import config


class PolymarketClient:
    """Fetches data from Polymarket CLOB and Gamma APIs."""

    def __init__(self):
        self.clob_base = config.polymarket.api_base
        self.gamma_base = config.polymarket.gamma_api
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self._last_request_time = 0.0
        self._min_interval = 1.0 / config.polymarket.rate_limit_per_second

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _get(self, url: str, params: Optional[dict] = None) -> dict:
        """Rate-limited GET request with retries."""
        self._rate_limit()
        response = self.session.get(
            url, params=params, timeout=config.polymarket.timeout_seconds
        )
        response.raise_for_status()
        return response.json()

    def fetch_all_markets(self, limit: int = 100) -> pd.DataFrame:
        """
        Fetch all available markets from Gamma API with pagination.

        Returns DataFrame with columns:
            id, question, slug, category, end_date, active,
            closed, liquidity, volume, outcomes, outcome_prices
        """
        all_markets = []
        offset = 0

        while True:
            logger.info(f"Fetching markets: offset={offset}, limit={limit}")
            data = self._get(
                f"{self.gamma_base}/markets",
                params={"limit": limit, "offset": offset, "closed": False},
            )

            if not data:
                break

            all_markets.extend(data)
            offset += limit

            if len(data) < limit:
                break

        df = pd.DataFrame(all_markets)
        logger.info(f"Fetched {len(df)} markets total")
        return df

    def fetch_market_history(
        self,
        token_id: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        fidelity: int = 60,
    ) -> pd.DataFrame:
        """
        Fetch price history for a specific market token.

        Args:
            token_id: The CLOB token ID
            start_ts: Start timestamp (unix seconds)
            end_ts: End timestamp (unix seconds)
            fidelity: Candle size in minutes (1, 5, 15, 60, 1440)

        Returns DataFrame with columns: timestamp, price, token_id
        """
        params = {"market": token_id, "interval": "max", "fidelity": fidelity}

        if start_ts:
            params["startTs"] = start_ts
        if end_ts:
            params["endTs"] = end_ts

        data = self._get(f"{self.clob_base}/prices-history", params=params)
        history = data.get("history", [])

        if not history:
            logger.warning(f"No history for token {token_id}")
            return pd.DataFrame(columns=["timestamp", "price", "token_id"])

        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
        df["price"] = df["p"].astype(float)
        df["token_id"] = token_id
        df = df[["timestamp", "price", "token_id"]].sort_values("timestamp")

        logger.info(f"Token {token_id}: {len(df)} price records")
        return df

    def fetch_order_book(self, token_id: str) -> dict:
        """Fetch current order book for a token."""
        data = self._get(
            f"{self.clob_base}/book", params={"token_id": token_id}
        )
        return {
            "bids": data.get("bids", []),
            "asks": data.get("asks", []),
            "spread": self._compute_spread(data),
            "token_id": token_id,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    def fetch_trades(
        self, token_id: str, limit: int = 500
    ) -> pd.DataFrame:
        """Fetch recent trades for a token."""
        all_trades = []
        cursor = None

        while len(all_trades) < limit:
            params = {"asset_id": token_id, "limit": min(100, limit - len(all_trades))}
            if cursor:
                params["cursor"] = cursor

            data = self._get(f"{self.clob_base}/trades", params=params)
            trades = data.get("data", [])

            if not trades:
                break

            all_trades.extend(trades)
            cursor = data.get("next_cursor")

            if not cursor:
                break

        df = pd.DataFrame(all_trades)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["match_time"], utc=True)
            df["price"] = df["price"].astype(float)
            df["size"] = df["size"].astype(float)
            df["token_id"] = token_id

        logger.info(f"Token {token_id}: {len(df)} trades")
        return df

    def fetch_resolved_markets(self) -> pd.DataFrame:
        """Fetch markets that have resolved (needed for ground truth labels)."""
        all_resolved = []
        offset = 0
        limit = 100

        while True:
            data = self._get(
                f"{self.gamma_base}/markets",
                params={"limit": limit, "offset": offset, "closed": True},
            )

            if not data:
                break

            for market in data:
                if market.get("resolved", False):
                    all_resolved.append(market)

            offset += limit
            if len(data) < limit:
                break

        df = pd.DataFrame(all_resolved)
        logger.info(f"Fetched {len(df)} resolved markets")
        return df

    @staticmethod
    def _compute_spread(book_data: dict) -> float:
        """Compute bid-ask spread from order book."""
        bids = book_data.get("bids", [])
        asks = book_data.get("asks", [])

        if not bids or not asks:
            return float("nan")

        best_bid = max(float(b["price"]) for b in bids)
        best_ask = min(float(a["price"]) for a in asks)
        return best_ask - best_bid
