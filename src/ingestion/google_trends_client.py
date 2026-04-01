"""Google Trends client for fetching search interest data."""
import time
from typing import List

import pandas as pd
from loguru import logger
from pytrends.request import TrendReq


class GoogleTrendsClient:
    """Fetches Google Trends data for market-related keywords."""

    def __init__(self, tz: int = 360):
        self.pytrends = TrendReq(hl="en-US", tz=tz, retries=3, backoff_factor=1.0)

    def fetch_interest_over_time(
        self,
        keywords: List[str],
        timeframe: str = "today 12-m",
        geo: str = "",
    ) -> pd.DataFrame:
        """
        Fetch Google Trends interest over time for given keywords.

        Args:
            keywords: Up to 5 search terms
            timeframe: e.g. 'today 12-m', 'today 3-m', '2024-01-01 2024-12-31'
            geo: Country code or empty for worldwide

        Returns DataFrame indexed by date with a column per keyword (0-100 scale).
        """
        keywords = keywords[:5]

        try:
            self.pytrends.build_payload(
                keywords, cat=0, timeframe=timeframe, geo=geo, gprop="",
            )
            df = self.pytrends.interest_over_time()

            if df.empty:
                logger.warning(f"No Trends data for keywords: {keywords}")
                return pd.DataFrame()

            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])

            df.index = pd.to_datetime(df.index, utc=True)
            logger.info(f"Trends data: {len(df)} points for {keywords}")
            return df

        except Exception as e:
            logger.error(f"Google Trends error for {keywords}: {e}")
            return pd.DataFrame()

    def fetch_related_queries(self, keyword: str) -> dict:
        """Fetch rising and top related queries for a keyword."""
        self.pytrends.build_payload([keyword], timeframe="today 3-m")
        related = self.pytrends.related_queries()

        result = {"keyword": keyword, "top": [], "rising": []}
        if keyword in related:
            top = related[keyword].get("top")
            rising = related[keyword].get("rising")
            if top is not None and not top.empty:
                result["top"] = top.to_dict("records")
            if rising is not None and not rising.empty:
                result["rising"] = rising.to_dict("records")

        return result

    def fetch_batch_interest(
        self,
        keyword_groups: List[List[str]],
        timeframe: str = "today 12-m",
        delay: float = 2.0,
    ) -> pd.DataFrame:
        """
        Fetch trends for multiple keyword groups with rate limiting.
        Merges results into a single DataFrame.
        """
        frames = []

        for i, group in enumerate(keyword_groups):
            logger.info(f"Trends batch {i + 1}/{len(keyword_groups)}: {group}")
            df = self.fetch_interest_over_time(group, timeframe=timeframe)
            if not df.empty:
                frames.append(df)
            time.sleep(delay)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()]
        return combined
