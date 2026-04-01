"""Tests for PolymarketClient - uses mocked HTTP responses."""
import pytest
import responses

from src.ingestion.polymarket_client import PolymarketClient


@responses.activate
def test_fetch_all_markets_returns_dataframe():
    responses.add(
        responses.GET,
        "https://gamma-api.polymarket.com/markets",
        json=[
            {"id": "1", "question": "Will X happen?", "outcomes": '["Yes","No"]',
             "active": True, "closed": False, "liquidity": 50000, "volume": 100000},
        ],
        status=200,
    )
    responses.add(
        responses.GET,
        "https://gamma-api.polymarket.com/markets",
        json=[],
        status=200,
    )

    client = PolymarketClient()
    df = client.fetch_all_markets()
    assert len(df) == 1
    assert "question" in df.columns


@responses.activate
def test_fetch_market_history_parses_timestamps():
    responses.add(
        responses.GET,
        "https://clob.polymarket.com/prices-history",
        json={"history": [{"t": 1700000000, "p": "0.65"}]},
        status=200,
    )

    client = PolymarketClient()
    df = client.fetch_market_history("token_abc")
    assert len(df) == 1
    assert df.iloc[0]["price"] == 0.65
    assert df.iloc[0]["token_id"] == "token_abc"


@responses.activate
def test_fetch_market_history_empty():
    responses.add(
        responses.GET,
        "https://clob.polymarket.com/prices-history",
        json={"history": []},
        status=200,
    )

    client = PolymarketClient()
    df = client.fetch_market_history("token_empty")
    assert len(df) == 0


@responses.activate
def test_compute_spread():
    book = {
        "bids": [{"price": "0.55", "size": "100"}, {"price": "0.54", "size": "200"}],
        "asks": [{"price": "0.58", "size": "150"}, {"price": "0.59", "size": "100"}],
    }
    spread = PolymarketClient._compute_spread(book)
    assert abs(spread - 0.03) < 0.001


def test_compute_spread_empty():
    assert PolymarketClient._compute_spread({"bids": [], "asks": []}) != PolymarketClient._compute_spread({"bids": [], "asks": []})  # NaN != NaN
    import math
    assert math.isnan(PolymarketClient._compute_spread({"bids": [], "asks": []}))
