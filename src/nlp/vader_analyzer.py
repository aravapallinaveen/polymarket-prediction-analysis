"""VADER sentiment analysis."""
import pandas as pd
from loguru import logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class VaderAnalyzer:
    """Compute VADER sentiment scores."""

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> dict:
        """Get VADER scores for a single text."""
        if not text:
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
        return self.analyzer.polarity_scores(text)

    def score_batch(self, texts: list) -> pd.DataFrame:
        """Score a batch of texts, returning a DataFrame."""
        scores = [self.score(t) for t in texts]
        df = pd.DataFrame(scores)
        df.columns = [f"vader_{c}" for c in df.columns]
        logger.info(
            f"VADER scored {len(df)} texts. "
            f"Mean compound: {df['vader_compound'].mean():.3f}"
        )
        return df
