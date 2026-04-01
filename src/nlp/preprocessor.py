"""Text preprocessing for NLP pipeline."""
import re
import html
from collections import Counter
from typing import List

from loguru import logger

from src.config.settings import config


class TextPreprocessor:
    """Cleans and normalizes text data for sentiment analysis."""

    def __init__(self):
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.mention_pattern = re.compile(r"@\w+")
        self.hashtag_pattern = re.compile(r"#(\w+)")
        self.special_chars = re.compile(r"[^a-zA-Z0-9\s.,!?'-]")
        self.whitespace = re.compile(r"\s+")
        self._nlp = None

    @property
    def nlp(self):
        """Lazy-load spaCy model only when needed."""
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load(config.nlp.spacy_model, disable=["ner", "parser"])
            except OSError:
                logger.warning(f"Downloading spaCy model: {config.nlp.spacy_model}")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", config.nlp.spacy_model])
                self._nlp = spacy.load(config.nlp.spacy_model, disable=["ner", "parser"])
        return self._nlp

    def clean(self, text: str) -> str:
        """Full cleaning pipeline for a single text."""
        if not text or not isinstance(text, str):
            return ""

        text = html.unescape(text)
        text = self.url_pattern.sub("", text)
        text = self.mention_pattern.sub("", text)
        text = self.hashtag_pattern.sub(r"\1", text)
        text = self.special_chars.sub(" ", text)
        text = self.whitespace.sub(" ", text).strip()

        if len(text) > config.nlp.max_text_length:
            text = text[: config.nlp.max_text_length]

        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a batch of texts."""
        return [self.clean(t) for t in texts]

    def lemmatize(self, text: str) -> str:
        """Lemmatize text using spaCy."""
        doc = self.nlp(text)
        return " ".join(
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and len(token.text) > 2
        )

    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Extract top keywords from text using spaCy POS tagging."""
        doc = self.nlp(text)
        nouns = [
            token.lemma_.lower()
            for token in doc
            if token.pos_ in ("NOUN", "PROPN") and len(token.text) > 2
        ]
        counts = Counter(nouns)
        return [word for word, _ in counts.most_common(top_n)]
