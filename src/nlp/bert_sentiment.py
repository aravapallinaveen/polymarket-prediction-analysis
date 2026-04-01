"""BERT-based sentiment analysis using HuggingFace transformers."""
from typing import List

import numpy as np
import pandas as pd
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config.settings import config


class BertSentimentAnalyzer:
    """Fine-grained sentiment using a pre-trained BERT model."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(config.nlp.hf_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.nlp.hf_model
        ).to(self.device)
        self.model.eval()
        logger.info(f"BERT sentiment model loaded on {self.device}")

    @torch.no_grad()
    def score_batch(self, texts: List[str], batch_size: int = None) -> pd.DataFrame:
        """
        Score a batch of texts.

        Returns DataFrame with columns: bert_score (0-1 normalized),
                                         bert_label (1-5 star rating)
        """
        batch_size = batch_size or config.nlp.batch_size
        all_scores = []
        all_labels = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            encodings = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=config.nlp.max_text_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1)

            # Convert 5-class probabilities to a single score in [0, 1]
            weights = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).to(self.device)
            weighted_scores = (probs * weights).sum(dim=-1) / 5.0

            labels = probs.argmax(dim=-1) + 1  # 1-5

            all_scores.extend(weighted_scores.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        df = pd.DataFrame({
            "bert_score": all_scores,
            "bert_label": all_labels,
        })
        logger.info(f"BERT scored {len(df)} texts. Mean: {df['bert_score'].mean():.3f}")
        return df
