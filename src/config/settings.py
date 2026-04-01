"""Centralized configuration with validation."""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class PolymarketConfig:
    api_base: str = os.getenv("POLYMARKET_API_BASE", "https://clob.polymarket.com")
    gamma_api: str = os.getenv("POLYMARKET_GAMMA_API", "https://gamma-api.polymarket.com")
    rate_limit_per_second: int = 5
    max_retries: int = 3
    timeout_seconds: int = 30


@dataclass(frozen=True)
class AWSConfig:
    region: str = os.getenv("AWS_REGION", "us-east-1")
    s3_bucket_raw: str = os.getenv("S3_BUCKET_RAW", "polymarket-raw-data")
    s3_bucket_processed: str = os.getenv("S3_BUCKET_PROCESSED", "polymarket-processed")
    rds_host: str = os.getenv("RDS_HOST", "localhost")
    rds_port: int = int(os.getenv("RDS_PORT", "5432"))
    rds_db: str = os.getenv("RDS_DB", "polymarket_analysis")
    rds_user: str = os.getenv("RDS_USER", "admin")
    rds_password: str = os.getenv("RDS_PASSWORD", "")

    @property
    def rds_connection_string(self) -> str:
        return (
            f"postgresql://{self.rds_user}:{self.rds_password}"
            f"@{self.rds_host}:{self.rds_port}/{self.rds_db}"
        )


@dataclass(frozen=True)
class RedditConfig:
    client_id: str = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret: str = os.getenv("REDDIT_CLIENT_SECRET", "")
    user_agent: str = os.getenv("REDDIT_USER_AGENT", "polymarket-analyzer/1.0")
    subreddits: list = field(default_factory=lambda: [
        "polymarket", "predictions", "wallstreetbets",
        "politics", "worldnews", "cryptocurrency"
    ])


@dataclass(frozen=True)
class NLPConfig:
    hf_model: str = os.getenv("HF_MODEL_NAME", "nlptown/bert-base-multilingual-uncased-sentiment")
    spacy_model: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
    max_text_length: int = 512
    batch_size: int = 64


@dataclass(frozen=True)
class ModelConfig:
    test_size: float = 0.2
    val_size: float = 0.15
    random_state: int = 42
    xgb_n_trials: int = 100
    cv_folds: int = 5
    early_stopping_rounds: int = 50


@dataclass(frozen=True)
class AppConfig:
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    aws: AWSConfig = field(default_factory=AWSConfig)
    reddit: RedditConfig = field(default_factory=RedditConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


config = AppConfig()
