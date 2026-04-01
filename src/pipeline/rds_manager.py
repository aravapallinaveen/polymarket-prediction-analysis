"""RDS (PostgreSQL) operations for the feature store."""
from contextlib import contextmanager

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.config.settings import config


class RDSManager:
    """Manages the PostgreSQL feature store on RDS."""

    def __init__(self):
        self.engine = create_engine(
            config.aws.rds_connection_string,
            pool_size=10,
            max_overflow=5,
            pool_pre_ping=True,
        )
        self.Session = sessionmaker(bind=self.engine)

    @contextmanager
    def session(self):
        """Context-managed session."""
        s = self.Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def initialize_schema(self):
        """Create all tables if they don't exist."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS markets (
            market_id VARCHAR(128) PRIMARY KEY,
            question TEXT NOT NULL,
            slug VARCHAR(512),
            category VARCHAR(128),
            end_date TIMESTAMP WITH TIME ZONE,
            resolved BOOLEAN DEFAULT FALSE,
            resolution_outcome VARCHAR(32),
            liquidity NUMERIC(18, 2),
            volume NUMERIC(18, 2),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS price_history (
            id BIGSERIAL PRIMARY KEY,
            market_id VARCHAR(128) REFERENCES markets(market_id),
            token_id VARCHAR(128),
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            price NUMERIC(10, 6) NOT NULL,
            UNIQUE (token_id, timestamp)
        );

        CREATE TABLE IF NOT EXISTS sentiment_scores (
            id BIGSERIAL PRIMARY KEY,
            market_id VARCHAR(128) REFERENCES markets(market_id),
            source VARCHAR(32) NOT NULL,
            text_id VARCHAR(256),
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            vader_compound NUMERIC(8, 6),
            textblob_polarity NUMERIC(8, 6),
            bert_score NUMERIC(8, 6),
            raw_text TEXT
        );

        CREATE TABLE IF NOT EXISTS google_trends (
            id BIGSERIAL PRIMARY KEY,
            market_id VARCHAR(128) REFERENCES markets(market_id),
            keyword VARCHAR(256),
            date DATE NOT NULL,
            interest_value INTEGER,
            UNIQUE (market_id, keyword, date)
        );

        CREATE TABLE IF NOT EXISTS feature_store (
            id BIGSERIAL PRIMARY KEY,
            market_id VARCHAR(128) REFERENCES markets(market_id),
            snapshot_date DATE NOT NULL,
            features JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            UNIQUE (market_id, snapshot_date)
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id BIGSERIAL PRIMARY KEY,
            market_id VARCHAR(128) REFERENCES markets(market_id),
            model_name VARCHAR(64) NOT NULL,
            model_version VARCHAR(32),
            predicted_probability NUMERIC(10, 6),
            actual_outcome SMALLINT,
            prediction_date DATE NOT NULL,
            brier_score NUMERIC(10, 6),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS model_metadata (
            id BIGSERIAL PRIMARY KEY,
            model_name VARCHAR(64) NOT NULL,
            model_version VARCHAR(32),
            feature_name VARCHAR(128),
            importance_score NUMERIC(12, 6),
            importance_rank INTEGER,
            trained_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_price_history_market
            ON price_history(market_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_sentiment_market
            ON sentiment_scores(market_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_features_market
            ON feature_store(market_id, snapshot_date);
        CREATE INDEX IF NOT EXISTS idx_predictions_market
            ON predictions(market_id, prediction_date);
        """
        with self.engine.connect() as conn:
            conn.execute(text(schema_sql))
            conn.commit()
        logger.info("Database schema initialized")

    def upsert_dataframe(
        self, df: pd.DataFrame, table: str, conflict_columns: list
    ) -> int:
        """Upsert DataFrame rows using ON CONFLICT DO UPDATE."""
        if df.empty:
            return 0

        rows_affected = 0
        with self.session() as session:
            for _, row in df.iterrows():
                cols = ", ".join(row.index)
                placeholders = ", ".join(f":{c}" for c in row.index)
                updates = ", ".join(
                    f"{c} = EXCLUDED.{c}"
                    for c in row.index
                    if c not in conflict_columns
                )
                conflict = ", ".join(conflict_columns)

                sql = f"""
                    INSERT INTO {table} ({cols})
                    VALUES ({placeholders})
                    ON CONFLICT ({conflict})
                    DO UPDATE SET {updates}
                """
                session.execute(text(sql), dict(row))
                rows_affected += 1

        logger.info(f"Upserted {rows_affected} rows into {table}")
        return rows_affected

    def query(self, sql: str, params: dict = None) -> pd.DataFrame:
        """Run a query and return a DataFrame."""
        return pd.read_sql(text(sql), self.engine, params=params)
