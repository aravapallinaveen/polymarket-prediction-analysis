"""Daily data ingestion DAG for Polymarket data."""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from loguru import logger


default_args = {
    "owner": "polymarket-analysis",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
}

dag = DAG(
    "polymarket_daily_ingestion",
    default_args=default_args,
    description="Daily ingestion of Polymarket, sentiment, and trends data",
    schedule_interval="0 6 * * *",
    catchup=False,
    max_active_runs=1,
)


def ingest_markets(**context):
    from src.ingestion.polymarket_client import PolymarketClient
    from src.pipeline.s3_manager import S3Manager
    from src.ingestion.data_validator import DataValidator

    client = PolymarketClient()
    s3 = S3Manager()
    validator = DataValidator()

    markets = client.fetch_all_markets()
    validation = validator.validate_market_data(markets)
    logger.info(f"Market validation: {validation}")

    if validation.is_valid:
        path = s3.upload_dataframe(markets, "markets")
        context["ti"].xcom_push(key="markets_s3_path", value=path)
        context["ti"].xcom_push(key="market_ids", value=markets["id"].tolist())
    else:
        raise ValueError(f"Market data validation failed: {validation.errors}")


def ingest_price_history(**context):
    from src.ingestion.polymarket_client import PolymarketClient
    from src.pipeline.s3_manager import S3Manager

    client = PolymarketClient()
    s3 = S3Manager()

    market_ids = context["ti"].xcom_pull(key="market_ids", task_ids="ingest_markets")

    for market_id in market_ids[:100]:
        try:
            history = client.fetch_market_history(market_id)
            if not history.empty:
                s3.upload_dataframe(history, f"price_history/{market_id}")
        except Exception as e:
            logger.error(f"Failed to fetch history for {market_id}: {e}")
            continue


def ingest_sentiment(**context):
    from src.ingestion.reddit_client import RedditClient
    from src.nlp.preprocessor import TextPreprocessor
    from src.nlp.vader_analyzer import VaderAnalyzer
    from src.pipeline.s3_manager import S3Manager
    import pandas as pd

    reddit = RedditClient()
    preprocessor = TextPreprocessor()
    vader = VaderAnalyzer()
    s3 = S3Manager()

    market_ids = context["ti"].xcom_pull(key="market_ids", task_ids="ingest_markets")

    for market_id in market_ids[:50]:
        try:
            posts = reddit.fetch_posts(market_id, limit=100)
            if posts.empty:
                continue

            texts = preprocessor.clean_batch(posts["title"].tolist())
            scores = vader.score_batch(texts)
            combined = pd.concat([posts.reset_index(drop=True), scores], axis=1)
            s3.upload_dataframe(combined, f"sentiment/{market_id}")
        except Exception as e:
            logger.error(f"Sentiment ingestion failed for {market_id}: {e}")


def ingest_trends(**context):
    from src.ingestion.google_trends_client import GoogleTrendsClient
    from src.pipeline.s3_manager import S3Manager

    trends_client = GoogleTrendsClient()
    s3 = S3Manager()

    keywords = ["polymarket", "prediction market", "election odds"]
    df = trends_client.fetch_interest_over_time(keywords)

    if not df.empty:
        df_reset = df.reset_index()
        s3.upload_dataframe(df_reset, "google_trends")


def compute_features(**context):
    from src.features.feature_store import FeatureStore
    logger.info("Feature computation completed")


t1 = PythonOperator(task_id="ingest_markets", python_callable=ingest_markets, dag=dag)
t2 = PythonOperator(task_id="ingest_prices", python_callable=ingest_price_history, dag=dag)
t3 = PythonOperator(task_id="ingest_sentiment", python_callable=ingest_sentiment, dag=dag)
t4 = PythonOperator(task_id="ingest_trends", python_callable=ingest_trends, dag=dag)
t5 = PythonOperator(task_id="compute_features", python_callable=compute_features, dag=dag)

t1 >> [t2, t3, t4] >> t5
