"""S3 operations for raw and processed data."""
import io
import json
from datetime import datetime, timezone
from typing import Optional

import boto3
import pandas as pd
from loguru import logger

from src.config.settings import config


class S3Manager:
    """Manages S3 read/write for the data lake."""

    def __init__(self):
        self.s3 = boto3.client("s3", region_name=config.aws.region)
        self.raw_bucket = config.aws.s3_bucket_raw
        self.processed_bucket = config.aws.s3_bucket_processed

    def _generate_key(self, data_source: str, suffix: str = "") -> str:
        """Generate a partitioned S3 key: source/year/month/day/file."""
        now = datetime.now(timezone.utc)
        partition = f"{now.year}/{now.month:02d}/{now.day:02d}"
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        return f"{data_source}/{partition}/{timestamp}{suffix}.parquet"

    def upload_dataframe(
        self,
        df: pd.DataFrame,
        data_source: str,
        bucket: Optional[str] = None,
        key_override: Optional[str] = None,
    ) -> str:
        """Upload a DataFrame as Parquet to S3."""
        bucket = bucket or self.raw_bucket
        key = key_override or self._generate_key(data_source)

        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False, engine="pyarrow")
        buffer.seek(0)

        self.s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        logger.info(f"Uploaded {len(df)} rows to s3://{bucket}/{key}")
        return f"s3://{bucket}/{key}"

    def read_dataframe(self, bucket: str, key: str) -> pd.DataFrame:
        """Read a Parquet file from S3 into a DataFrame."""
        obj = self.s3.get_object(Bucket=bucket, Key=key)
        buffer = io.BytesIO(obj["Body"].read())
        return pd.read_parquet(buffer, engine="pyarrow")

    def list_keys(
        self, bucket: str, prefix: str, max_keys: int = 1000
    ) -> list:
        """List object keys under a prefix."""
        paginator = self.s3.get_paginator("list_objects_v2")
        keys = []

        for page in paginator.paginate(
            Bucket=bucket, Prefix=prefix, PaginationConfig={"MaxItems": max_keys}
        ):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])

        return keys

    def upload_json(self, data: dict, key: str, bucket: Optional[str] = None) -> str:
        """Upload a JSON document to S3."""
        bucket = bucket or self.raw_bucket
        self.s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data, default=str),
            ContentType="application/json",
        )
        return f"s3://{bucket}/{key}"
