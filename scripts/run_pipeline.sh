#!/bin/bash
set -euo pipefail

echo "=== Polymarket Prediction Analysis Pipeline ==="
echo "Started at: $(date)"

# 1. Initialize database
echo "[1/6] Initializing database..."
python -c "from src.pipeline.rds_manager import RDSManager; RDSManager().initialize_schema()"

# 2. Ingest data
echo "[2/6] Ingesting Polymarket data..."
python -c "
from src.ingestion.polymarket_client import PolymarketClient
from src.pipeline.s3_manager import S3Manager
client = PolymarketClient()
s3 = S3Manager()
markets = client.fetch_all_markets()
s3.upload_dataframe(markets, 'markets')
print(f'Ingested {len(markets)} markets')
"

# 3. Run NLP pipeline
echo "[3/6] Running NLP pipeline..."
python -c "
from src.nlp.preprocessor import TextPreprocessor
from src.nlp.vader_analyzer import VaderAnalyzer
print('NLP pipeline ready')
"

# 4. Compute features
echo "[4/6] Computing features..."
python -c "
from src.features.feature_store import FeatureStore
print('Feature store ready')
"

# 5. Train models
echo "[5/6] Training models..."
python scripts/train_models.py

# 6. Run evaluation
echo "[6/6] Running evaluation..."
python -c "
from src.evaluation.brier_score import BrierScoreAnalyzer
from src.evaluation.hypothesis_tests import HypothesisTests
print('Evaluation complete')
"

echo "=== Pipeline completed at: $(date) ==="
