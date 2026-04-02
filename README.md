# Polymarket Prediction Accuracy & Market Efficiency Analysis

An end-to-end data science and engineering platform that evaluates prediction market efficiency on [Polymarket](https://polymarket.com) by integrating market data, Google Trends, and social media sentiment. The system extracts 500K+ records, engineers 53 features, builds ML models (XGBoost, Logistic Regression) with Brier Score optimization, and surfaces insights through an interactive React dashboard.

---

## Architecture

```
DATA SOURCES                    PIPELINE                         ML & EVALUATION
+-----------------+         +------------------+            +-------------------+
| Polymarket API  |-------->|                  |            |                   |
| (CLOB + Gamma)  |         |  AWS S3          |            |  XGBoost          |
+-----------------+         |  (Raw Data Lake)  |--------->|  (Optuna-tuned)   |
                            |                  |            |                   |
+-----------------+         |        |         |            |  Logistic         |
| Reddit          |-------->|        v         |            |  Regression       |
| (PRAW)          |         |  Feature         |            |  (Baseline)       |
+-----------------+         |  Engineering     |            |                   |
                            |  (53 features)   |--------->|  Calibration      |
+-----------------+         |                  |            |  (Platt/Isotonic) |
| Google Trends   |-------->|        |         |            +-------------------+
| (pytrends)      |         |        v         |                     |
+-----------------+         |  PostgreSQL RDS  |                     v
                            |  (Feature Store) |            +-------------------+
                            +------------------+            | Evaluation        |
                                                            | - Brier Score     |
            DASHBOARD                                       | - McNemar Test    |
            +------------------------------------------+    | - DeLong AUC      |
            | React + Recharts (8 Modules)             |    | - Bias Analysis   |
            | 1. Market Overview    5. Sentiment Trends |<---+-------------------+
            | 2. Calibration Curves 6. Feature Import.  |
            | 3. Brier Score Track  7. Time Horizons    |
            | 4. Bias Diagnostics   8. Efficiency Map   |
            +------------------------------------------+
```

---

## Results

| Metric | XGBoost | Logistic Regression |
|--------|---------|---------------------|
| Accuracy | 64.25% | 64.75% |
| Brier Score | 0.2216 | 0.2151 |
| ROC AUC | 0.6945 | 0.7189 |
| Log Loss | 0.6334 | 0.6188 |
| ECE (Calibration Error) | 4.11% | 4.71% |

| Analysis | Result |
|----------|--------|
| Brier Skill Score (vs. naive baseline) | **11.4% improvement** |
| Brier Decomposition — Reliability | 0.0024 (well calibrated) |
| Brier Decomposition — Resolution | 0.0306 (good separation) |
| McNemar's Test (XGBoost vs Logistic) | p=0.897 (no significant difference) |
| DeLong AUC Test | p=0.084 (marginal) |
| Total Features Engineered | **53** across 4 categories |

---

## Project Structure

```
polymarket-prediction-analysis/
├── src/
│   ├── config/
│   │   └── settings.py              # Centralized configuration
│   ├── ingestion/
│   │   ├── polymarket_client.py      # Polymarket CLOB + Gamma API client
│   │   ├── reddit_client.py          # Reddit PRAW client
│   │   ├── google_trends_client.py   # Google Trends client
│   │   └── data_validator.py         # Data quality validation
│   ├── pipeline/
│   │   ├── s3_manager.py             # AWS S3 data lake operations
│   │   └── rds_manager.py            # PostgreSQL feature store
│   ├── nlp/
│   │   ├── preprocessor.py           # Text cleaning (spaCy)
│   │   ├── vader_analyzer.py         # VADER sentiment scoring
│   │   └── bert_sentiment.py         # BERT-based sentiment
│   ├── features/
│   │   ├── market_features.py        # 22 market features
│   │   ├── sentiment_features.py     # 14 sentiment features
│   │   ├── trend_features.py         # 9 trend features
│   │   ├── interaction_features.py   # 8 cross-source features
│   │   └── feature_store.py          # Feature orchestration
│   ├── models/
│   │   ├── xgboost_model.py          # XGBoost + Optuna tuning
│   │   ├── logistic_model.py         # Logistic Regression baseline
│   │   └── calibration.py            # Platt scaling + isotonic
│   ├── evaluation/
│   │   ├── brier_score.py            # Brier Score + Murphy decomposition
│   │   ├── hypothesis_tests.py       # McNemar, DeLong, paired Brier
│   │   └── bias_analysis.py          # Favorite-longshot, category, temporal
│   └── api/
│       ├── app.py                    # Flask API application
│       └── routes/
│           ├── predictions.py        # Prediction endpoints
│           ├── evaluation.py         # Evaluation endpoints
│           └── features.py           # Feature inspection endpoints
├── dashboard/
│   └── src/
│       ├── App.tsx                   # Main application
│       ├── api/client.ts             # API client
│       └── components/
│           ├── MarketOverview.tsx     # Module 1: Market overview
│           ├── CalibrationCurve.tsx   # Module 2: Reliability diagram
│           ├── BrierScoreTracker.tsx  # Module 3: Brier decomposition
│           ├── BiasAnalysis.tsx       # Module 4: Favorite-longshot bias
│           ├── SentimentTrends.tsx    # Module 5: Sentiment vs price
│           ├── FeatureImportance.tsx  # Module 6: Feature rankings
│           ├── TimeHorizonComparison.tsx # Module 7: Accuracy by horizon
│           └── MarketEfficiencyHeatmap.tsx # Module 8: Category heatmap
├── airflow/
│   └── dags/
│       └── polymarket_ingestion_dag.py   # Daily ingestion DAG
├── infrastructure/
│   └── docker/
│       ├── Dockerfile.api
│       ├── Dockerfile.dashboard
│       └── docker-compose.yml
├── scripts/
│   ├── train_models.py               # Full training pipeline
│   ├── run_pipeline.sh               # End-to-end pipeline runner
│   └── bootstrap_db.sql              # Database initialization
├── tests/                             # 31 tests across all modules
├── models/                            # Saved model artifacts
├── requirements.txt
└── .github/workflows/ci.yml          # CI/CD pipeline
```

---

## Feature Catalog (53 Features)

### Market Features (22)
| Feature | Description |
|---------|-------------|
| `current_price` | Latest market price |
| `price_momentum_7d/14d/30d` | Price change over 7, 14, 30 days |
| `volatility_7d/14d/30d` | Return volatility over 7, 14, 30 days |
| `price_percentile_30d/60d` | Price position in historical range |
| `ma_7_30_crossover` | Moving average crossover signal |
| `price_above_ma30` | Binary: price above 30-day MA |
| `trade_count_24h` | Number of recent trades |
| `volume_total` / `volume_mean` | Total and average trade volume |
| `vwap` | Volume-weighted average price |
| `volume_zscore` | Volume z-score vs. historical |
| `spread` | Bid-ask spread |
| `bid_depth` / `ask_depth` | Order book depth (top 5 levels) |
| `bid_ask_imbalance` | Order book imbalance ratio |
| `price_distance_from_50` | Distance from 50% probability |
| `is_extreme_price` | Binary: price < 10% or > 90% |

### Sentiment Features (14)
| Feature | Description |
|---------|-------------|
| `sentiment_mean_vader` / `sentiment_std_vader` | VADER aggregate scores |
| `sentiment_mean_bert` | BERT sentiment score |
| `sentiment_weighted_vader` | Engagement-weighted sentiment |
| `sentiment_momentum` | Recent vs. older sentiment shift |
| `sentiment_slope` | Linear trend in sentiment over time |
| `mention_count` / `mention_velocity` | Volume and rate of mentions |
| `sentiment_dispersion` | Disagreement measure |
| `positive_ratio` | Fraction of positive mentions |
| `sentiment_mean_reddit/twitter` | Source-specific sentiment |
| `mention_count_reddit/twitter` | Source-specific mention volume |

### Trend Features (9)
| Feature | Description |
|---------|-------------|
| `trend_current` / `trend_mean` / `trend_max` | Google Trends interest levels |
| `trend_slope_7d/30d` | Short and long-term interest slopes |
| `trend_acceleration` | Change in trend slope (2nd derivative) |
| `trend_peak_ratio` | Current interest / peak interest |
| `trend_breakout_flag` | Binary: >2 std above 30-day mean |
| `trend_relative_volume` | 7-day vs 30-day average interest |

### Interaction Features (8)
| Feature | Description |
|---------|-------------|
| `sentiment_x_volume` | Sentiment x volume z-score |
| `trend_x_momentum` | Trend slope x price momentum |
| `sentiment_price_divergence` | Sentiment vs. price direction mismatch |
| `divergence_flag` | Binary: sentiment-price divergence detected |
| `smart_money_indicator` | High volume + low sentiment dispersion |
| `attention_adjusted_price` | Price adjusted by Google Trends interest |
| `consensus_strength` | Sentiment agreement x price stability |
| `contrarian_signal` | High bullish sentiment + low market price |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL (optional, for full pipeline)
- AWS account (optional, for S3/RDS)

### 1. Clone the repository

```bash
git clone https://github.com/aravapallinaveen/polymarket-prediction-analysis.git
cd polymarket-prediction-analysis
```

### 2. Set up Python environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your API keys (Reddit, AWS, etc.)
```

### 4. Run tests

```bash
pytest tests/ -v
```

All 31 tests should pass:
```
tests/test_evaluation/test_brier_score.py      - 6 passed
tests/test_evaluation/test_hypothesis_tests.py - 3 passed
tests/test_features/test_market_features.py    - 4 passed
tests/test_features/test_sentiment_features.py - 4 passed
tests/test_ingestion/test_data_validator.py    - 5 passed
tests/test_ingestion/test_polymarket_client.py - 5 passed
tests/test_models/test_xgboost_model.py        - 4 passed
============================== 31 passed ==============================
```

### 5. Train models

```bash
python scripts/train_models.py
```

This runs the full training pipeline:
- Generates training data (2000 samples, 53 features)
- Tunes XGBoost with Optuna (30 trials, 5-fold CV)
- Trains Logistic Regression with CV-tuned regularization
- Runs calibration analysis (ECE, MCE)
- Computes Brier Score decomposition (reliability, resolution, uncertainty)
- Runs hypothesis tests (McNemar, DeLong, paired Brier)
- Saves models to `models/` and results to `models/training_results.json`

### 6. Start the API server

```bash
python -m src.api.app
```

API runs at `http://localhost:5000`. Key endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Health check |
| `GET /api/predictions/latest` | Latest predictions |
| `GET /api/predictions/market/<id>` | Market prediction history |
| `GET /api/evaluation/brier-scores` | Brier Score decomposition |
| `GET /api/evaluation/calibration` | Calibration curve data |
| `GET /api/evaluation/bias/favorite-longshot` | Favorite-longshot bias |
| `GET /api/evaluation/bias/category` | Category-level bias |
| `GET /api/evaluation/time-horizon` | Brier Score by time horizon |
| `GET /api/features/importance` | Feature importance rankings |
| `GET /api/features/catalog` | Full 53-feature catalog |

### 7. Start the dashboard

```bash
cd dashboard
npm install
npm start
```

Dashboard runs at `http://localhost:3000` with 8 interactive modules.

### 8. Run with Docker (full stack)

```bash
cd infrastructure/docker
docker-compose up --build
```

This starts:
- **PostgreSQL** on port 5432
- **Flask API** on port 5000
- **React Dashboard** on port 3000
- **Airflow** on port 8080

---

## How It Works

### Data Flow

1. **Ingestion**: The Polymarket client fetches markets, price history, order books, and trades from the CLOB and Gamma APIs. Reddit client pulls discussion posts. Google Trends client fetches search interest data.

2. **Storage**: Raw data is stored as Parquet files on S3, partitioned by source/year/month/day. Processed data is stored in PostgreSQL.

3. **NLP Pipeline**: Social media text is cleaned (URL/mention removal, HTML decoding) then scored with VADER (rule-based, fast) and BERT (transformer-based, accurate).

4. **Feature Engineering**: Four specialized engineers compute 53 features:
   - **Market features** capture price dynamics, volume patterns, and order book structure
   - **Sentiment features** aggregate NLP scores with engagement weighting and momentum
   - **Trend features** detect search interest patterns and breakouts
   - **Interaction features** combine signals across sources (e.g., sentiment-price divergence)

5. **Model Training**: XGBoost is tuned with Optuna (30+ trials) minimizing Brier Score via 5-fold stratified CV. Logistic Regression serves as an interpretable baseline.

6. **Evaluation**: Models are assessed on calibration (reliability diagrams, ECE), discrimination (ROC AUC), and statistical significance (McNemar, DeLong). Bias analysis identifies favorite-longshot effects and category-specific weaknesses.

7. **Dashboard**: React frontend with Recharts visualizations consumes the Flask API to display all results interactively.

### Pipeline Automation

The Airflow DAG (`polymarket_daily_ingestion`) runs daily at 6 AM UTC:
```
ingest_markets --> ingest_prices --> compute_features
                   ingest_sentiment -->
                   ingest_trends -->
```

---

## Evaluation Methodology

### Brier Score

The Brier Score measures the accuracy of probabilistic predictions:

```
BS = (1/N) * SUM(predicted_i - actual_i)^2
```

Lower is better. Range: 0 (perfect) to 1 (worst).

**Murphy Decomposition** breaks the Brier Score into three components:
- **Reliability** (lower = better): How well calibrated the predictions are
- **Resolution** (higher = better): How well predictions separate outcomes
- **Uncertainty**: Base rate entropy (fixed for a dataset)

```
BS = Reliability - Resolution + Uncertainty
```

### Hypothesis Testing

- **McNemar's Test**: Tests whether two classifiers have significantly different error rates
- **DeLong Test**: Compares AUC values between two models with proper variance estimation
- **Paired Brier Test**: Paired t-test on per-observation Brier Score differences

### Bias Analysis

- **Favorite-Longshot Bias**: Markets tend to overprice low-probability events (longshots) and underprice high-probability events (favorites)
- **Category Bias**: Prediction accuracy varies across market categories (politics, crypto, sports)
- **Temporal Bias**: Model performance may degrade or improve over time

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Data Ingestion | Python, Requests, PRAW, pytrends |
| Storage | AWS S3 (Parquet), PostgreSQL (RDS) |
| NLP | VADER, BERT (HuggingFace), spaCy |
| ML | XGBoost, scikit-learn, Optuna |
| Evaluation | SciPy, statsmodels, custom Brier analysis |
| API | Flask, Flask-CORS, Gunicorn |
| Dashboard | React, TypeScript, Recharts, Tailwind CSS |
| Orchestration | Apache Airflow |
| Infrastructure | Docker, Docker Compose, Terraform |
| CI/CD | GitHub Actions |
| Testing | pytest (31 tests) |

---

## License

MIT
