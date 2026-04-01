export interface Market {
  market_id: string;
  question: string;
  category: string;
  current_price: number;
  volume: number;
  end_date: string;
}

export interface Prediction {
  market_id: string;
  model_name: string;
  predicted_probability: number;
  actual_outcome: number | null;
  prediction_date: string;
  brier_score: number | null;
  question?: string;
  category?: string;
}

export interface CalibrationPoint {
  bin_center: number;
  actual_frequency: number;
  predicted_mean: number;
  count: number;
  calibration_error: number;
}

export interface BrierDecomposition {
  brier_score: number;
  reliability: number;
  resolution: number;
  uncertainty: number;
  skill_score: number;
}

export interface BiasPoint {
  bin_center: number;
  predicted_prob: number;
  actual_freq: number;
  bias: number;
  bias_direction: string;
  n_samples: number;
}

export interface TimeHorizonResult {
  horizon: string;
  brier_score: number;
  n_samples: number;
  mean_prediction: number;
  base_rate: number;
}

export interface FeatureImportanceItem {
  feature: string;
  importance: number;
}

export interface SentimentTrend {
  date: string;
  vader_mean: number;
  bert_mean: number;
  mention_count: number;
  price: number;
}

export interface CategoryBias {
  category: string;
  brier_score: number;
  mean_bias: number;
  mean_prediction: number;
  base_rate: number;
  n_samples: number;
}
