import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

export const fetchLatestPredictions = (limit = 50) =>
  api.get(`/predictions/latest?limit=${limit}`).then(r => r.data);

export const fetchMarketPredictions = (marketId: string) =>
  api.get(`/predictions/market/${marketId}`).then(r => r.data);

export const fetchBrierScores = (model = 'xgboost') =>
  api.get(`/evaluation/brier-scores?model=${model}`).then(r => r.data);

export const fetchCalibrationData = (model = 'xgboost', bins = 10) =>
  api.get(`/evaluation/calibration?model=${model}&bins=${bins}`).then(r => r.data);

export const fetchFavoriteLongshotBias = () =>
  api.get('/evaluation/bias/favorite-longshot').then(r => r.data);

export const fetchCategoryBias = () =>
  api.get('/evaluation/bias/category').then(r => r.data);

export const fetchTimeHorizonAnalysis = () =>
  api.get('/evaluation/time-horizon').then(r => r.data);

export const fetchFeatureImportance = (model = 'xgboost') =>
  api.get(`/features/importance?model=${model}`).then(r => r.data);

export const fetchFeatureCatalog = () =>
  api.get('/features/catalog').then(r => r.data);

export const fetchModelComparison = () =>
  api.get('/predictions/compare').then(r => r.data);

export default api;
