import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { fetchLatestPredictions } from '../api/client';
import { Prediction } from '../types';

const MarketOverview: React.FC = () => {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchLatestPredictions(20)
      .then(setPredictions)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-4">Loading markets...</div>;

  const chartData = predictions.map(p => ({
    question: p.question?.slice(0, 40) + '...' || p.market_id.slice(0, 10),
    probability: (p.predicted_probability * 100).toFixed(1),
    value: p.predicted_probability,
    category: p.category,
  }));

  const getColor = (value: number) => {
    if (value > 0.7) return '#22c55e';
    if (value > 0.4) return '#eab308';
    return '#ef4444';
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-bold mb-4">Market Overview — Latest Predictions</h2>
      <p className="text-sm text-gray-500 mb-4">
        Top {predictions.length} markets by predicted probability (XGBoost model)
      </p>
      <ResponsiveContainer width="100%" height={500}>
        <BarChart data={chartData} layout="vertical" margin={{ left: 200 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" domain={[0, 100]} tickFormatter={v => `${v}%`} />
          <YAxis type="category" dataKey="question" width={190} tick={{ fontSize: 11 }} />
          <Tooltip formatter={(v: number) => `${v}%`} />
          <Bar dataKey="probability" radius={[0, 4, 4, 0]}>
            {chartData.map((entry, i) => (
              <Cell key={i} fill={getColor(entry.value)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default MarketOverview;
