import React, { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, Cell,
} from 'recharts';
import { fetchTimeHorizonAnalysis } from '../api/client';
import { TimeHorizonResult } from '../types';

const HORIZON_COLORS = ['#22c55e', '#3b82f6', '#eab308', '#f97316', '#ef4444'];

const TimeHorizonComparison: React.FC = () => {
  const [data, setData] = useState<TimeHorizonResult[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchTimeHorizonAnalysis()
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-4">Loading time horizon analysis...</div>;
  if (!data.length) return <div className="p-4">No time horizon data available</div>;

  const chartData = data.map(d => ({
    horizon: d.horizon,
    brier_score: +d.brier_score.toFixed(4),
    samples: d.n_samples,
    base_rate: +(d.base_rate * 100).toFixed(1),
    mean_pred: +(d.mean_prediction * 100).toFixed(1),
  }));

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-bold mb-2">Time Horizon Comparison</h2>
      <p className="text-sm text-gray-500 mb-4">
        Brier Score by days-to-resolution. Predictions closer to resolution are typically more accurate.
      </p>

      <ResponsiveContainer width="100%" height={350}>
        <BarChart data={chartData} margin={{ bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="horizon" />
          <YAxis label={{ value: 'Brier Score', angle: -90, position: 'left' }} />
          <Tooltip
            content={({ payload }) => {
              if (!payload?.length) return null;
              const d = payload[0].payload;
              return (
                <div className="bg-white p-2 border rounded shadow text-sm">
                  <p className="font-semibold">{d.horizon}</p>
                  <p>Brier Score: {d.brier_score}</p>
                  <p>Samples: {d.samples}</p>
                  <p>Base Rate: {d.base_rate}%</p>
                  <p>Mean Prediction: {d.mean_pred}%</p>
                </div>
              );
            }}
          />
          <Bar dataKey="brier_score" name="Brier Score" radius={[4, 4, 0, 0]}>
            {chartData.map((_, i) => (
              <Cell key={i} fill={HORIZON_COLORS[i % HORIZON_COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <p className="text-xs text-gray-400 mt-2">
        Lower Brier Score = better. Expect degradation as time horizon increases.
      </p>
    </div>
  );
};

export default TimeHorizonComparison;
