import React, { useEffect, useState } from 'react';
import {
  ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, ReferenceLine, Cell,
} from 'recharts';
import { fetchFavoriteLongshotBias } from '../api/client';
import { BiasPoint } from '../types';

const BiasAnalysis: React.FC = () => {
  const [data, setData] = useState<BiasPoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchFavoriteLongshotBias()
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-4">Loading bias analysis...</div>;
  if (!data.length) return <div className="p-4">No bias data available</div>;

  const chartData = data.map(d => ({
    bin: `${(d.bin_center * 100).toFixed(0)}%`,
    predicted: +(d.predicted_prob * 100).toFixed(1),
    actual: +(d.actual_freq * 100).toFixed(1),
    bias: +(d.bias * 100).toFixed(2),
    samples: d.n_samples,
    direction: d.bias_direction,
  }));

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-bold mb-2">Favorite-Longshot Bias Analysis</h2>
      <p className="text-sm text-gray-500 mb-4">
        Compares predicted probabilities against actual outcomes to detect systematic over/underconfidence.
        Markets priced below 20% (longshots) tend to be overpriced; markets above 80% (favorites) tend to be underpriced.
      </p>

      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart data={chartData} margin={{ top: 10, right: 30, bottom: 30, left: 30 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="bin"
            label={{ value: 'Predicted Probability Bin', position: 'bottom' }}
          />
          <YAxis
            label={{ value: 'Frequency (%)', angle: -90, position: 'left' }}
          />
          <Tooltip
            content={({ payload }) => {
              if (!payload?.length) return null;
              const d = payload[0].payload;
              return (
                <div className="bg-white p-2 border rounded shadow text-sm">
                  <p>Predicted: {d.predicted}%</p>
                  <p>Actual: {d.actual}%</p>
                  <p>Bias: {d.bias > 0 ? '+' : ''}{d.bias}%</p>
                  <p>Direction: {d.direction}</p>
                  <p>Samples: {d.samples}</p>
                </div>
              );
            }}
          />
          <Legend />
          <ReferenceLine y={0} stroke="#666" strokeDasharray="3 3" />
          <Bar name="Bias (Predicted - Actual)" dataKey="bias" opacity={0.7}>
            {chartData.map((entry, i) => (
              <Cell key={i} fill={entry.bias > 0 ? '#ef4444' : '#22c55e'} />
            ))}
          </Bar>
          <Line name="Predicted" dataKey="predicted" stroke="#3b82f6" strokeWidth={2} dot />
          <Line name="Actual" dataKey="actual" stroke="#f97316" strokeWidth={2} dot />
        </ComposedChart>
      </ResponsiveContainer>

      <div className="flex gap-4 mt-3 text-xs text-gray-500">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 bg-red-500 rounded" /> Overconfident (predicted &gt; actual)
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 bg-green-500 rounded" /> Underconfident (predicted &lt; actual)
        </span>
      </div>
    </div>
  );
};

export default BiasAnalysis;
