import React, { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts';
import { fetchFeatureImportance } from '../api/client';

interface FeatureItem {
  feature_name: string;
  importance_score: number;
  importance_rank: number;
}

const CATEGORY_COLORS: Record<string, string> = {
  market: '#3b82f6',
  sentiment: '#22c55e',
  trend: '#f97316',
  interaction: '#8b5cf6',
};

const getCategory = (name: string): string => {
  if (name.startsWith('sentiment') || name.startsWith('mention') || name.startsWith('positive'))
    return 'sentiment';
  if (name.startsWith('trend')) return 'trend';
  if (['sentiment_x_volume', 'trend_x_momentum', 'sentiment_price_divergence',
    'divergence_flag', 'smart_money_indicator', 'attention_adjusted_price',
    'consensus_strength', 'contrarian_signal'].includes(name))
    return 'interaction';
  return 'market';
};

const FeatureImportance: React.FC = () => {
  const [data, setData] = useState<FeatureItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchFeatureImportance()
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-4">Loading feature importance...</div>;

  const chartData = data.slice(0, 20).map(d => ({
    feature: d.feature_name,
    importance: +d.importance_score.toFixed(4),
    category: getCategory(d.feature_name),
  }));

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-bold mb-2">Feature Importance (Top 20)</h2>
      <p className="text-sm text-gray-500 mb-4">
        XGBoost gain-based importance. Colors indicate feature category.
      </p>

      <div className="flex gap-4 mb-3 text-xs">
        {Object.entries(CATEGORY_COLORS).map(([cat, color]) => (
          <span key={cat} className="flex items-center gap-1">
            <span className="w-3 h-3 rounded" style={{ backgroundColor: color }} />
            {cat.charAt(0).toUpperCase() + cat.slice(1)}
          </span>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={500}>
        <BarChart data={chartData} layout="vertical" margin={{ left: 180 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis type="category" dataKey="feature" width={170} tick={{ fontSize: 11 }} />
          <Tooltip />
          <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
            {chartData.map((entry, i) => (
              <Cell key={i} fill={CATEGORY_COLORS[entry.category] || '#94a3b8'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default FeatureImportance;
