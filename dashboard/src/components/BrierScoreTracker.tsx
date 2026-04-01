import React, { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, PieChart, Pie, Cell,
} from 'recharts';
import { fetchBrierScores } from '../api/client';
import { BrierDecomposition } from '../types';

const COLORS = ['#ef4444', '#22c55e', '#3b82f6'];

const BrierScoreTracker: React.FC = () => {
  const [data, setData] = useState<BrierDecomposition | null>(null);
  const [model, setModel] = useState('xgboost');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetchBrierScores(model)
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [model]);

  if (loading) return <div className="p-4">Loading Brier scores...</div>;
  if (!data) return <div className="p-4">No Brier score data available</div>;

  const decompositionData = [
    { name: 'Reliability', value: data.reliability, description: 'Lower = better calibrated' },
    { name: 'Resolution', value: data.resolution, description: 'Higher = better separation' },
    { name: 'Uncertainty', value: data.uncertainty, description: 'Base rate entropy (fixed)' },
  ];

  const summaryCards = [
    { label: 'Brier Score', value: data.brier_score.toFixed(4), color: 'bg-blue-50' },
    { label: 'Skill Score', value: (data.skill_score * 100).toFixed(1) + '%', color: 'bg-green-50' },
    { label: 'Reliability', value: data.reliability.toFixed(4), color: 'bg-red-50' },
    { label: 'Resolution', value: data.resolution.toFixed(4), color: 'bg-purple-50' },
  ];

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold">Brier Score Tracker</h2>
        <select
          value={model}
          onChange={e => setModel(e.target.value)}
          className="border rounded px-2 py-1 text-sm"
        >
          <option value="xgboost">XGBoost</option>
          <option value="logistic">Logistic Regression</option>
        </select>
      </div>

      <div className="grid grid-cols-4 gap-3 mb-6">
        {summaryCards.map(card => (
          <div key={card.label} className={`${card.color} rounded-lg p-3 text-center`}>
            <div className="text-xs text-gray-500">{card.label}</div>
            <div className="text-lg font-bold">{card.value}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <h3 className="text-sm font-semibold mb-2">Murphy Decomposition</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={decompositionData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis />
              <Tooltip
                content={({ payload }) => {
                  if (!payload?.length) return null;
                  const d = payload[0].payload;
                  return (
                    <div className="bg-white p-2 border rounded shadow text-sm">
                      <p className="font-semibold">{d.name}: {d.value.toFixed(4)}</p>
                      <p className="text-gray-500">{d.description}</p>
                    </div>
                  );
                }}
              />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {decompositionData.map((_, i) => (
                  <Cell key={i} fill={COLORS[i]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div>
          <h3 className="text-sm font-semibold mb-2">Component Breakdown</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={decompositionData}
                cx="50%"
                cy="50%"
                innerRadius={50}
                outerRadius={90}
                dataKey="value"
                nameKey="name"
                label={({ name, value }) => `${name}: ${value.toFixed(3)}`}
              >
                {decompositionData.map((_, i) => (
                  <Cell key={i} fill={COLORS[i]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      <p className="text-xs text-gray-400 mt-2">
        BS = Reliability - Resolution + Uncertainty. Skill Score = improvement over base-rate model.
      </p>
    </div>
  );
};

export default BrierScoreTracker;
