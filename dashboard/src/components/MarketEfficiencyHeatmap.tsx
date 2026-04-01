import React, { useEffect, useState } from 'react';
import { fetchCategoryBias } from '../api/client';
import { CategoryBias } from '../types';

const getColor = (brierScore: number): string => {
  if (brierScore < 0.1) return '#22c55e';
  if (brierScore < 0.15) return '#86efac';
  if (brierScore < 0.2) return '#fde047';
  if (brierScore < 0.25) return '#fb923c';
  return '#ef4444';
};

const getBiasColor = (bias: number): string => {
  const abs = Math.abs(bias);
  if (abs < 0.02) return '#22c55e';
  if (abs < 0.05) return '#fde047';
  return '#ef4444';
};

const MarketEfficiencyHeatmap: React.FC = () => {
  const [data, setData] = useState<CategoryBias[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchCategoryBias()
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-4">Loading efficiency heatmap...</div>;
  if (!data.length) return <div className="p-4">No category data available</div>;

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-bold mb-2">Market Efficiency Heatmap</h2>
      <p className="text-sm text-gray-500 mb-4">
        Prediction accuracy and bias by market category. Green = efficient, Red = inefficient.
      </p>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b">
              <th className="text-left py-2 px-3">Category</th>
              <th className="text-center py-2 px-3">Brier Score</th>
              <th className="text-center py-2 px-3">Mean Bias</th>
              <th className="text-center py-2 px-3">Base Rate</th>
              <th className="text-center py-2 px-3">Avg Prediction</th>
              <th className="text-center py-2 px-3">Samples</th>
              <th className="text-center py-2 px-3">Efficiency</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr key={i} className="border-b hover:bg-gray-50">
                <td className="py-2 px-3 font-medium">{row.category}</td>
                <td className="py-2 px-3 text-center">
                  <span
                    className="inline-block px-2 py-0.5 rounded text-white text-xs font-bold"
                    style={{ backgroundColor: getColor(row.brier_score) }}
                  >
                    {row.brier_score.toFixed(4)}
                  </span>
                </td>
                <td className="py-2 px-3 text-center">
                  <span
                    className="inline-block px-2 py-0.5 rounded text-xs font-bold"
                    style={{ backgroundColor: getBiasColor(row.mean_bias), color: '#fff' }}
                  >
                    {row.mean_bias > 0 ? '+' : ''}{(row.mean_bias * 100).toFixed(2)}%
                  </span>
                </td>
                <td className="py-2 px-3 text-center">{(row.base_rate * 100).toFixed(1)}%</td>
                <td className="py-2 px-3 text-center">{(row.mean_prediction * 100).toFixed(1)}%</td>
                <td className="py-2 px-3 text-center">{row.n_samples}</td>
                <td className="py-2 px-3 text-center">
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="h-2 rounded-full"
                      style={{
                        width: `${Math.max(5, (1 - row.brier_score / 0.3) * 100)}%`,
                        backgroundColor: getColor(row.brier_score),
                      }}
                    />
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex gap-4 mt-4 text-xs text-gray-500">
        <span>Brier Score Scale:</span>
        {[
          { label: '< 0.10', color: '#22c55e' },
          { label: '0.10-0.15', color: '#86efac' },
          { label: '0.15-0.20', color: '#fde047' },
          { label: '0.20-0.25', color: '#fb923c' },
          { label: '> 0.25', color: '#ef4444' },
        ].map(({ label, color }) => (
          <span key={label} className="flex items-center gap-1">
            <span className="w-3 h-3 rounded" style={{ backgroundColor: color }} />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
};

export default MarketEfficiencyHeatmap;
