import React, { useEffect, useState } from 'react';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts';
import { fetchCalibrationData } from '../api/client';

interface CalibrationData {
  fraction_of_positives: number[];
  mean_predicted_value: number[];
  bin_counts: number[];
  ece: number;
  mce: number;
}

const CalibrationCurve: React.FC = () => {
  const [data, setData] = useState<CalibrationData | null>(null);
  const [model, setModel] = useState('xgboost');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetchCalibrationData(model)
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [model]);

  if (loading) return <div className="p-4">Loading calibration data...</div>;
  if (!data) return <div className="p-4">No calibration data available</div>;

  const chartData = data.mean_predicted_value.map((pred, i) => ({
    predicted: +(pred * 100).toFixed(1),
    actual: +(data.fraction_of_positives[i] * 100).toFixed(1),
    count: data.bin_counts[i],
  }));

  const perfectLine = Array.from({ length: 11 }, (_, i) => ({
    predicted: i * 10,
    actual: i * 10,
  }));

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold">Calibration Curve (Reliability Diagram)</h2>
        <select
          value={model}
          onChange={e => setModel(e.target.value)}
          className="border rounded px-2 py-1 text-sm"
        >
          <option value="xgboost">XGBoost</option>
          <option value="logistic">Logistic Regression</option>
        </select>
      </div>

      <div className="flex gap-4 mb-4 text-sm">
        <div className="bg-blue-50 px-3 py-1 rounded">
          ECE: <strong>{(data.ece * 100).toFixed(2)}%</strong>
        </div>
        <div className="bg-red-50 px-3 py-1 rounded">
          MCE: <strong>{(data.mce * 100).toFixed(2)}%</strong>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart margin={{ top: 10, right: 30, bottom: 30, left: 30 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="predicted"
            domain={[0, 100]}
            name="Predicted"
            label={{ value: 'Mean Predicted Probability (%)', position: 'bottom' }}
          />
          <YAxis
            type="number"
            dataKey="actual"
            domain={[0, 100]}
            name="Actual"
            label={{ value: 'Actual Frequency (%)', angle: -90, position: 'left' }}
          />
          <Tooltip
            formatter={(value: number, name: string) => [`${value}%`, name]}
            content={({ payload }) => {
              if (!payload?.length) return null;
              const d = payload[0].payload;
              return (
                <div className="bg-white p-2 border rounded shadow text-sm">
                  <p>Predicted: {d.predicted}%</p>
                  <p>Actual: {d.actual}%</p>
                  <p>Samples: {d.count}</p>
                </div>
              );
            }}
          />
          <Legend />
          <ReferenceLine
            segment={[{ x: 0, y: 0 }, { x: 100, y: 100 }]}
            stroke="#94a3b8"
            strokeDasharray="5 5"
            label="Perfect"
          />
          <Scatter name="Model Calibration" data={chartData} fill="#3b82f6" r={6} />
        </ScatterChart>
      </ResponsiveContainer>
      <p className="text-xs text-gray-400 mt-2">
        Points on the diagonal = perfectly calibrated. Below = overconfident. Above = underconfident.
      </p>
    </div>
  );
};

export default CalibrationCurve;
