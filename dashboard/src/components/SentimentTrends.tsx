import React, { useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, Area, ComposedChart, Bar,
} from 'recharts';

// Demo data — in production this comes from the API
const DEMO_DATA = Array.from({ length: 60 }, (_, i) => {
  const date = new Date(2024, 0, 1 + i);
  const basePrice = 0.5 + 0.2 * Math.sin(i / 15) + (Math.random() - 0.5) * 0.05;
  return {
    date: date.toISOString().slice(0, 10),
    vader_mean: 0.1 * Math.sin(i / 10) + (Math.random() - 0.5) * 0.3,
    bert_mean: 0.5 + 0.15 * Math.sin(i / 12) + (Math.random() - 0.5) * 0.1,
    mention_count: Math.floor(20 + 30 * Math.random() + 15 * Math.sin(i / 8)),
    price: Math.max(0, Math.min(1, basePrice)),
  };
});

const SentimentTrends: React.FC = () => {
  const [data] = useState(DEMO_DATA);

  const chartData = data.map(d => ({
    ...d,
    vader_pct: +((d.vader_mean + 1) * 50).toFixed(1), // scale -1..1 to 0..100
    bert_pct: +(d.bert_mean * 100).toFixed(1),
    price_pct: +(d.price * 100).toFixed(1),
  }));

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-bold mb-2">Sentiment Trend Analysis</h2>
      <p className="text-sm text-gray-500 mb-4">
        Overlay of VADER/BERT sentiment scores, mention volume, and market price over time.
        Divergences between sentiment and price signal potential mispricings.
      </p>

      <ResponsiveContainer width="100%" height={350}>
        <ComposedChart data={chartData} margin={{ top: 10, right: 30, bottom: 10, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" tick={{ fontSize: 10 }} interval={7} />
          <YAxis yAxisId="pct" domain={[0, 100]} label={{ value: '%', position: 'top' }} />
          <YAxis yAxisId="mentions" orientation="right" label={{ value: 'Mentions', position: 'top' }} />
          <Tooltip />
          <Legend />
          <Area yAxisId="mentions" type="monotone" dataKey="mention_count"
            name="Mention Volume" fill="#e0e7ff" stroke="#818cf8" fillOpacity={0.3} />
          <Line yAxisId="pct" type="monotone" dataKey="price_pct"
            name="Market Price" stroke="#3b82f6" strokeWidth={2} dot={false} />
          <Line yAxisId="pct" type="monotone" dataKey="vader_pct"
            name="VADER Sentiment" stroke="#22c55e" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
          <Line yAxisId="pct" type="monotone" dataKey="bert_pct"
            name="BERT Sentiment" stroke="#f97316" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
        </ComposedChart>
      </ResponsiveContainer>

      <p className="text-xs text-gray-400 mt-2">
        When sentiment diverges significantly from price, the market may be mispriced.
      </p>
    </div>
  );
};

export default SentimentTrends;
