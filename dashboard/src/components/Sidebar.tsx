import React from 'react';

interface SidebarProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

const TABS = [
  { id: 'overview', label: 'Market Overview', icon: '|||' },
  { id: 'calibration', label: 'Calibration Curves', icon: '~' },
  { id: 'brier', label: 'Brier Score Tracker', icon: '#' },
  { id: 'bias', label: 'Bias Diagnostics', icon: '!' },
  { id: 'sentiment', label: 'Sentiment Trends', icon: '@' },
  { id: 'features', label: 'Feature Importance', icon: '*' },
  { id: 'horizon', label: 'Time Horizons', icon: '>' },
  { id: 'heatmap', label: 'Efficiency Heatmap', icon: '%' },
];

const Sidebar: React.FC<SidebarProps> = ({ activeTab, onTabChange }) => {
  return (
    <div className="w-64 bg-gray-900 text-white flex flex-col">
      <div className="p-4 border-b border-gray-700">
        <h1 className="text-lg font-bold">Polymarket</h1>
        <p className="text-xs text-gray-400">Prediction Accuracy Analysis</p>
      </div>
      <nav className="flex-1 py-4">
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`w-full text-left px-4 py-3 text-sm flex items-center gap-3 transition-colors ${
              activeTab === tab.id
                ? 'bg-blue-600 text-white'
                : 'text-gray-300 hover:bg-gray-800 hover:text-white'
            }`}
          >
            <span className="w-5 text-center font-mono text-xs">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </nav>
      <div className="p-4 border-t border-gray-700 text-xs text-gray-500">
        XGBoost + 53 Features
        <br />
        300+ Resolved Markets
      </div>
    </div>
  );
};

export default Sidebar;
