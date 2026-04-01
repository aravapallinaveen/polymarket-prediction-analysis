import React, { useState } from 'react';
import Layout from './components/Layout';
import MarketOverview from './components/MarketOverview';
import CalibrationCurve from './components/CalibrationCurve';
import BrierScoreTracker from './components/BrierScoreTracker';
import BiasAnalysis from './components/BiasAnalysis';
import SentimentTrends from './components/SentimentTrends';
import FeatureImportance from './components/FeatureImportance';
import TimeHorizonComparison from './components/TimeHorizonComparison';
import MarketEfficiencyHeatmap from './components/MarketEfficiencyHeatmap';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');

  const renderContent = () => {
    switch (activeTab) {
      case 'overview':
        return <MarketOverview />;
      case 'calibration':
        return <CalibrationCurve />;
      case 'brier':
        return <BrierScoreTracker />;
      case 'bias':
        return <BiasAnalysis />;
      case 'sentiment':
        return <SentimentTrends />;
      case 'features':
        return <FeatureImportance />;
      case 'horizon':
        return <TimeHorizonComparison />;
      case 'heatmap':
        return <MarketEfficiencyHeatmap />;
      default:
        return <MarketOverview />;
    }
  };

  return (
    <Layout activeTab={activeTab} onTabChange={setActiveTab}>
      {renderContent()}
    </Layout>
  );
};

export default App;
