import React, { useState, useEffect } from 'react';
import './Statistics.css';

const Statistics = ({ onStatisticsLoaded }) => {
  const [statistics, setStatistics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchStatistics();
  }, []);

  const fetchStatistics = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://127.0.0.1:8000/model/statistics');
      const data = await response.json();
      
      if (data.success) {
        setStatistics(data.statistics);
        if (onStatisticsLoaded) {
          onStatisticsLoaded(data.statistics);
        }
      } else {
        setError(data.error || 'Failed to load statistics');
      }
    } catch (err) {
      setError('Unable to connect to server');
    } finally {
      setLoading(false);
    }
  };

  const formatNumber = (num) => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toLocaleString();
  };

  const formatPercentage = (num) => {
    return (num * 100).toFixed(1) + '%';
  };

  if (loading) {
    return (
      <div className="statistics-container">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <p>Loading model statistics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="statistics-container">
        <div className="error-message">
          <h3>❌ Unable to Load Statistics</h3>
          <p>{error}</p>
          <button onClick={fetchStatistics} className="retry-button">
            🔄 Retry
          </button>
        </div>
      </div>
    );
  }

  if (!statistics) {
    return (
      <div className="statistics-container">
        <div className="no-data">
          <h3>📊 No Statistics Available</h3>
          <p>Train the model first to see performance statistics</p>
        </div>
      </div>
    );
  }

  const {
    model_info,
    dataset_statistics,
    training_configuration,
    model_parameters,
    training_metrics,
    performance_rating,
    recommendations
  } = statistics;

  return (
    <div className="statistics-container">
      <div className="statistics-header">
        <h2>📊 Model Performance Dashboard</h2>
        <p>Comprehensive analysis of your pneumonia detection AI model</p>
      </div>

      {/* Performance Overview */}
      <div className="stats-grid">
        <div className="stat-card performance-overview">
          <div className="card-header">
            <h3>🎯 Performance Rating</h3>
          </div>
          <div className="performance-rating">
            <div className={`rating-badge ${performance_rating?.color || 'gray'}`}>
              {performance_rating?.overall || 'Unknown'}
            </div>
            <div className="rating-score">
              {performance_rating?.score ? formatPercentage(performance_rating.score) : 'N/A'}
            </div>
          </div>
          {performance_rating?.details && (
            <div className="performance-details">
              <div className="metric">
                <span>Accuracy:</span>
                <span>{formatPercentage(performance_rating.details.accuracy)}</span>
              </div>
              <div className="metric">
                <span>Precision:</span>
                <span>{formatPercentage(performance_rating.details.precision)}</span>
              </div>
              <div className="metric">
                <span>Recall:</span>
                <span>{formatPercentage(performance_rating.details.recall)}</span>
              </div>
              <div className="metric">
                <span>F1-Score:</span>
                <span>{formatPercentage(performance_rating.f1_score)}</span>
              </div>
            </div>
          )}
        </div>

        {/* Model Architecture */}
        <div className="stat-card">
          <div className="card-header">
            <h3>🏗️ Model Architecture</h3>
          </div>
          <div className="architecture-info">
            <div className="metric">
              <span>Type:</span>
              <span>{training_configuration?.architecture || 'Unknown'}</span>
            </div>
            <div className="metric">
              <span>Total Parameters:</span>
              <span>{model_parameters?.total_parameters ? formatNumber(model_parameters.total_parameters) : 'N/A'}</span>
            </div>
            <div className="metric">
              <span>Trainable:</span>
              <span>{model_parameters?.trainable_parameters ? formatNumber(model_parameters.trainable_parameters) : 'N/A'}</span>
            </div>
            <div className="metric">
              <span>Model Size:</span>
              <span>{model_parameters?.model_size_mb ? `${model_parameters.model_size_mb} MB` : 'N/A'}</span>
            </div>
            <div className="metric">
              <span>Layers:</span>
              <span>{model_parameters?.layer_count || 'N/A'}</span>
            </div>
          </div>
        </div>

        {/* Training Metrics */}
        <div className="stat-card">
          <div className="card-header">
            <h3>📈 Training Metrics</h3>
          </div>
          <div className="training-info">
            <div className="metric">
              <span>Epochs Trained:</span>
              <span>{training_configuration?.epochs_trained || 'N/A'}</span>
            </div>
            <div className="metric">
              <span>Final Accuracy:</span>
              <span>{training_metrics?.final_val_accuracy ? formatPercentage(training_metrics.final_val_accuracy) : 'N/A'}</span>
            </div>
            <div className="metric">
              <span>Best Accuracy:</span>
              <span>{training_metrics?.best_val_accuracy ? formatPercentage(training_metrics.best_val_accuracy) : 'N/A'}</span>
            </div>
            <div className="metric">
              <span>Final Loss:</span>
              <span>{training_metrics?.final_val_loss ? training_metrics.final_val_loss.toFixed(4) : 'N/A'}</span>
            </div>
            <div className="metric">
              <span>Improvement:</span>
              <span className={training_metrics?.accuracy_improvement > 0 ? 'positive' : 'negative'}>
                {training_metrics?.accuracy_improvement ? `${(training_metrics.accuracy_improvement * 100).toFixed(1)}%` : 'N/A'}
              </span>
            </div>
          </div>
        </div>

        {/* Dataset Statistics */}
        <div className="stat-card">
          <div className="card-header">
            <h3>📊 Dataset Statistics</h3>
          </div>
          <div className="dataset-info">
            <div className="metric">
              <span>Total Images:</span>
              <span>{dataset_statistics?.total_images ? formatNumber(dataset_statistics.total_images) : 'N/A'}</span>
            </div>
            <div className="metric">
              <span>Image Size:</span>
              <span>{dataset_statistics?.image_size ? `${dataset_statistics.image_size[0]}×${dataset_statistics.image_size[1]}` : 'N/A'}</span>
            </div>
            <div className="metric">
              <span>Batch Size:</span>
              <span>{dataset_statistics?.batch_size || 'N/A'}</span>
            </div>
            {dataset_statistics?.class_balance && (
              <div className="class-balance">
                <div className="balance-item">
                  <span>Normal:</span>
                  <span>{dataset_statistics.class_balance.NORMAL}%</span>
                </div>
                <div className="balance-item">
                  <span>Pneumonia:</span>
                  <span>{dataset_statistics.class_balance.PNEUMONIA}%</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Recommendations */}
      {recommendations && recommendations.length > 0 && (
        <div className="recommendations-section">
          <h3>💡 Improvement Recommendations</h3>
          <div className="recommendations-list">
            {recommendations.map((rec, index) => (
              <div key={index} className="recommendation-item">
                <span className="recommendation-icon">•</span>
                <span className="recommendation-text">{rec}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Technical Details */}
      <div className="technical-section">
        <h3>🔧 Technical Configuration</h3>
        <div className="technical-grid">
          <div className="tech-item">
            <span>Optimizer:</span>
            <span>{training_configuration?.optimizer || 'Adam'}</span>
          </div>
          <div className="tech-item">
            <span>Loss Function:</span>
            <span>{training_configuration?.loss_function || 'Binary Crossentropy'}</span>
          </div>
          <div className="tech-item">
            <span>Metrics:</span>
            <span>{training_configuration?.metrics?.join(', ') || 'Accuracy, Precision, Recall'}</span>
          </div>
          <div className="tech-item">
            <span>Model Version:</span>
            <span>{model_info?.version || '1.0.0'}</span>
          </div>
        </div>
      </div>

      <div className="refresh-section">
        <button onClick={fetchStatistics} className="refresh-button">
          🔄 Refresh Statistics
        </button>
        <p className="last-updated">
          Last updated: {new Date().toLocaleString()}
        </p>
      </div>
    </div>
  );
};

export default Statistics;
