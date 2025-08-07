import React, { useState, useEffect } from 'react';
import './ModelImages.css';

const ModelImages = () => {
  const [visualizations, setVisualizations] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);

  useEffect(() => {
    fetchVisualizations();
  }, []);

  const fetchVisualizations = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://127.0.0.1:8000/model/visualizations');
      const data = await response.json();
      
      if (data.success) {
        setVisualizations(data.visualizations);
      } else {
        setError(data.error || 'Failed to load visualizations');
      }
    } catch (err) {
      setError('Unable to connect to server');
    } finally {
      setLoading(false);
    }
  };

  const refreshVisualizations = () => {
    fetchVisualizations();
  };

  const openImageModal = (imageKey, imageData) => {
    setSelectedImage({ key: imageKey, data: imageData });
  };

  const closeImageModal = () => {
    setSelectedImage(null);
  };

  const downloadImage = (imageKey, imageData) => {
    const link = document.createElement('a');
    link.href = imageData;
    link.download = `model_${imageKey}_visualization.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const visualizationTitles = {
    confusion_matrix: {
      title: '🎯 Confusion Matrix',
      description: 'Shows how well the model distinguishes between Normal and Pneumonia cases'
    },
    roc_curve: {
      title: '📈 ROC Curve',
      description: 'Receiver Operating Characteristic curve showing classification performance'
    },
    performance_metrics: {
      title: '⭐ Performance Radar',
      description: 'Radar chart displaying all performance metrics in one view'
    },
    class_distribution: {
      title: '📊 Class Distribution',
      description: 'Distribution of Normal vs Pneumonia cases in the dataset'
    },
    training_history: {
      title: '📉 Training History',
      description: 'Model accuracy and loss progression during training epochs'
    },
    model_architecture: {
      title: '🏗️ Model Architecture',
      description: 'Detailed breakdown of model structure and parameters'
    },
    prediction_confidence: {
      title: '🎲 Prediction Confidence',
      description: 'Analysis of model confidence across different predictions'
    }
  };

  if (loading) {
    return (
      <div className="model-images-container">
        <div className="loading-spinner">
          <div className="spinner"></div>
          <h3>🎨 Generating Model Visualizations...</h3>
          <p>Creating statistical plots and performance matrices</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="model-images-container">
        <div className="error-state">
          <h3>❌ Visualization Error</h3>
          <p>{error}</p>
          <button onClick={refreshVisualizations} className="retry-button">
            🔄 Retry Loading
          </button>
        </div>
      </div>
    );
  }

  if (!visualizations || Object.keys(visualizations).length === 0) {
    return (
      <div className="model-images-container">
        <div className="no-data">
          <h3>🎨 No Visualizations Available</h3>
          <p>Model visualizations could not be generated</p>
          <button onClick={refreshVisualizations} className="retry-button">
            🔄 Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="model-images-container">
      <div className="images-header">
        <h2>🎨 Model Visualization Gallery</h2>
        <p>Statistical plots, matrices, and performance visualizations of your pneumonia detection AI</p>
        <div className="header-actions">
          <button onClick={refreshVisualizations} className="refresh-button">
            🔄 Refresh
          </button>
          <span className="image-count">
            {Object.keys(visualizations).length} visualizations available
          </span>
        </div>
      </div>

      <div className="visualization-grid">
        {Object.entries(visualizations).map(([key, imageData]) => {
          const vizInfo = visualizationTitles[key] || { 
            title: key.replace('_', ' ').toUpperCase(), 
            description: 'Model visualization chart' 
          };
          
          return (
            <div key={key} className="visualization-card">
              <div className="card-header">
                <h3>{vizInfo.title}</h3>
                <p>{vizInfo.description}</p>
              </div>
              
              <div className="image-container">
                <img 
                  src={imageData} 
                  alt={vizInfo.title}
                  onClick={() => openImageModal(key, imageData)}
                  loading="lazy"
                />
                <div className="image-overlay">
                  <button 
                    className="overlay-button view-button"
                    onClick={() => openImageModal(key, imageData)}
                    title="View Full Size"
                  >
                    🔍 View
                  </button>
                  <button 
                    className="overlay-button download-button"
                    onClick={() => downloadImage(key, imageData)}
                    title="Download Image"
                  >
                    💾 Save
                  </button>
                </div>
              </div>
              
              <div className="card-footer">
                <span className="image-type">PNG Image</span>
                <span className="image-quality">High Resolution</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Image Modal */}
      {selectedImage && (
        <div className="image-modal" onClick={closeImageModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{visualizationTitles[selectedImage.key]?.title || selectedImage.key}</h3>
              <button className="close-button" onClick={closeImageModal}>
                ✕
              </button>
            </div>
            <div className="modal-body">
              <img 
                src={selectedImage.data} 
                alt={selectedImage.key}
                className="modal-image"
              />
            </div>
            <div className="modal-footer">
              <button 
                className="modal-download-button"
                onClick={() => downloadImage(selectedImage.key, selectedImage.data)}
              >
                💾 Download Full Resolution
              </button>
              <p>{visualizationTitles[selectedImage.key]?.description}</p>
            </div>
          </div>
        </div>
      )}

      {/* Statistics Summary */}
      <div className="visualization-summary">
        <h3>📋 Visualization Summary</h3>
        <div className="summary-grid">
          <div className="summary-card">
            <div className="summary-icon">🎯</div>
            <div className="summary-content">
              <h4>Performance Analysis</h4>
              <p>Confusion matrix and ROC curves show model classification accuracy</p>
            </div>
          </div>
          
          <div className="summary-card">
            <div className="summary-icon">📊</div>
            <div className="summary-content">
              <h4>Data Insights</h4>
              <p>Class distribution reveals dataset characteristics and potential biases</p>
            </div>
          </div>
          
          <div className="summary-card">
            <div className="summary-icon">🏗️</div>
            <div className="summary-content">
              <h4>Model Structure</h4>
              <p>Architecture breakdown shows complexity and parameter distribution</p>
            </div>
          </div>
          
          <div className="summary-card">
            <div className="summary-icon">📈</div>
            <div className="summary-content">
              <h4>Training Progress</h4>
              <p>Historical plots track learning curves and optimization progress</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelImages;
