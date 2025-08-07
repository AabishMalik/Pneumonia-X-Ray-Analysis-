import React, { useState, useCallback } from 'react';
import Statistics from './Statistics';
import ModelVisuals from './ModelVisuals';
import ModelImages from './ModelImages';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [currentView, setCurrentView] = useState('prediction'); // 'prediction', 'statistics', 'visuals', or 'images'
  const [statistics, setStatistics] = useState(null);

  const handleFileSelect = useCallback((file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setError(null);
      setPrediction(null);
    } else {
      setError('Please select a valid image file');
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
  }, [handleFileSelect]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleFileChange = useCallback((e) => {
    const file = e.target.files[0];
    handleFileSelect(file);
  }, [handleFileSelect]);

  const predictPneumonia = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setPrediction(result);
    } catch (err) {
      setError(`Prediction failed: ${err.message}`);
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const resetApp = () => {
    setSelectedFile(null);
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🫁 BWRP Pneumonia Detection</h1>
        <p>AI-powered chest X-ray analysis for pneumonia detection</p>
        <div className="sdg-badge">
          <span>🎯 SDG Goal 3: Good Health and Well-being</span>
        </div>
        
        {/* Navigation */}
        <nav className="app-navigation">
          <button 
            className={`nav-button ${currentView === 'prediction' ? 'active' : ''}`}
            onClick={() => setCurrentView('prediction')}
          >
            🔬 AI Prediction
          </button>
          <button 
            className={`nav-button ${currentView === 'statistics' ? 'active' : ''}`}
            onClick={() => setCurrentView('statistics')}
          >
            📊 Model Statistics
          </button>
          <button 
            className={`nav-button ${currentView === 'visuals' ? 'active' : ''}`}
            onClick={() => setCurrentView('visuals')}
          >
            📈 Data Visualizations
          </button>
          <button 
            className={`nav-button ${currentView === 'images' ? 'active' : ''}`}
            onClick={() => setCurrentView('images')}
          >
            🎨 Model Images
          </button>
        </nav>
      </header>

      <main className="App-main">
        {currentView === 'prediction' ? (
          <>
            <div className="upload-section">
          <div
            className={`upload-area ${dragOver ? 'drag-over' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            {selectedFile ? (
              <div className="file-preview">
                <img
                  src={URL.createObjectURL(selectedFile)}
                  alt="Selected chest X-ray"
                  className="preview-image"
                />
                <p className="file-info">
                  📁 {selectedFile.name}
                  <br />
                  📊 {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            ) : (
              <div className="upload-prompt">
                <div className="upload-icon">📤</div>
                <h3>Upload Chest X-ray Image</h3>
                <p>Drag and drop an image here, or click to select</p>
                <p className="supported-formats">
                  Supported: JPG, PNG, JPEG
                </p>
              </div>
            )}
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="file-input"
            />
          </div>

          <div className="action-buttons">
            {selectedFile && (
              <>
                <button
                  onClick={predictPneumonia}
                  disabled={loading}
                  className="predict-button"
                >
                  {loading ? '🔄 Analyzing...' : '🔬 Analyze X-ray'}
                </button>
                <button onClick={resetApp} className="reset-button">
                  🔄 Upload New Image
                </button>
              </>
            )}
          </div>
        </div>

        {error && (
          <div className="error-message">
            <h3>❌ Error</h3>
            <p>{error}</p>
            <div className="error-help">
              <p>💡 <strong>Troubleshooting:</strong></p>
              <ul>
                <li>Make sure the backend server is running on port 8000</li>
                <li>Check that you've uploaded a valid image file</li>
                <li>Ensure the image is a chest X-ray for best results</li>
              </ul>
            </div>
          </div>
        )}

        {prediction && (
          <div className="prediction-result">
            <h3>📋 Analysis Results</h3>
            <div className="result-summary">
              <div className={`diagnosis ${prediction.predicted_class.toLowerCase()}`}>
                <div className="diagnosis-icon">
                  {prediction.predicted_class === 'PNEUMONIA' ? '⚠️' : '✅'}
                </div>
                <div className="diagnosis-text">
                  <h4>{prediction.predicted_class}</h4>
                  <p>Confidence: {(prediction.confidence * 100).toFixed(1)}%</p>
                </div>
              </div>
            </div>

            <div className="detailed-results">
              <div className="probability-bars">
                <div className="probability-item">
                  <label>Normal Probability:</label>
                  <div className="progress-bar">
                    <div
                      className="progress-fill normal"
                      style={{ width: `${prediction.normal_probability * 100}%` }}
                    ></div>
                  </div>
                  <span>{(prediction.normal_probability * 100).toFixed(1)}%</span>
                </div>
                <div className="probability-item">
                  <label>Pneumonia Probability:</label>
                  <div className="progress-bar">
                    <div
                      className="progress-fill pneumonia"
                      style={{ width: `${prediction.pneumonia_probability * 100}%` }}
                    ></div>
                  </div>
                  <span>{(prediction.pneumonia_probability * 100).toFixed(1)}%</span>
                </div>
              </div>

              <div className="medical-disclaimer">
                <h4>⚕️ Important Medical Disclaimer</h4>
                <p>
                  This AI analysis is for educational and research purposes only.
                  It should not be used as a substitute for professional medical diagnosis.
                  Always consult with qualified healthcare professionals for medical decisions.
                </p>
              </div>
            </div>
          </div>
        )}

        <div className="info-section">
          <div className="feature-grid">
            <div className="feature-card">
              <div className="feature-icon">🤖</div>
              <h3>AI-Powered</h3>
              <p>Advanced deep learning model trained on thousands of chest X-rays</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">⚡</div>
              <h3>Fast Analysis</h3>
              <p>Get results in seconds with high accuracy detection</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🔒</div>
              <h3>Privacy First</h3>
              <p>Your images are processed locally and not stored</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🌍</div>
              <h3>Global Health</h3>
              <p>Supporting UN SDG 3: Good Health and Well-being</p>
            </div>
          </div>
        </div>
        </>
        ) : currentView === 'statistics' ? (
          <Statistics onStatisticsLoaded={setStatistics} />
        ) : currentView === 'visuals' ? (
          <ModelVisuals statistics={statistics} />
        ) : (
          <ModelImages />
        )}
      </main>

      <footer className="App-footer">
        <div className="tech-stack">
          <h4>🛠️ Technology Stack</h4>
          <div className="tech-badges">
            <span className="tech-badge">React</span>
            <span className="tech-badge">Python</span>
            <span className="tech-badge">TensorFlow</span>
            <span className="tech-badge">FastAPI</span>
            <span className="tech-badge">Kaggle</span>
          </div>
        </div>
        <p>© 2024 BWRP Team - Pneumonia Detection AI</p>
      </footer>
    </div>
  );
}

export default App;
