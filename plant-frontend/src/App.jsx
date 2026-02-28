// src/App.jsx
import { useState, useEffect } from 'react';
import axios from 'axios';
import FileUpload from './components/FileUpload';
import Loader from './components/Loader';
import Results from './components/Results';
import ExportButton from './components/ExportButton';
import { FiUpload, FiInfo } from 'react-icons/fi';
import { getPlantTypeFromFilename } from './utils/plantType';
import './App.css';

const API_URL = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/predict`
  : 'http://localhost:8000/predict';

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [previewUrls, setPreviewUrls] = useState([]);
  const [predictionsList, setPredictionsList] = useState([]);
  const [scale, setScale] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFilesSelect = (files) => {
    setSelectedFiles(files);
    setPredictionsList([]);
    setError('');

    const urls = files.map(file => URL.createObjectURL(file));
    setPreviewUrls(urls);
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) {
      alert('Выберите хотя бы одно изображение');
      return;
    }

    setLoading(true);
    setError('');
    setPredictionsList([]);

    const results = [];

    for (const file of selectedFiles) {
      const formData = new FormData();
      formData.append('file', file);

      if (scale && !isNaN(parseFloat(scale))) {
        formData.append('scale', parseFloat(scale));
      }

      try {
        const response = await axios.post(API_URL, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });

        results.push({
          fileName: file.name,
          predictions: response.data.predictions || []
        });
      } catch (err) {
        results.push({
          fileName: file.name,
          predictions: [],
          error: err.response?.data?.detail || err.message
        });
      }
    }

    setPredictionsList(results);
    setLoading(false);
  };

  // Очистка object URLs при размонтировании
  useEffect(() => {
    return () => {
      previewUrls.forEach(url => URL.revokeObjectURL(url));
    };
  }, [previewUrls]);

  return (
    <div className="app">
      <header className="header">
        <h1>🌱 PlantSeg</h1>
        <p className="subtitle">Анализ длины и площади корней, стеблей и листьев</p>
      </header>

      <main className="main">
        <div className="upload-card">
          <FileUpload onFilesSelect={handleFilesSelect} />

          {previewUrls.length > 0 && (
            <div className="preview-container" style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: '1rem',
              justifyContent: 'center',
              margin: '2rem 0'
            }}>
              {previewUrls.map((url, index) => (
                <div key={index} style={{ textAlign: 'center', maxWidth: '220px' }}>
                  <img
                    src={url}
                    alt={`preview-${index}`}
                    className="preview-image"
                    style={{ maxHeight: '200px', width: '100%', objectFit: 'contain' }}
                  />
                  <p className="file-name">{selectedFiles[index]?.name}</p>
                </div>
              ))}
            </div>
          )}

          {selectedFiles.length > 0 && (
            <div className="control-panel">
              <div className="scale-input">
                <label htmlFor="scale">Масштаб (пикселей на см):</label>
                <input
                  type="number"
                  id="scale"
                  value={scale}
                  onChange={(e) => setScale(e.target.value)}
                  placeholder="Например, 10.5"
                  step="0.1"
                  min="0"
                />
                <span className="scale-hint">
                  <FiInfo /> Если не указать — будет использовано значение по умолчанию (93.8 px/cm)
                </span>
              </div>

              <button
                className="analyze-button"
                onClick={handleUpload}
                disabled={loading}
              >
                {loading ? 'Анализ...' : (
                  <>
                    <FiUpload /> Анализировать все изображения
                  </>
                )}
              </button>
            </div>
          )}
        </div>

        {loading && <Loader />}

        {error && <div className="error-message">{error}</div>}

        {predictionsList.length > 0 && (
          <div className="results-container">
            <h2>Результаты анализа ({predictionsList.length} изображений)</h2>

            {predictionsList.map((item, index) => (
              <div key={index} style={{ marginBottom: '3rem' }}>
                <h3 style={{
                  color: '#1e293b',
                  marginBottom: '0.5rem',
                  borderBottom: '2px solid #e2e8f0',
                  paddingBottom: '0.75rem'
                }}>
                  📄 {item.fileName}
                </h3>

                <p style={{
                  fontSize: '1.05rem',
                  color: '#334155',
                  marginBottom: '1rem',
                  fontWeight: 500
                }}>
                  Тип растения: <strong>{getPlantTypeFromFilename(item.fileName)}</strong>
                </p>

                {item.error ? (
                  <div className="error-message">{item.error}</div>
                ) : item.predictions.length > 0 ? (
                  <Results predictions={item.predictions} />
                ) : (
                  <p className="no-results">На этом изображении ничего не обнаружено.</p>
                )}
              </div>
            ))}

            <ExportButton predictionsList={predictionsList} />
          </div>
        )}
      </main>

      <footer className="footer">
        <p>© 2026 PlantSeg — нейросетевая сегментация растений</p>
      </footer>
    </div>
  );
}

export default App;