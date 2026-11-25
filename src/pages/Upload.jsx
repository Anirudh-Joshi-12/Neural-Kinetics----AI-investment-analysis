// src/pages/Upload.jsx
import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { FileText, Loader2, CheckCircle, AlertCircle, X, Upload as UploadIcon, RotateCcw } from 'lucide-react';
import { api } from '../lib/api';
import { useNavigate } from 'react-router-dom';
import { EASE } from '../constants';

const Upload = () => {
  const [file, setFile] = useState(null);
  const [stage, setStage] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const navigate = useNavigate();
  const pollIntervalRef = useRef(null);

  const stages = [
    { id: 'upload', label: 'Uploading', icon: UploadIcon },
    { id: 'index', label: 'Indexing', icon: Loader2 },
    { id: 'extract', label: 'Extracting KPIs', icon: FileText },
    { id: 'score', label: 'Scoring', icon: Loader2 },
    { id: 'brief', label: 'Generating Brief', icon: FileText },
  ];

  // Map backend status -> frontend stage
  const statusToStage = {
    pending: 'upload',
    processing: 'index',
    completed: 'complete',
    failed: 'error',
  };

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  const isValidPdf = (f) => {
    if (!f) return false;
    const nameOk = f.name?.toLowerCase().endsWith('.pdf');
    const typeOk = f.type === 'application/pdf' || f.type === 'application/x-pdf' || f.type === 'application/octet-stream';
    return nameOk || typeOk;
  };

  const handleFileChange = (e) => {
    const f = e.target.files[0];
    if (isValidPdf(f)) {
      setFile(f);
      setError(null);
    } else {
      setError('Please select a valid PDF file');
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const f = e.dataTransfer.files[0];
    if (isValidPdf(f)) {
      setFile(f);
      setError(null);
    } else {
      setError('Please drop a valid PDF file');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    setStage('upload');
    setProgress(0);
    setError(null);
    setResult(null);

    try {
      const data = await api.uploadReport(file);
      const reportId = data.id;

      // Poll for status
      pollIntervalRef.current = setInterval(async () => {
        try {
          const statusData = await api.getReportStatus(reportId);
          setProgress(statusData.progress_pct);
          
          // Update stage based on backend status
          const newStage = statusToStage[statusData.status] || 'upload';
          setStage(newStage);
          
          if (statusData.status === 'completed') {
            clearInterval(pollIntervalRef.current);
            setStage('complete');
            setProgress(100);
            setTimeout(() => {
              navigate(`/analysis/${reportId}`);
            }, 1500);
          } else if (statusData.status === 'failed') {
            clearInterval(pollIntervalRef.current);
            setStage('error');
            setError(statusData.error_message || 'Analysis failed');
            setProgress(0);
          }
        } catch (err) {
          clearInterval(pollIntervalRef.current);
          setStage('error');
          setError(err.message);
          setProgress(0);
        }
      }, 1000);

    } catch (err) {
      setStage('error');
      setError(err.message);
      setProgress(0);
    }
  };

  const handleRetry = () => {
    setStage('idle');
    setError(null);
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="page upload-page">
      <motion.div
        className="upload-container"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
      >
        <motion.div
          className="upload-header"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
        >
          <motion.span
            className="page-badge"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
          >
            <span className="badge-dot" />
            Upload Annual Report
          </motion.span>

          <motion.h1
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
          >
            Upload an Annual Report for AI Analysis
          </motion.h1>

          <motion.p
            className="page-subtitle"
            initial={{ y: 16, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
          >
            Upload a company's annual report (10-K) PDF. Our pipeline will extract 5 key financial KPIs,
            score the investment with our ML model, and generate an investment brief with SHAP explainability.
          </motion.p>
        </motion.div>

        <motion.div
          className="upload-zone"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4, ease: [0.16, 1, 0.3, 1] }}
        >
          {(stage === 'idle' || stage === 'error') && (
            <div
              className={`drop-zone ${file ? 'has-file' : ''}`}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              <input
                type="file"
                accept="application/pdf"
                onChange={handleFileChange}
                id="file-upload"
                className="file-input"
              />
              <label htmlFor="file-upload" className="drop-zone-label">
                <motion.div
                  className="drop-icon"
                  animate={{ y: [0, -8, 0] }}
                  transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
                >
                  <UploadIcon size={48} />
                </motion.div>
                <h3>{file ? 'File Selected' : 'Drop PDF Here or Click to Browse'}</h3>
                <p>{file ? `${file.name} • ${(file.size / (1024 * 1024)).toFixed(2)} MB` : 'PDF only • Max 50MB'}</p>
                {file && (
                  <motion.button
                    className="btn-secondary remove-file"
                    onClick={() => setFile(null)}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <X size={16} /> Remove
                  </motion.button>
                )}
              </label>
            </div>
          )}

          {file && (
            <motion.button
              className="btn-primary upload-btn"
              onClick={handleSubmit}
              disabled={stage !== 'idle' && stage !== 'error'}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <UploadIcon size={18} />
              {stage === 'error' ? 'Retry Analysis' : 'Start Analysis'}
            </motion.button>
          )}

        </motion.div>

        <AnimatePresence mode="wait">
          {stage !== 'idle' && (
            <motion.div
              className="progress-section"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.5, ease: [0.16, 1, 0.3, 1] }}
            >
              <div className="progress-header">
                <h3>{stage === 'error' ? 'Analysis Failed' : stage === 'complete' ? 'Analysis Complete' : 'Processing Report'}</h3>
                <span className="progress-percent">{Math.round(progress)}%</span>
              </div>
              <div className="progress-bar">
                <motion.div
                  className="progress-fill"
                  initial={{ width: 0 }}
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
                />
              </div>

              <div className="stage-indicators">
                {stages.map((s, i) => (
                  <motion.div
                    key={s.id}
                    className={`stage-indicator ${i < stages.findIndex(st => st.id === stage) ? 'completed' : stage === s.id ? 'active' : 'pending'}`}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.4, delay: i * 0.1 }}
                  >
                    <div className="stage-icon">
                      <s.icon size={16} />
                    </div>
                    <span className="stage-label">{s.label}</span>
                    <motion.div
                      className="stage-dot"
                      animate={{ scale: stage === s.id ? [1, 1.2, 1] : 1 }}
                      transition={{ duration: 0.8, repeat: stage === s.id ? Infinity : 0 }}
                    />
                  </motion.div>
                ))}
              </div>

              {stage === 'error' && (
                <motion.div
                  className="error-message"
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <AlertCircle size={20} />
                  <span>{error}</span>
                  <motion.button
                    className="btn-secondary"
                    onClick={handleRetry}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <RotateCcw size={16} /> Retry
                  </motion.button>
                </motion.div>
              )}

              {stage === 'complete' && (
                <motion.div
                  className="success-message"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ type: 'spring', damping: 20, stiffness: 200 }}
                >
                  <motion.div className="success-icon" animate={{ scale: [0, 1.2, 1] }} transition={{ type: 'spring', damping: 15, stiffness: 200 }}>
                    <CheckCircle size={48} />
                  </motion.div>
                  <h3>Analysis Complete!</h3>
                  <p>Redirecting to full analysis...</p>
                </motion.div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </div>
  );
};

export default Upload;