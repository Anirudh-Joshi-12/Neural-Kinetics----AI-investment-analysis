// src/pages/Analysis.jsx
import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ExternalLink, Download, RotateCcw, Trash2, ChevronLeft, ChevronRight,
  TrendingUp, TrendingDown, Minus, Plus, AlertCircle, CheckCircle, X,
  Target, Activity, Zap
} from 'lucide-react';
import { api } from '../lib/api';
import { EASE } from '../constants';

const labelColors = {
  'Strong Buy': 'strong-buy',
  'Buy': 'buy',
  'Hold': 'hold',
  'Avoid': 'avoid',
  'Strong Avoid': 'strong-avoid',
};

const statusColors = {
  'completed': 'completed',
  'processing': 'processing',
  'failed': 'failed',
  'pending': 'pending',
};

const formatScore = (score) => score !== null ? score.toFixed(2) : '\u2014';
const formatKPI = (key, value) => {
  if (value === null || value === undefined) return 'N/A';
  if (key === 'debt_equity' || key === 'current_ratio') return value.toFixed(2);
  return (value * 100).toFixed(2) + '%';
};

const kpiLabels = {
  revenue_growth: 'Revenue Growth',
  net_margin: 'Net Margin',
  debt_equity: 'Debt/Equity',
  roe: 'Return on Equity',
  current_ratio: 'Current Ratio',
};

const kpiDescriptions = {
  revenue_growth: 'Year-over-year revenue growth rate',
  net_margin: 'Net income as % of revenue',
  debt_equity: 'Total debt to shareholders equity',
  roe: 'Net income / Shareholders equity',
  current_ratio: 'Current assets / Current liabilities',
};

const Analysis = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    const fetchReport = async () => {
      try {
        setLoading(true);
        const data = await api.getReport(id);
        setReport(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchReport();
  }, [id]);

  const handleDelete = async () => {
    if (!window.confirm('Delete this analysis? This cannot be undone.')) return;
    try {
      setDeleting(true);
      await api.deleteReport(id);
      navigate('/reports');
    } catch (err) {
      alert('Failed to delete: ' + err.message);
    } finally {
      setDeleting(false);
    }
  };

  const handleRetry = async () => {
    try {
      setLoading(true);
      const data = await api.reprocessReport(id);
      setReport(data);
    } catch (err) {
      alert('Retry failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    if (!report) return;
    try {
      await api.downloadReportPdf(id);
    } catch (err) {
      alert('Download failed: ' + err.message);
    }
  };

  if (loading) {
    return (
      <div className="page analysis-page loading">
        <div className="loading-skeleton">
          <div className="skeleton-header" />
          <div className="skeleton-metrics" />
          <div className="skeleton-kpis" />
          <div className="skeleton-shap" />
          <div className="skeleton-brief" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page analysis-page error">
        <div className="error-state">
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="error-icon"
          >
            <AlertCircle size={64} />
          </motion.div>
          <h2>Failed to Load Analysis</h2>
          <p>{error}</p>
          <motion.button
            className="btn-primary"
            onClick={() => window.location.reload()}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <RotateCcw size={18} /> Reload
          </motion.button>
        </div>
      </div>
    );
  }

  const kpiItems = [
    { key: 'revenue_growth', label: 'Revenue Growth', desc: 'YoY revenue growth rate' },
    { key: 'net_margin', label: 'Net Margin', desc: 'Net income / Revenue' },
    { key: 'debt_equity', label: 'Debt/Equity', desc: 'Total debt / Shareholders equity' },
    { key: 'roe', label: 'Return on Equity', desc: 'Net income / Shareholders equity' },
    { key: 'current_ratio', label: 'Current Ratio', desc: 'Current assets / Current liabilities' },
  ];

  const shapItems = report?.shap_values
    ? Object.entries(report.shap_values)
        .sort(([,a], [,b]) => Math.abs(b) - Math.abs(a))
        .map(([key, value]) => ({ key, value }))
    : [];

  const scoreClass = (score) => {
    if (score === null) return '';
    if (score >= 75) return 'high';
    if (score >= 50) return 'mid';
    return 'low';
  };

  const labelClass = (label) => labelColors[label] || '';

  // Early return if report is null/undefined or has no kpis
  if (!report || !report.kpis) {
    return (
      <div className="page analysis-page error">
        <div className="error-state">
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="error-icon"
          >
            <AlertCircle size={64} />
          </motion.div>
          <h2>Analysis Not Available</h2>
          <p>This report is not available or has failed processing.</p>
          <motion.button
            className="btn-primary"
            onClick={() => window.location.reload()}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <RotateCcw size={18} /> Reload
          </motion.button>
        </div>
      </div>
    );
  }

  return (
    <div className="page analysis-page">
      <motion.div
        className="analysis-container"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      >
        {/* Header */}
        <motion.header
          className="analysis-header"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        >
          <motion.button
            className="back-btn"
            onClick={() => navigate(-1)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M19 12H5M12 19l-7-7 7-7" />
            </svg>
            <span>Back to Reports</span>
          </motion.button>

          <motion.div
            className="header-info"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <motion.span
              className={`status-badge ${statusColors[report?.status] || ''}`}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 }}
            >
              {report?.status || 'Unknown'}
            </motion.span>
            <h1>{report?.original_filename || 'Analysis'}</h1>
            <p className="meta">
              {report?.model_version || 'rf_v1'} • {report?.llm_model || 'Nemotron'} •
              {report?.created_at ? new Date(report.created_at).toLocaleDateString() : ''}
            </p>
          </motion.div>

          <motion.div
            className="header-actions"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <motion.button
              className="btn-secondary"
              onClick={handleRetry}
              disabled={loading}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <RotateCcw size={16} /> Re-run
            </motion.button>
            <motion.button
              className="btn-secondary"
              onClick={() => handleDownload()}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Download size={16} /> Download PDF
            </motion.button>
            <motion.button
              className="btn-danger"
              onClick={() => setDeleting(true)}
              whileHover={{ scale: 1.02, backgroundColor: '#fee' }}
              whileTap={{ scale: 0.95 }}
            >
              <Trash2 size={16} /> Delete
            </motion.button>
          </motion.div>
        </motion.header>

        {/* Exceptional Company Banner */}
        {report?.is_exceptional && (
          <motion.div
            className="ood-banner"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            style={{ borderLeft: '5px solid #10b981' }}
          >
            <motion.div className="ood-icon" animate={{ scale: [1, 1.1, 1] }} transition={{ duration: 1.5, repeat: Infinity }}>
              <AlertCircle size={24} style={{ color: '#10b981' }} />
            </motion.div>
            <div className="ood-content">
              <strong>✓ Exceptional Company</strong>
              <span>Out-of-distribution in a good way — all KPIs are top-tier and ML score is Strong Buy or better.</span>
            </div>
            <div className="ood-metrics">
              <span>Coverage: {(report.ood_info?.coverage_score * 100 || 0).toFixed(1)}%</span>
              <span>kNN %ile: {(report.ood_info?.knn_percentile * 100 || 0).toFixed(1)}%</span>
            </div>
          </motion.div>
        )}
        {/* OOD Banner */}
        {report?.is_ood && !report?.is_exceptional && (
          <motion.div
            className="ood-banner"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <motion.div className="ood-icon" animate={{ scale: [1, 1.1, 1] }} transition={{ duration: 1.5, repeat: Infinity }}>
              <AlertCircle size={24} />
            </motion.div>
            <div className="ood-content">
              <strong>⚠ Out-of-Distribution Detected</strong>
              <span>{report.ood_info?.ood_reason || 'KPI profile falls outside training distribution'}</span>
            </div>
            <div className="ood-metrics">
              <span>Coverage: {(report.ood_info?.coverage_score * 100 || 0).toFixed(1)}%</span>
              <span>kNN %ile: {(report.ood_info?.knn_percentile * 100 || 0).toFixed(1)}%</span>
            </div>
          </motion.div>
        )}

        {/* Score Card */}
        <motion.section
          className="score-section"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
        >
          <div className="score-card">
            <div className="score-main">
              <motion.div
                className={`score-circle ${scoreClass(report?.score)}`}
                initial={{ scale: 0.5, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ type: 'spring', damping: 15, stiffness: 150, delay: 0.4 }}
              >
                <span className="score-value">{report?.score !== null ? report.score.toFixed(1) : '\u2014'}</span>
                <span className="score-label">/ 100</span>
              </motion.div>
              <div className="score-label-info">
                <motion.span
                  className={`score-label-text ${labelClass(report?.label)}`}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5 }}
                >
                  {report?.label || '\u2014'}
                </motion.span>
                <motion.p
                  className="confidence"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.6 }}
                >
                  Confidence: {report?.confidence !== null ? report.confidence.toFixed(1) + '%' : '\u2014'}
                </motion.p>
              </div>
            </div>

            <motion.div
              className="score-breakdown"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 }}
            >
              {kpiItems.map((kpi, i) => (
                <motion.div
                  key={kpi.key}
                  className="breakdown-item"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.7 + i * 0.08 }}
                >
                  <span className="breakdown-label">{kpi.label}</span>
                  <div className="breakdown-bar">
                    <motion.div
                      className="bar-fill"
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(Math.abs(report?.kpis?.[kpi.key] || 0) * 100, 100)}%` }}
                      transition={{ duration: 1, delay: 0.8 + i * 0.08, type: 'spring', stiffness: 100 }}
                      style={{ background: report?.kpis?.[kpi.key] !== null && report?.kpis?.[kpi.key] >= 0 ? 'var(--positive)' : 'var(--negative)' }}
                    />
                  </div>
                  <span className="breakdown-value">
                    {report?.kpis?.[kpi.key] !== null ? formatKPI(kpi.key, report?.kpis?.[kpi.key]) : 'N/A'}
                  </span>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </motion.section>

        {/* KPI Details Grid */}
        <motion.section
          className="kpi-section"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.8, ease: [0.16, 1, 0.3, 1] }}
        >
          <h2>Key Performance Indicators</h2>
          <div className="kpi-grid">
            {kpiItems.map((kpi, i) => (
              <motion.div
                key={kpi.key}
                className={`kpi-card ${report?.kpis?.[kpi.key] !== null && report?.kpis?.[kpi.key] >= 0 ? 'positive' : report?.kpis?.[kpi.key] !== null ? 'negative' : ''}`}
                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.5, delay: 0.9 + i * 0.08, type: 'spring', damping: 20 }}
              >
                <div className="kpi-header">
                  <h3>{kpi.label}</h3>
                  <span className="kpi-value">
                    {report?.kpis?.[kpi.key] !== null ? formatKPI(kpi.key, report?.kpis?.[kpi.key]) : 'N/A'}
                  </span>
                </div>
                <p className="kpi-desc">{kpi.descriptions?.[kpi.key] || kpiDescriptions[kpi.key]}</p>
                <div className="kpi-bar">
                  <motion.div
                    className="kpi-bar-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(Math.abs(report?.kpis?.[kpi.key] || 0) * 100, 100)}%` }}
                    transition={{ duration: 0.8, delay: 1.1 + i * 0.08, type: 'spring', stiffness: 100 }}
                    style={{ background: report?.kpis?.[kpi.key] !== null && report?.kpis?.[kpi.key] >= 0 ? 'var(--positive)' : 'var(--negative)' }}
                  />
                </div>
              </motion.div>
            ))}
          </div>
        </motion.section>

        {/* SHAP Feature Importance */}
        {shapItems.length > 0 && (
          <motion.div
            className="shap-card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1.1, ease: [0.16, 1, 0.3, 1] }}
          >
            <section className="shap-section">
              <h2>SHAP Feature Importance</h2>
              <p className="section-subtitle">How each KPI drove the score (positive = boosted score, negative = reduced score)</p>
              <div className="shap-chart">
                {shapItems.map((item, i) => (
                  <motion.div
                    key={item.key}
                    className={`shap-bar ${item.value >= 0 ? 'positive' : 'negative'}`}
                    initial={{ opacity: 0, x: item.value >= 0 ? -30 : 30 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: 1.2 + i * 0.08, type: 'spring', damping: 20 }}
                  >
                    <div className="shap-label">
                      <span className="shap-name">{kpiLabels[item.key] || item.key}</span>
                      <span className="shap-value">{item.value >= 0 ? '+' : ''}{item.value.toFixed(4)}</span>
                    </div>
                    <motion.div
                      className="shap-bar-visual"
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(Math.abs(item.value) * 100, 100)}%` }}
                      transition={{ duration: 0.8, delay: 1.3 + i * 0.08, type: 'spring', stiffness: 100 }}
                      style={{ background: item.value >= 0 ? 'var(--positive)' : 'var(--negative)' }}
                    />
                  </motion.div>
                ))}
              </div>
            </section>
          </motion.div>
        )}

        {/* Investment Brief */}
        {report?.brief && (
          <motion.section
            className="brief-section"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1.5, ease: [0.16, 1, 0.3, 1] }}
          >
            <h2>Investment Brief</h2>
            <motion.div
              className="brief-content"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1.6 }}
            >
              {report.brief.split('\n').map((para, i) => (
                <motion.p
                  key={i}
                  className="brief-paragraph"
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 1.6 + i * 0.1 }}
                >
                  {para}
                </motion.p>
              ))}
            </motion.div>
          </motion.section>
        )}

        {/* OOD Info Detail */}
        {report?.is_ood && report?.ood_info && (
          <motion.section
            className="ood-detail-section"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1.7 }}
          >
            <h2>Out-of-Distribution Details</h2>
            <div className="ood-detail-grid">
              <div className="ood-detail-item">
                <span className="detail-label">Coverage Score</span>
                <span className="detail-value">{(report.ood_info.coverage_score * 100 || 0).toFixed(1)}%</span>
              </div>
              <div className="ood-detail-item">
                <span className="detail-label">kNN Percentile</span>
                <span className="detail-value">{(report.ood_info.knn_percentile * 100 || 0).toFixed(1)}%</span>
              </div>
              <div className="ood-detail-item">
                <span className="detail-label">kNN Distance</span>
                <span className="detail-value">{report.ood_info.knn_distance?.toFixed(4) || '—'}</span>
              </div>
            </div>
          </motion.section>
        )}

        {/* OOD Metrics Summary */}
        {report?.ood_info && (
          <motion.section
            className="ood-metrics-section"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 1.5, ease: [0.16, 1, 0.3, 1] }}
          >
            <h2>OOD Detection Metrics</h2>
            <p className="section-subtitle">How unusual is this KPI profile compared to training data?</p>
            <div className="ood-metrics-grid">
              <motion.div className="ood-metric-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 1.6 }}>
                <span className="metric-icon">
                  <TrendingUp size={24} />
                </span>
                <div className="metric-info">
                  <span className="metric-label">kNN Percentile</span>
                  <span className="metric-value">{((report.ood_info.knn_percentile || 0) * 100).toFixed(1)}%</span>
                </div>
              </motion.div>
              <motion.div className="ood-metric-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
                <span className="metric-icon"><Target size={24} /></span>
                <div className="metric-info">
                  <span className="metric-label">Coverage Score</span>
                  <span className="metric-value">{(report.ood_info.coverage_score * 100 || 0).toFixed(1)}%</span>
                </div>
              </motion.div>
              <motion.div className="ood-metric-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
                <span className="metric-icon"><Activity size={24} /></span>
                <div className="metric-info">
                  <span className="metric-label">kNN Distance</span>
                  <span className="metric-value">{report.ood_info.knn_distance?.toFixed(4) || '—'}</span>
                </div>
              </motion.div>
            </div>
            <motion.p className="ood-explanation" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
              {report.is_ood ? (
                <>
                  <strong>⚠ This report is flagged as Out-of-Distribution.</strong> The extracted KPI profile falls outside the regime
                  the model was trained on. The score and confidence should be treated as a structural screen only, not a calibrated probability.
                </>
              ) : (
                <>
                  <strong>✓ In-Distribution.</strong> The KPI profile falls within the model's training distribution.
                  Score and confidence are more reliable for this profile.
                </>
              )}
            </motion.p>
          </motion.section>
        )}

        {/* Footer Actions */}
        <motion.div
          className="analysis-footer"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.8 }}
        >
          <motion.button
            className="btn-primary"
            onClick={() => navigate('/reports')}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <ChevronLeft size={18} /> Back to Reports
          </motion.button>
          <motion.button
            className="btn-secondary"
            onClick={handleRetry}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <RotateCcw size={18} /> Re-run Analysis
          </motion.button>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default Analysis;