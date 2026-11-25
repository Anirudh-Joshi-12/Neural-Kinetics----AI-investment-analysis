// src/pages/Reports.jsx
import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { ChevronLeft, ChevronRight, Filter, Trash2, RotateCcw, Download, ExternalLink, Plus, FileText, Upload } from 'lucide-react';
import { api } from '../lib/api';
import { format } from 'date-fns';
import { useNavigate, Link } from 'react-router-dom';

const Reports = () => {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({ page: 1, limit: 10, status: '', minScore: '' });
  const [total, setTotal] = useState(0);
  const [deletingId, setDeletingId] = useState(null);
  const navigate = useNavigate();

  const fetchReports = async () => {
    setLoading(true);
    try {
      const data = await api.listReports(filters);
      setReports(data.items);
      setTotal(data.total);
    } catch (err) {
      console.error('Failed to fetch reports:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReports();
  }, [filters]);

  const handleDelete = async (id) => {
    if (!window.confirm('Delete this analysis?')) return;
    setDeletingId(id);
    try {
      await api.deleteReport(id);
      fetchReports();
    } catch (err) {
      alert('Failed to delete');
    } finally {
      setDeletingId(null);
    }
  };

  const handleRetry = async (report) => {
    alert('Re-run analysis by uploading the same PDF on the Upload page.');
  };

  const handleDownload = (report) => {
    const content = `INVESTMENT ANALYSIS REPORT
===========================
File: ${report.original_filename}
Date: ${format(new Date(report.created_at), 'PPP')}
Model: ${report.model_version}

SCORE: ${report.score}/100 (${report.label})
Confidence: ${report.confidence}%
OOD: ${report.is_ood ? 'YES - ' + report.ood_info?.ood_reason : 'No'}

KPIs:
${Object.entries(report.kpis || {}).map(([k, v]) => `  ${k}: ${v !== null ? v.toFixed(4) : 'N/A'}`).join('\n')}

SHAP Values:
${Object.entries(report.shap_values || {}).map(([k, v]) => `  ${k}: ${v > 0 ? '+' : ''}${v.toFixed(4)}`).join('\n')}

BRIEF:
${report.brief}
    `.trim();

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `report-${report.id.slice(0, 8)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const statusColors = {
    completed: 'status-completed',
    failed: 'status-failed',
    processing: 'status-processing',
    pending: 'status-pending',
  };

  const labelColors = {
    'Strong Buy': 'label-strong-buy',
    'Buy': 'label-buy',
    'Hold': 'label-hold',
    'Avoid': 'label-avoid',
    'Strong Avoid': 'label-strong-avoid',
  };

  return (
    <div className="page reports-page">
      <motion.div
        className="reports-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      >
        <div className="header-content">
          <div>
            <h1>Analysis History</h1>
            <p>All your past analyses in one place</p>
          </div>
          <motion.a href="/upload" className="btn-primary" whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
            <Plus size={18} /> New Analysis
          </motion.a>
        </div>

        <motion.div
          className="filters-bar"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="filter-group">
            <Filter size={16} className="filter-icon" />
            <select
              value={filters.status}
              onChange={(e) => setFilters({ ...filters, status: e.target.value, page: 1 })}
              className="filter-select"
            >
              <option value="">All Status</option>
              <option value="completed">Completed</option>
              <option value="failed">Failed</option>
              <option value="processing">Processing</option>
            </select>
            <input
              type="number"
              placeholder="Min Score"
              value={filters.minScore}
              onChange={(e) => setFilters({ ...filters, minScore: e.target.value, page: 1 })}
              className="filter-input"
            />
            <input
              type="date"
              value={filters.dateFrom || ''}
              onChange={(e) => setFilters({ ...filters, dateFrom: e.target.value, page: 1 })}
              className="filter-input"
            />
            <input
              type="date"
              value={filters.dateTo || ''}
              onChange={(e) => setFilters({ ...filters, dateTo: e.target.value, page: 1 })}
              className="filter-input"
            />
          </div>
        </motion.div>
      </motion.div>

      <motion.div
        className="reports-table-container"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        {loading ? (
          <div className="loading-state">
            <div className="spinner" />
            <span>Loading reports...</span>
          </div>
        ) : reports.length === 0 ? (
<motion.div
             className="empty-state"
             initial={{ opacity: 0, y: 20 }}
             animate={{ opacity: 1, y: 0 }}
           >
             <FileText size={48} />
             <h3>No analyses yet</h3>
             <p>Upload your first annual report to get started</p>
<motion.a href="/upload" className="btn-primary" whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
                <Upload size={18} /> Upload First Report
              </motion.a>
           </motion.div>
        ) : (
          <div className="table-wrapper">
            <table className="reports-table">
              <thead>
                <tr>
                  <th>File</th>
                  <th>Date</th>
                  <th>Score</th>
                  <th>Label</th>
                  <th>Confidence</th>
                  <th>OOD</th>
                  <th>Status</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                <AnimatePresence>
                  {reports.map((report) => (
                    <motion.tr
                      key={report.id}
                      initial={{ opacity: 0, y: 16 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      transition={{ duration: 0.3 }}
                    >
                      <td>
                        <span className="filename" title={report.original_filename}>
                          {report.original_filename}
                        </span>
                      </td>
                      <td>{format(new Date(report.created_at), 'MMM d, yyyy HH:mm')}</td>
                      <td className="score-cell">
                        <span className={`score-value ${report.score !== null ? (report.score >= 75 ? 'high' : report.score >= 50 ? 'mid' : 'low') : ''}`}>
                          {report.score !== null ? report.score.toFixed(1) : '\u2014'}
                        </span>
                      </td>
                      <td>
                        {report.label && (
                          <span className={`label-badge ${labelColors[report.label] || ''}`}>
                            {report.label}
                          </span>
                        )}
                      </td>
                      <td>{report.confidence !== null ? `${report.confidence.toFixed(1)}%` : '\u2014'}</td>
                      <td>
                        {report.is_exceptional ? (
                          <span className="ood-badge" style={{ color: '#10b981', borderColor: '#10b981', backgroundColor: '#ecfdf5' }}>
                            ✓ Exceptional
                          </span>
                        ) : (
                          <span className={`ood-badge ${report.is_ood ? 'ood-true' : ''}`}>
                            {report.is_ood ? '⚠ OOD' : '✓ In-Distribution'}
                          </span>
                        )}
                      </td>
                      <td>
                        <span className={`status-badge ${statusColors[report.status] || ''}`}>
                          {report.status}
                        </span>
                      </td>
                      <td>
<div className="actions-cell">
                           <motion.button
                             className="action-btn"
                             onClick={() => navigate(`/analysis/${report.id}`)}
                             whileHover={{ scale: 1.1 }}
                             whileTap={{ scale: 0.95 }}
                             title="View Analysis"
                           >
                             <ExternalLink size={16} />
                           </motion.button>
                          {report.status === 'failed' && (
                            <motion.button
                              className="action-btn retry"
                              onClick={() => handleRetry(report)}
                              whileHover={{ scale: 1.1 }}
                              whileTap={{ scale: 0.95 }}
                              title="Retry Analysis"
                            >
                              <RotateCcw size={16} />
                            </motion.button>
                          )}
                          <motion.button
                            className="action-btn download"
                            onClick={() => handleDownload(report)}
                            whileHover={{ scale: 1.1 }}
                            whileTap={{ scale: 0.95 }}
                            title="Download Report"
                          >
                            <Download size={16} />
                          </motion.button>
                          {deletingId === report.id ? (
                            <div className="spinner-small" />
                          ) : (
                            <motion.button
                              className="action-btn delete"
                              onClick={() => handleDelete(report.id)}
                              whileHover={{ scale: 1.1, backgroundColor: '#fee' }}
                              whileTap={{ scale: 0.95 }}
                              title="Delete"
                            >
                              <Trash2 size={16} />
                            </motion.button>
                          )}
                        </div>
                      </td>
                    </motion.tr>
                  ))}
                </AnimatePresence>
              </tbody>
            </table>

            <motion.div
              className="pagination"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <span>Showing {Math.max(0, (filters.page - 1) * filters.limit + 1)} - {Math.min(filters.page * filters.limit, total)} of {total}</span>
              <div className="pagination-controls">
                <motion.button
                  onClick={() => setFilters({ ...filters, page: Math.max(1, filters.page - 1) })}
                  disabled={filters.page <= 1}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  disabled={filters.page <= 1}
                >
                  <ChevronLeft size={18} />
                </motion.button>
                <span className="page-info">
                  Page {filters.page} of {Math.ceil(total / filters.limit) || 1}
                </span>
                <motion.button
                  onClick={() => setFilters({ ...filters, page: filters.page + 1 })}
                  disabled={filters.page * filters.limit >= total}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  disabled={filters.page * filters.limit >= total}
                >
                  <ChevronRight size={18} />
                </motion.button>
              </div>
            </motion.div>
          </div>
        )}
      </motion.div>
    </div>
  );
};

export default Reports;