// src/pages/About.jsx
import { motion } from 'motion/react';
import {
  FileText, Cpu, Zap, Target, Brain, Shield,
  AlertTriangle, CheckCircle, XCircle, TrendingUp,
  Upload, XCircle as XCircle2, CheckCircle as CheckCircle2, AlertTriangle as AlertTriangle2, TrendingUp as TrendingUp2, Target as Target2, Layers,
  Zap as ZapIcon, Shield as ShieldIcon, Database, Cpu as CpuIcon, Globe, Lock, FileText as FileText2,
  BookOpen
} from 'lucide-react';
import { EASE } from '../constants';

const About = () => (
  <div className="page about-page">
    <motion.div
      className="about-container"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
    >
      {/* Hero Section */}
      <motion.section
        className="about-hero"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
      >
        <motion.div className="hero-content">
          <motion.span
            className="hero-badge"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
          >
            <span className="badge-dot" />
            Methodology & Model Card
          </motion.span>

          <motion.h1
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
          >
            How <span className="highlight">NeuralKinetics</span> Works
          </motion.h1>

          <motion.p
            className="hero-subtitle"
            initial={{ y: 16, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.4, ease: [0.16, 1, 0.3, 1] }}
          >
            Transparent methodology behind our AI-powered investment analysis pipeline.
            From PDF ingestion to calibrated scoring with out-of-distribution detection.
          </motion.p>
        </motion.div>

        <motion.div
          className="hero-visual"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1, delay: 0.6, ease: [0.16, 1, 0.3, 1] }}
        >
          <div className="pipeline-diagram">
            <motion.div
              className="pipeline-step"
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.7 }}
            >
              <div className="step-icon"><span className="icon-bg"><FileText size={24} /></span></div>
              <h3>PDF Ingestion</h3>
              <p>PyMuPDF extracts text, tables, and structure from 10-K filings</p>
            </motion.div>

            <motion.div className="pipeline-arrow">\u2192</motion.div>

            <motion.div
              className="pipeline-step"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.8 }}
            >
              <div className="step-icon"><span className="icon-bg"><Cpu size={24} /></span></div>
              <h3>Semantic Chunking</h3>
              <p>Intelligent section-aware chunking with 800-token windows, 100-token overlap</p>
            </motion.div>

            <motion.div className="pipeline-arrow">\u2192</motion.div>

            <motion.div
              className="pipeline-step"
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.9 }}
            >
              <div className="step-icon"><span className="icon-bg"><Zap size={24} /></span></div>
              <h3>Nemotron Embeddings</h3>
              <p>nvidia/nemotron-3.5-embed-1b (2048-dim) batched vector embeddings</p>
            </motion.div>

            <motion.div className="pipeline-arrow">\u2192</motion.div>

            <motion.div
              className="pipeline-step"
              initial={{ opacity: 0, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 1.0 }}
            >
              <div className="step-icon"><span className="icon-bg"><Target size={24} /></span></div>
              <h3>pgVector Search</h3>
              <p>Cosine similarity retrieval of financial-statement chunks (k=6)</p>
            </motion.div>

            <motion.div className="pipeline-arrow">\u2192</motion.div>

            <motion.div
              className="pipeline-step highlight"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 1.1 }}
            >
              <div className="step-icon"><span className="icon-bg highlight"><Brain size={24} /></span></div>
              <h3>Nemotron 3 Ultra LLM</h3>
              <p>Single-call KPI extraction from retrieved context (5 KPIs + formulas)</p>
            </motion.div>

            <motion.div className="pipeline-arrow">\u2192</motion.div>

            <motion.div
              className="pipeline-step"
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 1.2 }}
            >
              <div className="step-icon"><span className="icon-bg"><TrendingUp size={24} /></span></div>
              <h3>RF Scoring + SHAP</h3>
              <p>Random Forest (500 trees) + SHAP TreeExplainer for attribution</p>
            </motion.div>

            <motion.div className="pipeline-arrow">\u2192</motion.div>

            <motion.div
              className="pipeline-step"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 1.3 }}
            >
              <div className="step-icon"><span className="icon-bg"><BookOpen size={24} /></span></div>
              <h3>Investment Brief</h3>
              <p>Nemotron 3 Ultra generates structured investment thesis with SHAP narrative</p>
            </motion.div>
          </div>
        </motion.div>
      </motion.section>

      {/* Model Card Section */}
      <motion.section
        className="section model-card"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
      >
        <div className="section-header">
          <motion.span className="section-badge" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}>
            <span className="badge-dot" />
            Model Card v1.0
          </motion.span>
          <motion.h2 initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.2 }}>
            Model Card: <span className="highlight">Random Forest Classifier v1.0</span>
          </motion.h2>
        </div>

        <div className="model-card-grid">
          <motion.div className="card spec-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <h3><Brain size={20} /> Architecture</h3>
            <ul>
              <li><strong>Algorithm:</strong> Random Forest Classifier (sklearn)</li>
              <li><strong>Trees:</strong> 500 estimators</li>
              <li><strong>Max Depth:</strong> 8</li>
              <li><strong>Min Samples Leaf:</strong> 2</li>
              <li><strong>Class Weight:</strong> Balanced</li>
              <li><strong>Features:</strong> 5 KPIs (revenue_growth, net_margin, debt_equity, roe, current_ratio)</li>
            </ul>
          </motion.div>

          <motion.div className="card spec-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
            <h3><TrendingUp size={20} /> Training Data</h3>
            <ul>
              <li><strong>Samples:</strong> 1,040 (990 synthetic + 50 real anchors \u00d7 5x weight)</li>
              <li><strong>Regimes:</strong> 6 archetypes (Stable, Quality Growth, Hypergrowth, Distressed, Releveraged, Borderline)</li>
              <li><strong>Positive Rate:</strong> 47% (balanced via class_weight)</li>
              <li><strong>Test Split:</strong> 20% stratified</li>
              <li><strong>CV Folds:</strong> 5-fold stratified</li>
              <li><strong>Synthetic Engine:</strong> Beta-distributed KPIs with regime-specific moments</li>
            </ul>
          </motion.div>

          <motion.div className="card spec-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
            <h3><TrendingUp size={20} /> Performance</h3>
            <ul>
              <li><strong>Accuracy:</strong> 87.0%</li>
              <li><strong>F1 Score:</strong> 85.9%</li>
              <li><strong>CV F1 (5-fold):</strong> 84.7% \u00b1 5.4%</li>
              <li><strong>Mean Confidence:</strong> 46.4% (calibrated)</li>
              <li><strong>Thresholds:</strong> Strong Buy \u226575, Buy \u226560, Hold \u226545, Avoid \u226530</li>
              <li><strong>SHAP:</strong> TreeExplainer (exact, per-prediction)</li>
            </ul>
          </motion.div>

          <motion.div className="card spec-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
            <h3><Shield size={20} /> OOD Detection</h3>
            <ul>
              <li><strong>Method:</strong> kNN Distance (k=10) in Standardized Feature Space</li>
              <li><strong>Reference:</strong> 1,040 training samples (standardized)</li>
              <li><strong>Threshold:</strong> 95th percentile of training kNN distances</li>
              <li><strong>Coverage Score:</strong> 1 - percentile(kNN distance)</li>
              <li><strong>Fallback:</strong> Mahalanobis distance (MinCovDet robust covariance)</li>
              <li><strong>Flag:</strong> {'is_ood = true when kNN pct > 95% OR coverage < 5%'}</li>
            </ul>
          </motion.div>
        </div>
      </motion.section>

      {/* KPI Definitions */}
      <motion.section
        className="section kpi-definitions"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
      >
        <motion.div className="section-header" initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.2 }}>
          <motion.span className="section-badge"><span className="badge-dot" /> KPI Definitions</motion.span>
          <motion.h2>Financial KPIs <span className="highlight">Extracted & Scored</span></motion.h2>
        </motion.div>

        <div className="kpi-grid">
          {[
            { key: 'revenue_growth', icon: TrendingUp, formula: '(Revenue_t - Revenue_t-1) / Revenue_t-1', desc: 'Year-over-year top-line growth. Primary driver of Buy signals.' },
            { key: 'net_margin', icon: TrendingUp, formula: 'Net Income / Revenue', desc: 'Net profit margin. High margin = efficient operations.' },
            { key: 'debt_equity', icon: Shield, formula: 'Total Debt / Shareholders Equity', desc: 'Leverage ratio. Lower = stronger balance sheet, less financial risk.' },
            { key: 'roe', icon: TrendingUp, formula: 'Net Income / Shareholders Equity', desc: 'Capital efficiency. ROE > 15% typically signals quality compounding.' },
            { key: 'current_ratio', icon: Shield, formula: 'Current Assets / Current Liabilities', desc: 'Short-term liquidity. > 1.5 = comfortable, > 3.0 = potential inefficiency.' },
          ].map(({ key, icon: Icon, formula, desc }, i) => (
            <motion.div
              key={key}
              className="kpi-def-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 + i * 0.1 }}
            >
              <div className="kpi-icon"><Icon size={24} /></div>
              <h3>{key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</h3>
              <code className="formula">{formula}</code>
              <p>{desc}</p>
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* OOD Explanation */}
      <motion.section
        className="section ood-explanation"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
      >
        <motion.div className="section-header" initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.2 }}>
          <motion.span className="section-badge"><span className="badge-dot" style={{ background: '#ffc107' }} /> Out-of-Distribution Detection</motion.span>
          <motion.h2>Why <span className="highlight">OOD Matters</span></motion.h2>
        </motion.div>

        <div className="ood-content">
          <motion.div className="ood-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <h3><AlertTriangle size={24} /> The Problem</h3>
            <p>Machine learning models are interpolators, not extrapolators. When a company's KPI profile falls far outside the training distribution (e.g., NVIDIA's 65% revenue growth, 76% ROE), the model has never seen similar patterns during training.</p>
            <p>The model will still output a score, but that score is an extrapolation \u2014 not a calibrated probability. Confidence becomes unreliable.</p>
          </motion.div>

          <motion.div className="ood-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <h3><CheckCircle size={24} /> Our Solution</h3>
            <p>We compute the k-Nearest Neighbor distance in standardized KPI space. If the new sample's 10th-nearest-neighbor distance exceeds the 95th percentile of training distances, we flag it as Out-of-Distribution.</p>
            <ul>
              <li><strong>kNN Distance:</strong> Euclidean distance in standardized 5-D KPI space (k=10)</li>
              <li><strong>Threshold:</strong> 95th percentile of training sample distances</li>
              <li><strong>Coverage Score:</strong> 1 - percentile(kNN distance) \u2014 1.0 = center of distribution, 0.0 = extreme outlier</li>
              <li><strong>Fallback:</strong> Mahalanobis distance with MinCovDet robust covariance</li>
            </ul>
          </motion.div>

          <motion.div className="ood-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
            <h3><AlertTriangle size={24} /> How to Interpret</h3>
            <ul>
              <li><strong>{'is_ood = false, coverage > 0.5:'}</strong> High confidence. Score is well-calibrated.</li>
              <li><strong>{'is_ood = true, coverage 0.1-0.5:'}</strong> Moderate extrapolation. Score directional but confidence low.</li>
              <li><strong>{'is_ood = true, coverage < 0.1:'}</strong> Extreme extrapolation. Score is structural screen only \u2014 not a calibrated probability.</li>
            </ul>
            <p className="ood-note">The OOD badge in the analysis view signals when to treat the score as a structural screen rather than a calibrated probability.</p>
          </motion.div>
        </div>
      </motion.section>

      {/* Pipeline Visualization */}
      <motion.section
        className="section pipeline"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
      >
        <motion.div className="section-header" initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.2 }}>
          <motion.span className="section-badge"><span className="badge-dot" /> Pipeline</motion.span>
          <motion.h2>End-to-End <span className="highlight">Processing Pipeline</span></motion.h2>
        </motion.div>

        <div className="pipeline-steps">
          {[
            { icon: FileText, title: 'PDF Ingestion', desc: 'PyMuPDF extracts text, tables, and structure from 10-K filings' },
            { icon: Cpu, title: 'Semantic Chunking', desc: 'Section-aware chunking with 800-token windows, 100-token overlap' },
            { icon: Zap, title: 'Nemotron Embeddings', desc: 'nvidia/nemotron-3.5-embed-1b (2048-dim) batched vector embeddings' },
            { icon: Target, title: 'pgVector Search', desc: 'Cosine similarity retrieval of financial-statement chunks (k=6)' },
            { icon: Brain, title: 'Nemotron 3 Ultra LLM', desc: 'Single-call KPI extraction from retrieved context (5 KPIs + formulas)' },
            { icon: TrendingUp, title: 'RF Scoring + SHAP', desc: 'Random Forest (500 trees) + SHAP TreeExplainer for attribution' },
            { icon: AlertTriangle, title: 'OOD Detection', desc: 'kNN distance (k=10) in standardized space + Mahalanobis fallback' },
            { icon: BookOpen, title: 'Investment Brief', desc: 'Nemotron 3 Ultra generates structured investment thesis with SHAP narrative' },
          ].map((step, i) => (
            <motion.div
              key={step.title}
              className="pipeline-step-card"
              initial={{ opacity: 0, x: i % 2 === 0 ? -30 : 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.6, delay: i * 0.1 }}
            >
              <div className="step-number">{i + 1}</div>
              <div className="step-icon"><step.icon size={28} /></div>
              <h3>{step.title}</h3>
              <p>{step.desc}</p>
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* Limitations & Privacy */}
      <motion.section
        className="section limitations"
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
      >
        <motion.div className="section-header" initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.2 }}>
          <motion.span className="section-badge"><span className="badge-dot" /> Limitations & Privacy</motion.span>
          <motion.h2>Known <span className="highlight">Limitations</span> & Data Privacy</motion.h2>
        </motion.div>

        <div className="limitations-grid">
          <motion.div className="limitation-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <h3><XCircle size={24} /> Synthetic Training Data</h3>
            <p>The model is trained on 990 synthetic samples generated from regime archetypes, plus 50 real anchors. While regimes capture real financial patterns, synthetic data cannot capture all real-world nuances, edge cases, or regime transitions.</p>
          </motion.div>

          <motion.div className="limitation-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <h3><XCircle size={24} /> No Forward Returns</h3>
            <p>The model predicts a "Buy/Hold/Avoid" label based on current KPIs, not future stock returns. It has no knowledge of market sentiment, macro regime, competitive dynamics, or future earnings surprises.</p>
          </motion.div>

          <motion.div className="limitation-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
            <h3><XCircle size={24} /> Single Model, No Ensemble</h3>
            <p>A single Random Forest is used. No gradient boosting, neural nets, or ensemble voting. This limits expressivity but improves interpretability (exact SHAP values).</p>
          </motion.div>

          <motion.div className="limitation-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
            <h3><XCircle size={24} /> US-Only, Large-Cap Bias</h3>
            <p>Training anchors are US mega-cap tech/consumer/industrial. Performance on small-caps, international filers, REITs, banks, or insurance companies is unvalidated.</p>
          </motion.div>

          <motion.div className="limitation-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
            <h3><Lock size={24} /> Data Privacy</h3>
            <p>Uploaded PDFs are stored locally on your server (backend/data/). Only extracted text chunks are sent to NVIDIA NIM for embeddings and KPI extraction. No raw financials leave your infrastructure.</p>
          </motion.div>

          <motion.div className="limitation-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
            <h3><Globe size={24} /> NVIDIA NIM Dependency</h3>
            <p>The pipeline requires access to NVIDIA NIM endpoints for Nemotron embeddings and LLM calls. Ensure your network allows HTTPS to integrate.api.nvidia.com.</p>
          </motion.div>
        </div>
      </motion.section>

      {/* Footer CTA */}
      <motion.footer
        className="about-footer"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.5, ease: [0.16, 1, 0.3, 1] }}
      >
        <motion.div className="footer-content">
          <motion.h2 initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.2 }}>
            Ready to Analyze Your First Report?
          </motion.h2>
          <motion.p initial={{ y: 16, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.3 }}>
            Upload a 10-K PDF and get a scored investment brief with SHAP explainability in under 30 seconds.
          </motion.p>
          <motion.div className="footer-cta" initial={{ y: 16, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.4 }}>
            <a href="/upload" className="btn-primary">
              <Upload size={18} /> Upload First Report
            </a>
            <a href="/reports" className="btn-secondary">View History</a>
          </motion.div>
        </motion.div>
      </motion.footer>
    </motion.div>
  </div>
);

export default About;