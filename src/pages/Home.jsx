// src/pages/Home.jsx
import { motion } from 'motion/react';
import { Link } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';
import { EASE } from '../constants';

const Home = () => (
  <div className="page home-page">
    <motion.div
      className="hero"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 1, ease: EASE }}
    >
      <motion.div
        className="hero-content"
        initial={{ y: 30, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 1, delay: 0.3, ease: EASE }}
      >
        <motion.span
          className="hero-badge"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.4, ease: EASE }}
        >
          <span className="badge-dot" />
          AI-Powered Investment Analysis
        </motion.span>

        <motion.h1
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.5, ease: EASE }}
        >
          <span className="heading-line">Extract. Analyze. Decide.</span>
          <br />
          <span className="heading-line highlight">With Confidence.</span>
        </motion.h1>

        <motion.p
          className="hero-subtitle"
          initial={{ y: 16, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.7, ease: EASE }}
        >
          Upload any annual report PDF. Our pipeline extracts 5 key financial KPIs,
          scores the investment with a calibrated Random Forest model, and generates
          an investment brief with SHAP explainability — all in under 30 seconds.
        </motion.p>

        <motion.div
          className="hero-cta"
          initial={{ y: 16, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.9, ease: EASE }}
        >
          <Link to="/upload" className="btn-primary">
            <span>Analyze a Report</span>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" aria-hidden="true">
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </Link>
          <Link to="/reports" className="btn-ghost">View History</Link>
        </motion.div>

        <motion.div
          className="hero-stats"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 1.1, ease: EASE }}
        >
          <div className="stat">
            <span className="stat-value">5</span>
            <span className="stat-label">KPIs Extracted</span>
          </div>
          <div className="stat">
            <span className="stat-value">30s</span>
            <span className="stat-label">Avg. Processing</span>
          </div>
          <div className="stat">
            <span className="stat-value">OOD</span>
            <span className="stat-label">Detection</span>
          </div>
        </motion.div>
      </motion.div>
    </motion.div>
  </div>
);

export default Home;