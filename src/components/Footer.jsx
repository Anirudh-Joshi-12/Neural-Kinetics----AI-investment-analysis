// src/components/Footer.jsx
import { motion } from 'motion/react';
import { FOOTER_SUBTITLE, FOOTER_HEADING_LINE_1, FOOTER_HEADING_LINE_2, FOOTER_TAGS, EASE } from '../constants';

const Footer = () => (
  <motion.div
    className="footer"
    initial={{ y: 20, opacity: 0 }}
    animate={{ y: 0, opacity: 1 }}
    transition={{ duration: 1, delay: 0.5, ease: EASE }}
  >
    <div className="footer-inner">
      <div className="footer-left">
        <motion.div
          className="subtitle"
          initial={{ y: 16, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.6, ease: EASE }}
        >
          <span className="dot" aria-hidden="true" />
          <span>{FOOTER_SUBTITLE}</span>
        </motion.div>

        <motion.h1
          className="heading"
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.8, ease: EASE }}
        >
          {FOOTER_HEADING_LINE_1}<br />{FOOTER_HEADING_LINE_2}
        </motion.h1>

        <motion.div
          className="footer-buttons"
          initial={{ y: 16, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 1.0, ease: EASE }}
        >
          <button className="btn-primary">See Features</button>
          <button className="btn-secondary">How It Works</button>
        </motion.div>
      </div>

      <div className="footer-right">
        {FOOTER_TAGS.map((tag, i) => (
          <span key={i} className="tag-pill">{tag}</span>
        ))}
      </div>
    </div>
  </motion.div>
);

export default Footer;