// src/components/Navbar.jsx
import { motion } from 'motion/react';
import { Link } from 'react-router-dom';
import LogoIcon from './icons/LogoIcon';
import { BRAND_NAME, EASE } from '../constants';

const Navbar = () => (
  <motion.nav
    className="navbar"
    initial={{ y: -16, opacity: 0 }}
    animate={{ y: 0, opacity: 1 }}
    transition={{ duration: 0.8, ease: EASE }}
  >
    <div className="navbar-left">
      <Link to="/" className="brand" aria-label="Go to homepage">
        <LogoIcon />
        <span className="brand-name">{BRAND_NAME}</span>
      </Link>
    </div>

    <div className="navbar-right">
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.4, type: "spring", damping: 20, stiffness: 150 }}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <Link to="/reports" className="reports-btn">
          Reports
        </Link>
      </motion.div>
    </div>
  </motion.nav>
);

export default Navbar;