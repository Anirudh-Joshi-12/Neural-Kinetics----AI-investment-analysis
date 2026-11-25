// src/components/BackgroundVideo.jsx
import { motion } from 'motion/react';
import { VIDEO_URL, EASE } from '../constants';

const BackgroundVideo = () => (
  <motion.div
    className="video-layer"
    initial={{ opacity: 0, scale: 1.05 }}
    animate={{ opacity: 1, scale: 1 }}
    transition={{ duration: 1.8, ease: EASE }}
  >
    <div className="video-wrapper">
      <video
        className="video-el"
        src={VIDEO_URL}
        autoPlay
        muted
        playsInline
        preload="metadata"
      />
    </div>
  </motion.div>
);

export default BackgroundVideo;