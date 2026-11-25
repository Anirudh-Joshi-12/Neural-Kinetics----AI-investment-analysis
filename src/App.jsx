// src/App.jsx
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import BackgroundVideo from './components/BackgroundVideo';
import Footer from './components/Footer';
import Home from './pages/Home';
import Upload from './pages/Upload';
import Reports from './pages/Reports';
import Analysis from './pages/Analysis';
import About from './pages/About';
import './index.css';

function App() {
  return (
    <BrowserRouter>
      <div className="page">
        <Navbar />
        <BackgroundVideo />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/analysis/:id" element={<Analysis />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  );
}

export default App;