import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Home from "./pages/Home";
import Detector from "./pages/Detector";
import ModelInfo from "./pages/ModelInfo";
import About from "./pages/About";
import LoginSignup from "./pages/LoginSignup";
import History from "./pages/History";
import "./index.css";
import Database from "./pages/DataBase";
import Upload from "./pages/Upload";

function App() {
  return (
    <Router>
      <div className="app-container">
        {" "}
        {/* Wrap content in a flex container */}
        <Navbar />
        <div className="content">
          {" "}
          {/* Wrap page content */}
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/history" element={<History />} />
            <Route path="/detector" element={<Detector />} />
            <Route path="/modelInfo" element={<ModelInfo />} />
            <Route path="/db" element={<Database />} />
            <Route path="/about" element={<About />} />
            <Route path="/loginSignup" element={<LoginSignup />} />
            <Route path="/upload" element={<Upload />} />
          </Routes>
        </div>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
