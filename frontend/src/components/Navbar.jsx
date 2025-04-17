import React, { useState } from "react";
import { Link } from "react-router-dom";
import { FaVideo, FaBars, FaTimes } from "react-icons/fa";
import { LuCctv } from "react-icons/lu";
import "../index.css";

function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <Link to="/" className="navbar-logo">
          <LuCctv className="navbar-icon" />
          SnapVision
        </Link>
        <div className="menu-icon" onClick={toggleMenu}>
          {isMenuOpen ? <FaTimes /> : <FaBars />}
        </div>
        <ul className={`nav-menu ${isMenuOpen ? "active" : ""}`}>
          <li className="nav-item">
            <Link
              to="/history"
              className="nav-links"
              onClick={() => setIsMenuOpen(false)}
            >
              History
            </Link>
          </li>
          <li className="nav-item">
            <Link
              to="/detector"
              className="nav-links"
              onClick={() => setIsMenuOpen(false)}
            >
              Detection Engine
            </Link>
          </li>
          <li className="nav-item">
            <Link
              to="/modelInfo"
              className="nav-links"
              onClick={() => setIsMenuOpen(false)}
            >
              Model Info
            </Link>
          </li>
          <li className="nav-item">
            <Link
              to="/db"
              className="nav-links"
              onClick={() => setIsMenuOpen(false)}
            >
              Database
            </Link>
          </li>
          <li className="nav-item">
            <Link
              to="/about"
              className="nav-links"
              onClick={() => setIsMenuOpen(false)}
            >
              About Us
            </Link>
          </li>
          <li className="nav-item">
            <Link
              to="/loginSignup"
              className="nav-links"
              onClick={() => setIsMenuOpen(false)}
            >
              Login/Signup
            </Link>
          </li>
        </ul>
      </div>
    </nav>
  );
}

export default Navbar;
