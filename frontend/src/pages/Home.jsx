import React from 'react';
import { Link } from 'react-router-dom';
import '../index.css';

function Home() {
  return (
    <div className="home">
      <h1>Welcome to SnapVision</h1>
      <p className="home-description">
        Unlock the power of AI to transform your video surveillance. Effortlessly detect anomalies and suspicious activities, saving you time and ensuring maximum security. Experience the future of intelligent video analysis with SnapVision.
      </p>
      <Link to="/detector" className="get-started-button">
        Get Started
      </Link>
    </div>
  );
}

export default Home;