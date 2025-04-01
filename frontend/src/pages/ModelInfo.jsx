import React from 'react';
// import modelDiagram from '../assets/model_diagram.png';
import '../index.css';

function ModelInfo() {
  return (
    <div className="model-info">
      <h1>How SnapVision Works</h1>
      {/* <img src={modelDiagram} alt="Model Diagram" /> */}
      <p>Our AI model analyzes video footage to detect anomalies and activities.</p>
    </div>
  );
}

export default ModelInfo;