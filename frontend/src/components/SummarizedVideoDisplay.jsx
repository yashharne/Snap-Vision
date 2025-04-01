import React from 'react';

function SummarizedVideoDisplay({ videoUrl, onDownload, onCancel }) {
  return (
    <div className="summarized-video-display">
      <video src={videoUrl} controls width="640" height="360" />
      <button onClick={onDownload}>Download</button>
      <button onClick={onCancel}>Cancel</button>
    </div>
  );
}

export default SummarizedVideoDisplay;