import React, { useState } from 'react';
import { ClipLoader } from 'react-spinners';
import '../index.css';

function VideoUploader({ onDetect }) {
  const [video, setVideo] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('yellow'); // yellow, green, red
  const [detectType, setDetectType] = useState('anomalies');
  const [loading, setLoading] = useState(false);

  const handleVideoChange = (event) => {
    const file = event.target.files[0];
    setVideo(file);
    setUploadStatus('green');
  };

  // const handleDetect = async () => {
  //   if (!video) {
  //     alert('Please upload a video first.');
  //     return;
  //   }
  //   setLoading(true);
  //   onDetect(video, detectType, () => setLoading(false));
  // };
  const handleDetect = async () => {
    if (!video) {
      alert('Please upload a video first.');
      return;
    }
    setLoading(true);
    try {
      await onDetect(video);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="video-uploader">
      <h1>Video Summarization & Detection Engine</h1>
      <input type="file" accept="video/*" onChange={handleVideoChange} />
      <div className={`status-light ${uploadStatus}`} />
      <select value={detectType} onChange={(e) => setDetectType(e.target.value)}>
        <option value="anomalies">Detect Anomalies</option>
        <option value="activities">Detect Activities</option>
      </select>
      <button onClick={handleDetect}>Start Detection</button>
      {loading && <ClipLoader color="#36D7B7" loading={loading} size={30} />}
    </div>
  );
}

export default VideoUploader;