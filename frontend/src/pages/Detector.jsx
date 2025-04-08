import React, { useState } from 'react';
import VideoUploader from '../components/VideoUploader';
import LoadingSpinner from '../components/LoadingSpinner';
import SummarizedVideoDisplay from '../components/SummarizedVideoDisplay';
import axios from 'axios';
import '../index.css';

function Detector() {
  const [summarizedVideoUrl, setSummarizedVideoUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleDetect = async (video) => {
    setIsLoading(true);

    const formData = new FormData();
    formData.append('video', video);
    // formData.append('type', detectType);

    try {
      const response = await axios.post('http://127.0.0.1:5000/snap-video', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        responseType: 'blob',
      });

      const videoBlob = new Blob([response.data], { type: 'video/mp4' });
      const videoUrl = URL.createObjectURL(videoBlob);
      setSummarizedVideoUrl(videoUrl);
    } catch (error) {
      console.error('Error detecting:', error);
      alert('Error processing video.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = summarizedVideoUrl;
    link.download = 'summarized_video.mp4';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleCancel = () => {
    setSummarizedVideoUrl(null);
  };

  return (
    <div className="detector">
      {isLoading ? (
        <LoadingSpinner />
      ) : summarizedVideoUrl ? (
        <SummarizedVideoDisplay videoUrl={summarizedVideoUrl} onDownload={handleDownload} onCancel={handleCancel} />
      ) : (
        <VideoUploader onDetect={handleDetect} />
      )}
    </div>
  );
}

export default Detector;