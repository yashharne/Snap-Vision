import React from 'react';
import '../index.css';

function History() {
    const historyData = [
        {
            id: 1,
            videoUrl: 'example-video-1.mp4', // Replace with actual URLs
            date: '2023-10-27',
            time: '14:30',
        },
        {
            id: 2,
            videoUrl: 'example-video-2.mp4',
            date: '2023-10-26',
            time: '11:15',
        },
        {
            id: 3,
            videoUrl: 'example-video-3.mp4', // Replace with actual URLs
            date: '2023-10-27',
            time: '14:30',
        },
        {
            id: 4,
            videoUrl: 'example-video-4.mp4',
            date: '2023-10-26',
            time: '11:15',
        },
        {
            id: 5,
            videoUrl: 'example-video-5.mp4', // Replace with actual URLs
            date: '2023-10-27',
            time: '14:30',
        },
        {
            id: 6,
            videoUrl: 'example-video-6.mp4',
            date: '2023-10-26',
            time: '11:15',
        },
        // Add more history data
    ];

    return (
        <div className="history-container">
            {historyData.map((item) => (
                <div key={item.id} className="history-card">
                    <h2>Processed Video {item.id}</h2>
                    <video src={item.videoUrl} controls />
                    <p><strong>Date:</strong> {item.date}</p>
                    <p><strong>Time:</strong> {item.time}</p>
                </div>
            ))}
        </div>
    );
}

export default History;