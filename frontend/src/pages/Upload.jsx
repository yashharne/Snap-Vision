import React, { useState } from "react";
import { supabase } from "../utils/SupabaseClient";
import {
  getVideoDuration,
  formatDuration,
  formatFileSize,
} from "../helpers/getVideoInfo";

function UploadVideo() {
  const [videoFile, setVideoFile] = useState(null);
  const [videoType, setVideoType] = useState("full"); // 'full' or 'summarised'
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState("");

  const handleUpload = async () => {
    if (!videoFile) {
      setStatus("❌ Please select a video to upload.");
      return;
    }

    setUploading(true);
    setStatus("🔄 Uploading...");

    // Get file details
    const fileExt = videoFile.name.split(".").pop();
    const fileName = `${Date.now()}.${fileExt}`;
    const bucketName =
      videoType === "full" ? "full-videos" : "summarised-videos";

    const videoSize = formatFileSize(videoFile.size); // Format the file size

    // Step 1: Upload to correct bucket
    const { error: uploadError } = await supabase.storage
      .from(bucketName)
      .upload(fileName, videoFile);

    if (uploadError) {
      setStatus(`❌ Upload failed: ${uploadError.message}`);
      setUploading(false);
      return;
    }

    // Step 2: Get duration from the video
    const durationInSeconds = await getVideoDuration(videoFile);
    const formattedDuration = formatDuration(durationInSeconds);

    // Step 3: Insert metadata into 'video-data' table
    const { error: dbError } = await supabase.from("video-data").insert([
      {
        video_name: videoFile.name,
        bucket_name: bucketName,
        storage_path: fileName,
        duration: formattedDuration,
        size: videoSize, // Add size metadata
      },
    ]);

    if (dbError) {
      setStatus(
        `⚠️ Uploaded to bucket, but DB insert failed: ${dbError.message}`
      );
    } else {
      setStatus("✅ Upload successful and metadata saved!");
    }

    setUploading(false);
    setVideoFile(null);
  };

  return (
    <div className="upload-form">
      <h2>📤 Upload Video</h2>

      <input
        type="file"
        accept="video/*"
        onChange={(e) => setVideoFile(e.target.files[0])}
        disabled={uploading}
      />

      <div style={{ marginTop: "1rem" }}>
        <label>
          <input
            type="radio"
            name="videoType"
            value="full"
            checked={videoType === "full"}
            onChange={() => setVideoType("full")}
            disabled={uploading}
          />
          Full Video
        </label>
        <label style={{ marginLeft: "1rem" }}>
          <input
            type="radio"
            name="videoType"
            value="summarised"
            checked={videoType === "summarised"}
            onChange={() => setVideoType("summarised")}
            disabled={uploading}
          />
          Summarised Video
        </label>
      </div>

      <button
        onClick={handleUpload}
        disabled={uploading}
        style={{ marginTop: "1rem" }}
      >
        {uploading ? "Uploading..." : "Upload"}
      </button>

      <p style={{ marginTop: "1rem", color: "green" }}>{status}</p>
    </div>
  );
}

export default UploadVideo;
