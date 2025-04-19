import { supabase } from "../utils/SupabaseClient";

// Configurable
const DEFAULT_DAYS = 30;

export const convertOldVideos = async (days = DEFAULT_DAYS) => {
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - days);

  // 1. Fetch full videos
  const { data: fullVideos, error } = await supabase
    .from("video-data")
    .select("*")
    .eq("bucket_name", "full-videos");

  if (error) throw new Error("Error fetching full videos");

  const oldVideos = fullVideos.filter((video) => {
    const created = new Date(video.created_at);
    return created < cutoff;
  });

  if (oldVideos.length === 0) return "✅ No old videos found.";

  for (const video of oldVideos) {
    try {
      // 2. Get signed URL to download the file
      const { data: signedUrlData, error: urlError } = await supabase.storage
        .from("full-videos")
        .createSignedUrl(video.storage_path, 60);

      if (urlError) throw new Error("Signed URL error");

      const response = await fetch(signedUrlData.signedUrl);
      const blob = await response.blob();

      // 3. Summarise the video (🔧 Replace with real model)
      const summarisedBlob = await fakeSummariseVideo(blob); // replace this

      // 4. Upload to summarised bucket
      const newPath = `summary-${video.video_name}`;
      const { error: uploadError } = await supabase.storage
        .from("summarised-videos")
        .upload(newPath, summarisedBlob, {
          contentType: blob.type,
        });

      if (uploadError) throw new Error("Upload failed");

      // 5. Insert metadata
      await supabase.from("video-data").insert([
        {
          video_name: `summary-${video.video_name}`,
          storage_path: newPath,
          bucket_name: "summarised-videos",
          duration: video.duration,
          size: summarisedBlob.size,
        },
      ]);

      // 6. Delete original video
      await supabase.storage.from("full-videos").remove([video.storage_path]);
      await supabase.from("video-data").delete().eq("id", video.id);

      console.log(`✅ Converted ${video.video_name}`);
    } catch (err) {
      console.error(`❌ Failed to convert ${video.video_name}:`, err.message);
    }
  }

  return `🎉 Converted ${oldVideos.length} old videos.`;
};

// Placeholder summarization
async function fakeSummariseVideo(blob) {
  // In real implementation, you'd use a backend API call to a video summarizer
  console.log("⚙️ Simulating video summarization...");
  await new Promise((res) => setTimeout(res, 1000));
  return blob; // just pass the original for now
}
