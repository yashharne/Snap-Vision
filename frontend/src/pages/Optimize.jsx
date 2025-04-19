import React, { useState } from "react";
import { convertOldVideos } from "../helpers/convertOldVideos";

function Optimize() {
  const [optimizeEnabled, setOptimizeEnabled] = useState(() => {
    return localStorage.getItem("optimizeEnabled") === "true";
  });

  const [optimizeDays, setOptimizeDays] = useState(30);
  const [convertStatus, setConvertStatus] = useState("");

  const handleToggleChange = () => {
    const newVal = !optimizeEnabled;
    setOptimizeEnabled(newVal);
    localStorage.setItem("optimizeEnabled", newVal);
  };

  const handleApplyOptimization = async () => {
    if (!optimizeEnabled) return;

    setConvertStatus("🔄 Optimizing...");
    try {
      const result = await convertOldVideos(optimizeDays);
      setConvertStatus(result);
    } catch (err) {
      setConvertStatus("❌ Failed: " + err.message);
    }
  };

  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        margin: "10rem",
      }}
    >
      <div
        style={{
          background: "rgba(255, 255, 255, 0.15)",
          backdropFilter: "blur(12px)",
          borderRadius: "16px",
          padding: "2rem",
          width: "400px",
          boxShadow: "0 8px 32px rgba(0, 0, 0, 0.25)",
          textAlign: "center",
        }}
      >
        <h2>⚙️ DB Storage Optimization</h2>

        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            margin: "2.5rem ",
          }}
        >
          <label style={{ fontSize: "1rem", fontWeight: 500 }}>
            Enable Optimization
          </label>
          <label
            style={{
              position: "relative",
              display: "inline-block",
              width: "52px",
              height: "28px",
            }}
          >
            <input
              type="checkbox"
              checked={optimizeEnabled}
              onChange={handleToggleChange}
              style={{ opacity: 0, width: 0, height: 0 }}
            />
            <span
              style={{
                position: "absolute",
                cursor: "pointer",
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                backgroundColor: optimizeEnabled ? "#4caf50" : "#ccc",
                transition: "0.4s",
                borderRadius: "34px",
              }}
            >
              <span
                style={{
                  position: "absolute",
                  height: "20px",
                  width: "20px",
                  left: optimizeEnabled ? "28px" : "4px",
                  bottom: "4px",
                  backgroundColor: "white",
                  transition: "0.4s",
                  borderRadius: "50%",
                }}
              />
            </span>
          </label>
        </div>

        <div
          style={{
            marginBottom: "1.5rem",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "0.5rem",
          }}
        >
          <label>Convert videos older than </label>
          <input
            type="number"
            value={optimizeDays}
            onChange={(e) => setOptimizeDays(Number(e.target.value))}
            style={{
              width: "60px",
              padding: "0.3rem",
              borderRadius: "6px",
              border: "1px solid #aaa",
              textAlign: "center",
            }}
          />
          <span>days</span>
        </div>

        <button
          onClick={handleApplyOptimization}
          disabled={!optimizeEnabled}
          style={{
            backgroundColor: optimizeEnabled ? "#4caf50" : "#aaa",
            color: "white",
            padding: "0.6rem 1.2rem",
            border: "none",
            borderRadius: "8px",
            cursor: optimizeEnabled ? "pointer" : "not-allowed",
            fontWeight: 500,
            marginTop: "2rem",
            transition: "background-color 0.3s",
          }}
          onMouseOver={(e) => {
            if (optimizeEnabled) e.target.style.backgroundColor = "#43a047";
          }}
          onMouseOut={(e) => {
            if (optimizeEnabled) e.target.style.backgroundColor = "#4caf50";
          }}
        >
          Apply Optimization
        </button>

        {convertStatus && (
          <p style={{ marginTop: "2rem", fontWeight: 500, color: "#fff" }}>
            {convertStatus}
          </p>
        )}
      </div>
    </div>
  );
}

export default Optimize;
