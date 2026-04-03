import React, { useState } from "react";
import "./App.css";
import headphones from "./headphones.png";

function App() {
  const [file, setFile] = useState(null);
  const [audioURL, setAudioURL] = useState("");
  const [enhancedURL, setEnhancedURL] = useState("");

  const handleUpload = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setAudioURL(URL.createObjectURL(selectedFile));
  };

  const handleEnhance = async () => {
    // TEMP: simulate output (replace later with backend)
    setEnhancedURL(audioURL);
  };

  return (
    <div className="container">
      <h1 className="title">Near-End Listening Enhancement</h1>

      <div className="main">
        
        {/* LEFT SIDE */}
        <div className="side">
          <h3>Noisy Audio (Input)</h3>

          <input type="file" accept=".wav" onChange={handleUpload} />

          {audioURL && (
            <audio controls src={audioURL}></audio>
          )}
        </div>

        {/* CENTER HEADPHONE */}
        <div className="center">
  <img src={headphones} alt="headphones" />

  <div className="enhance-btn">
    <button onClick={handleEnhance}>Enhance →</button>
  </div>
</div>

        {/* RIGHT SIDE */}
        <div className="side">
          <h3>Enhanced Audio (Output)</h3>

          {enhancedURL && (
            <>
              <audio controls src={enhancedURL}></audio>
              <a href={enhancedURL} download="enhanced.wav">
                <button>Download</button>
              </a>
            </>
          )}
        </div>

      </div>
    </div>
  );
}

export default App;