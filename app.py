import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tempfile
import tensorflow as tf
import os

# ==========================================
# CONFIG
# ==========================================
SR = 16000
N_FFT = 512
HOP_LENGTH = 128

st.set_page_config(page_title="NELE Audio Enhancement", layout="wide")

# ==========================================
# STYLE
# ==========================================
st.markdown("""
<style>
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL (WITH STREAMLIT LFS FIX)
# ==========================================
@st.cache_resource
def load_model():
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Input
    
    # Re-declare your shapes from Colab
    FRAMES = 250 
    N_BINS = 257
    
    # 1. Build the exact same architecture
    model = Sequential([
        Input(shape=(FRAMES, N_BINS)),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Dense(N_BINS)
    ])
    
    weights_path = "models/model.weights.h5"
    
    # 2. Check if the file is missing OR if it's a tiny Git LFS text pointer (< 1000 bytes)
    if not os.path.exists(weights_path) or os.path.getsize(weights_path) < 1000:
        st.warning("Downloading heavy model weights... this will only happen once ⏳")
        
        # Direct raw link to the weights file on your GitHub deployment branch
        raw_github_url = "https://github.com/kaniikesh/nele_mlops/raw/deployment/models/model.weights.h5"
        
        # Download the true binary file
        weights_path = tf.keras.utils.get_file("model_weights_real.h5", origin=raw_github_url)
        
    model.load_weights(weights_path)
    return model

model = load_model()

# ==========================================
# AUDIO ENHANCEMENT FUNCTION
# ==========================================
def enhance_audio_bytes(audio_bytes):
    # Save temp input
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        input_path = tmp.name

    # Load audio
    audio, _ = librosa.load(input_path, sr=SR)

    # STFT
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag = np.abs(stft)
    phase = np.angle(stft)

    log_mag = np.log1p(mag).T

    # Model prediction
    pred = model.predict(np.expand_dims(log_mag, 0), verbose=0)[0]

    # Reconstruct
    enhanced_mag = np.expm1(pred.T)
    enhanced_mag = np.maximum(enhanced_mag, 0)

    enhanced_audio = librosa.istft(
        enhanced_mag * np.exp(1j * phase),
        hop_length=HOP_LENGTH
    )

    # Normalize
    enhanced_audio = enhanced_audio / (np.max(np.abs(enhanced_audio)) + 1e-8)

    # Save output
    output_path = input_path.replace(".wav", "_enhanced.wav")
    sf.write(output_path, enhanced_audio, SR)

    # Return bytes
    with open(output_path, "rb") as f:
        return f.read()

# ==========================================
# UI
# ==========================================
st.title("🎧 Near-End Listening Enhancement")
st.markdown("Upload a noisy audio file and enhance it")

col1, col2 = st.columns(2)

# ---------- INPUT ----------
with col1:
    st.subheader("🔊 Input Audio")
    uploaded_file = st.file_uploader("Upload WAV", type=["wav"])

    if uploaded_file is not None:
        input_bytes = uploaded_file.getvalue()
        st.audio(input_bytes)

# ---------- BUTTON ----------
st.markdown("---")
col_center = st.columns([1,2,1])[1]

with col_center:
    enhance_clicked = st.button("🚀 Enhance Audio")

# ---------- PROCESS ----------
# We run the processing logic BEFORE rendering the output column!
if enhance_clicked:
    if uploaded_file is None:
        st.warning("Please upload a file first!")
    else:
        with st.spinner("Processing... Please wait ⏳"):
            try:
                enhanced_bytes = enhance_audio_bytes(input_bytes)
                st.session_state["enhanced_audio"] = enhanced_bytes
                st.success("Enhancement complete!")
            except Exception as e:
                st.error(f"Error: {e}")

# ---------- OUTPUT ----------
# Because this is down here, it can instantly see the new session_state!
with col2:
    st.subheader("🎧 Enhanced Audio")

    if "enhanced_audio" in st.session_state:
        st.audio(st.session_state["enhanced_audio"])

        st.download_button(
            "Download Enhanced Audio",
            data=st.session_state["enhanced_audio"],
            file_name="enhanced.wav",
            mime="audio/wav"
        )