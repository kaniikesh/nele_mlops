import streamlit as st
import tempfile
import os
import shutil

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="NELE Audio Enhancement", layout="wide")

# ---------- DARK BLUE BACKGROUND ----------
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #0f172a;
}

/* Text */
h1, h2, h3, p {
    color: white;
}

/* Buttons (FIXED) */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
}

/* Light mode fix */
@media (prefers-color-scheme: light) {
    .stApp {
        background-color: #f9fafb;
    }
    h1, h2, h3, p {
        color: black;
    }
}

</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("<h1> Near-End Listening Enhancement</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload a noisy audio file and enhance it using Machine Learning</p>", unsafe_allow_html=True)

# ---------- CREATE OUTPUT FOLDER ----------
os.makedirs("output", exist_ok=True)

# ---------- LAYOUT ----------
col1, col2 = st.columns(2)

# ---------- LEFT SIDE (UPLOAD) ----------
with col1:
    st.subheader("🔊 Noisy Audio (Input)")

    uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])

    input_path = None

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            input_path = tmp.name

        st.audio(input_path)

# ---------- RIGHT SIDE (OUTPUT) ----------
with col2:
    st.subheader("🎧 Enhanced Audio (Output)")

    if "audio_bytes" in st.session_state:
        st.audio(st.session_state["audio_bytes"])

        st.download_button(
            label="Download Enhanced Audio",
            data=st.session_state["audio_bytes"],
            file_name="enhanced.wav",
            mime="audio/wav"
        )

# ---------- BUTTON ----------
import io

if enhance_clicked:
    if uploaded_file is None:
        st.warning("Please upload file first!")
    else:
        st.info("Processing... Please wait ⏳")

        try:
            # Read uploaded audio bytes
            audio_bytes = uploaded_file.read()

            # TEMPORARY PROCESS 
            enhanced_bytes = audio_bytes

            # Store in session
            st.session_state["audio_bytes"] = enhanced_bytes

            st.success("Audio Enhanced Successfully!")

        except Exception as e:
            st.error(f"Error: {e}")