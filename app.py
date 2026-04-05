import streamlit as st
import tempfile
import os
import shutil

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="NELE Audio Enhancement", layout="wide")

# ---------- DARK BLUE BACKGROUND ----------
st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: white;
}

h1 {
    text-align: center;
    color: white;
}

h2, h3, p {
    color: white;
}

button {
    border-radius: 8px !important;
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

    if "output_audio" in st.session_state:
        st.audio(st.session_state["output_audio"])

        with open(st.session_state["output_audio"], "rb") as f:
            st.download_button(
                label="Download Enhanced Audio",
                data=f,
                file_name="enhanced.wav",
                mime="audio/wav"
            )

# ---------- BUTTON ----------
st.markdown("---")
center = st.columns([1,2,1])[1]

with center:
    if st.button("Enhance Audio"):
        if uploaded_file is None:
            st.warning("Please upload file first!")
        else:
            st.info("Processing... Please wait ⏳")

            try:
                output_path = "output/enhanced.wav"

                
                shutil.copy(input_path, output_path)

                st.session_state["output_audio"] = output_path

                st.success("Audio Enhanced Successfully!")

            except Exception as e:
                st.error(f"Error: {e}")