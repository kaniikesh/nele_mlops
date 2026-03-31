import streamlit as st
import numpy as np
import librosa
import torch
import soundfile as sf
import os
from model import AudioEnhancementCNN

# --- Page Configuration ---
st.set_page_config(page_title="Audio Enhancer AI", layout="centered")
st.title("🎙️ Near-End Listening Enhancer")
st.write("Upload a noisy audio clip, and our PyTorch model will enhance the voice clarity.")

def process_audio(file_path):
    """Handles the heavy lifting: STFT -> Model -> Inverse STFT"""
    # 1. Load the original audio
    noisy_audio, sr = librosa.load(file_path, sr=16000)
    
    # 2. Extract Spectrogram and Phase
    stft = librosa.stft(noisy_audio, n_fft=512, hop_length=256)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Convert to DB and crop to 256 for the CNN
    spectrogram_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    input_db = spectrogram_db[:256, :] 
    
    # --- THE FIX: PADDING ---
    original_time_frames = input_db.shape[1]
    
    # Calculate how many frames we need to add to make it a multiple of 4
    pad_amount = (4 - (original_time_frames % 4)) % 4 
    
    if pad_amount > 0:
        # Pad the time dimension with zeros (silence)
        input_db = np.pad(input_db, ((0, 0), (0, pad_amount)), mode='constant')
    
    # 3. Prepare tensor for the model [1, 1, 256, Time]
    input_tensor = torch.FloatTensor(input_db).unsqueeze(0).unsqueeze(0)
    
    # 4. LOAD THE RAW PYTORCH MODEL DIRECTLY
    model = AudioEnhancementCNN()
    try:
        model.load_state_dict(torch.load("audio_model_weights.pth", weights_only=True))
        model.eval() 
    except FileNotFoundError:
        st.error("Could not find 'audio_model_weights.pth'. Please run train.py first!")
        return None, sr
    
    # 5. Run Inference! Get the mask from the CNN
    with torch.no_grad():
        mask_tensor = model(input_tensor)
        mask = mask_tensor.squeeze().numpy() 
        
    # --- THE FIX: CROPPING ---
    # Shave off the extra padding we added so it perfectly matches the original audio shape
    mask = mask[:, :original_time_frames]
        
    # 6. Apply the Mask to the original audio magnitude
    enhanced_magnitude = magnitude[:256, :] * mask
    
    # Add back the top frequency bin we cropped out (fill with zeros)
    enhanced_magnitude = np.pad(enhanced_magnitude, ((0, 1), (0, 0)), mode='constant')
    
    # 7. Reconstruct the Audio (Inverse STFT)
    enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
    enhanced_audio = librosa.istft(enhanced_stft, hop_length=256)
    
    return enhanced_audio, sr

    
    # 5. Run Inference! Get the mask from the CNN
    with torch.no_grad():
        mask_tensor = model(input_tensor)
        mask = mask_tensor.squeeze().numpy() # Convert back to numpy array
        
    # 6. Apply the Mask to the original audio magnitude
    enhanced_magnitude = magnitude[:256, :] * mask
    
    # Add back the top frequency bin we cropped out (fill with zeros)
    enhanced_magnitude = np.pad(enhanced_magnitude, ((0, 1), (0, 0)), mode='constant')
    
    # 7. Reconstruct the Audio (Inverse STFT)
    enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
    enhanced_audio = librosa.istft(enhanced_stft, hop_length=256)
    
    return enhanced_audio, sr

# --- File Uploader UI ---
uploaded_file = st.file_uploader("Upload Noisy Audio (WAV/MP3/FLAC)", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    st.subheader("1. Original Noisy Audio")
    st.audio(uploaded_file)
    
    if st.button("Enhance Audio (Run Inference)"):
        with st.spinner('Loading PyTorch model and processing audio...'):
            
            # Save uploaded file temporarily
            temp_path = "temp_upload.wav"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Run our pipeline
            enhanced_audio, sample_rate = process_audio(temp_path)
            
            if enhanced_audio is not None:
                # Save output to a temporary file for the web player
                output_path = "temp_enhanced.wav"
                sf.write(output_path, enhanced_audio, sample_rate)
                
                st.success("Enhancement Complete!")
                
                st.subheader("2. Enhanced Audio Output")
                st.audio(output_path)
                
                # Cleanup temporary files
                if os.path.exists(temp_path):
                    os.remove(temp_path)