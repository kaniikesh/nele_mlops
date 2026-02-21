import soundfile as sf
from pystoi import stoi
import numpy as np

def evaluate_audio(clean_file_path, enhanced_file_path, noisy_file_path=None):
    print("Loading audio files...")
    # Using soundfile directly instead of librosa is much faster
    clean_y, sr_clean = sf.read(clean_file_path)
    enhanced_y, sr_enh = sf.read(enhanced_file_path)
    
    # Optional: If your audio is stereo (2D array), convert to mono
    if len(clean_y.shape) > 1: clean_y = clean_y.mean(axis=1)
    if len(enhanced_y.shape) > 1: enhanced_y = enhanced_y.mean(axis=1)

    # Ensure both audio arrays are the exact same length
    min_len = min(len(clean_y), len(enhanced_y))
    clean_y = clean_y[:min_len]
    enhanced_y = enhanced_y[:min_len]

    print("\n--- Calculating Metric ---")
    
    # Calculate STOI (Intelligibility)
    # Note: We assume the sample rate is 16000 based on your previous scripts
    stoi_score = stoi(clean_y, enhanced_y, 16000, extended=False)
    print(f"STOI Score: {stoi_score:.4f} (Range: 0.0 to 1.0, Higher is better)")

if __name__ == "__main__":
    import os
    
    # --- SET YOUR PATHS ---
    # Put your absolute paths here, just like you did in the extraction script
    clean_reference = r"E:\PROJECTS\nele_mlops\data\raw\clean_trainset_28spk_wav\p226_002.wav" 
    model_output = r"output\enhanced_test.wav"
    noisy_input = r"E:\PROJECTS\nele_mlops\data\raw\noisy_trainset_28spk_wav\p226_002.wav"
    
    if os.path.exists(clean_reference) and os.path.exists(model_output):
        evaluate_audio(clean_reference, model_output, noisy_input)
    else:
        print("❌ ERROR: Double check your file paths.")