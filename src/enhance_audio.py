import os
import numpy as np
import librosa
import soundfile as sf
import joblib

def enhance_audio(noisy_file_path, model_path, output_path):
    print(f"Loading trained model from: {model_path}...")
    model = joblib.load(model_path)
    
    print(f"Loading noisy audio: {noisy_file_path}")

    y, sr = librosa.load(noisy_file_path, sr=16000)
    
    # Feature Extraction (STFT)
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)
    
    # Reshape for the ML model (frames, frequency_bins)
    X = magnitude.T
    
    X = np.nan_to_num(X)
    
    print("Enhancing audio through the Ridge Regression model...")
    # Predict the clean spectrogram
    predicted_magnitude = model.predict(X)

    predicted_magnitude = np.maximum(predicted_magnitude, 0)
    
    # Reshape back to the audio format (frequency_bins, frames)
    predicted_magnitude = predicted_magnitude.T
    
    print("Reconstructing audio waveform (Griffin-Lim)...")

    # Reconstruct the time domain audio from the spectrogram
    enhanced_y = librosa.griffinlim(predicted_magnitude, n_fft=2048, hop_length=512)
    
    # Saving enhanced file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, enhanced_y, sr)
    print(f"Success! Enhanced audio saved to: {output_path}")

if __name__ == "__main__":
    import sys
    
    noisy_test_file = r"E:\PROJECTS\nele_mlops\data\raw\noisy_trainset_28spk_wav\p226_002.wav"
    
    model_file = r"models\baseline_ridge_model.pkl"
    output_file = r"output\enhanced_test.wav"

    if not os.path.exists(noisy_test_file):
        print(f" ERROR: Cannot find the test audio file: {noisy_test_file}")
        sys.exit()
        
    enhance_audio(noisy_test_file, model_file, output_file)