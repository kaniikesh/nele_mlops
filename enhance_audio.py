import os
import numpy as np
import librosa
import soundfile as sf
import joblib

def enhance_audio(noisy_file_path, model_path, output_path):
    print("Loading model...")
    model = joblib.load(model_path)

    print("Loading audio...")
    y, sr = librosa.load(noisy_file_path, sr=16000)

    # STFT
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)

    # Prepare input
    X = magnitude.T
    X = np.nan_to_num(X)

    print("Predicting...")
    predicted_magnitude = model.predict(X)

    predicted_magnitude = np.maximum(predicted_magnitude, 0)
    predicted_magnitude = predicted_magnitude.T

    print("Reconstructing...")
    enhanced_y = librosa.griffinlim(predicted_magnitude, n_fft=2048, hop_length=512)

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, enhanced_y, sr)

    print("Done!")