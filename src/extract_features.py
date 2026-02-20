import os
import numpy as np
import librosa
from pathlib import Path

def process_audio(file_path, n_fft=2048, hop_length=512):
    """Loads audio and extracts the magnitude spectrogram."""
    # Load audio at a consistent 16kHz sample rate
    y, sr = librosa.load(file_path, sr=16000)
    
    # Compute Short-Time Fourier Transform (STFT)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    
    # Extract magnitude (we only need the magnitude to train the regressor)
    magnitude = np.abs(stft)
    
    return magnitude

def process_dataset(input_dir, output_dir):
    """Processes all .wav files in a directory and saves features as .npy arrays."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for audio_file in input_path.rglob("*.wav"):
        print(f"Processing: {audio_file.name}")
        mag_spec = process_audio(audio_file)
        
        # Save as a numpy array for fast loading during training
        save_name = output_path / f"{audio_file.stem}_mag.npy"
        np.save(save_name, mag_spec)

if __name__ == "__main__":
    # Define your paths (update these once you download VoiceBank-DEMAND)
    noisy_train_dir = "../data/raw/noisy_train"
    clean_train_dir = "../data/raw/clean_train"
    
    noisy_out_dir = "../data/processed/noisy_train"
    clean_out_dir = "../data/processed/clean_train"
    
    print("Feature extraction pipeline initialized.")
    # Uncomment below to run the extraction once data is in place
    # process_dataset(noisy_train_dir, noisy_out_dir)
    # process_dataset(clean_train_dir, clean_out_dir)