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
    import os
    
    # --- 1. SET YOUR PATHS HERE ---
    # If the script is in the SAME folder as the dataset folders, just use the folder names:
    noisy_train_dir = r"E:\PROJECTS\nele_mlops\data\raw\clean_trainset_28spk_wav" 
    clean_train_dir = r"E:\PROJECTS\nele_mlops\data\raw\clean_trainset_28spk_wav"
    
    # We will create a "processed" folder right here to hold the outputs
    noisy_out_dir = "processed/noisy_train"
    clean_out_dir = "processed/clean_train"
    
    # --- 2. DEBUGGING / PATH CHECKER ---
    print(f"Current Working Directory: {os.getcwd()}")
    print("-" * 40)
    
    if not os.path.exists(noisy_train_dir):
        print(f" ERROR: Cannot find the folder '{noisy_train_dir}'")
        print("Make sure this script is running from the exact folder where your dataset is located, OR replace the path variables above with the full absolute path (e.g., 'C:/Users/.../noisy_trainset_28spk_wav').")
    else:
        print(f"Found noisy folder: {noisy_train_dir}")
        print("Starting processing... (This may take a few minutes)")
        
        # Run the extraction
        process_dataset(noisy_train_dir, noisy_out_dir)
        process_dataset(clean_train_dir, clean_out_dir)
        
        print(" Feature extraction complete!")