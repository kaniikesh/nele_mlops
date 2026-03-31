import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
import numpy as np

# Import the mixing functions we wrote in the previous step
# (Make sure your previous script is saved as mix_audio.py)
from mix_audio import mix_audio, extract_features_for_ml

class AudioEnhancementDataset(Dataset):
    def __init__(self, clean_dir, noise_dir, target_length_frames=128):
        """
        Scans the directories for audio files and sets up the dataset.
        """
        self.clean_dir = Path(clean_dir)
        self.noise_dir = Path(noise_dir)
        self.target_length = target_length_frames
        
        # Recursively find all .flac (LibriSpeech) and .wav (ESC-50/TBUS) files
        self.clean_files = list(self.clean_dir.rglob('*.flac')) + list(self.clean_dir.rglob('*.wav'))
        self.noise_files = list(self.noise_dir.rglob('*.wav'))
        
        print(f"Found {len(self.clean_files)} clean files and {len(self.noise_files)} noise files.")

    def __len__(self):
        # The length of an epoch is determined by how many clean speech files we have
        return len(self.clean_files)

    def __getitem__(self, idx):
        """
        This function is called automatically by PyTorch for every single training step.
        """
        clean_file = self.clean_files[idx]
        
        # 1. Pick a random noise file for this specific clean voice
        noise_file = random.choice(self.noise_files)
        
        # 2. Pick a random SNR (Signal-to-Noise Ratio) between 0dB (hard) and 10dB (easier)
        # This forces the model to learn how to handle different volume levels of noise
        random_snr = random.uniform(0, 10)
        
        # 3. Mix the audio using our previous script
        noisy_waveform, clean_waveform, _ = mix_audio(clean_file, noise_file, random_snr)
        
        # 4. Convert audio waveforms into 2D Spectrograms
        noisy_spec = extract_features_for_ml(noisy_waveform)
        clean_spec = extract_features_for_ml(clean_waveform)
        
        # 5. Crop or Pad the spectrograms to ensure they are all the exact same size 
        # (Neural networks require fixed-size inputs. E.g., 128x128)
        noisy_spec = self._pad_or_crop(noisy_spec, self.target_length)
        clean_spec = self._pad_or_crop(clean_spec, self.target_length)
        
        # 6. Convert numpy arrays to PyTorch Tensors and add a "Channel" dimension
        # PyTorch expects shape: [Channels, Height, Width] -> [1, Freq, Time]
        X_tensor = torch.FloatTensor(noisy_spec).unsqueeze(0)
        Y_tensor = torch.FloatTensor(clean_spec).unsqueeze(0)
        
        return X_tensor, Y_tensor

    def _pad_or_crop(self, spectrogram, target_length):
        """Helper function to ensure all spectrograms are the same time length."""
        current_length = spectrogram.shape[1]
        if current_length > target_length:
            # Crop it
            return spectrogram[:, :target_length]
        elif current_length < target_length:
            # Pad it with zeros
            pad_width = target_length - current_length
            return np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
        return spectrogram

# ==========================================
# Test the DataLoader
# ==========================================
if __name__ == "__main__":
    # Point these to the folders you showed in your screenshots
    CLEAN_PATH = "data/clean_speech"
    NOISE_PATH = "data/noise"
    
    # Initialize the Dataset
    my_dataset = AudioEnhancementDataset(clean_dir=CLEAN_PATH, noise_dir=NOISE_PATH)
    
    # Initialize the DataLoader (Groups data into batches of 8, shuffles them, and uses 2 background workers)
    my_dataloader = DataLoader(my_dataset, batch_size=8, shuffle=True, num_workers=2)
    
    # Fetch one batch to see if it works
    noisy_batch, clean_batch = next(iter(my_dataloader))
    
    print("\nDataLoader Success!")
    print(f"Noisy Batch Shape: {noisy_batch.shape}") # Should be [8, 1, Frequency_Bins, 128]
    print(f"Clean Batch Shape: {clean_batch.shape}")