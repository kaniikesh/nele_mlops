import librosa
import numpy as np

def calculate_signal_power(signal):
    """Calculates the average power of a 1D audio signal."""
    return np.mean(signal ** 2)

def mix_audio(clean_path, noise_path, target_snr_db, sample_rate=16000):
    """
    Mixes a clean voice and a background noise file at a specific SNR.
    """
    # 1. Load the audio files (resampling to 16kHz)
    clean_audio, _ = librosa.load(clean_path, sr=sample_rate)
    noise_audio, _ = librosa.load(noise_path, sr=sample_rate)

    # 2. Match the lengths (loop the noise if it's too short, crop if too long)
    if len(noise_audio) < len(clean_audio):
        pad_length = len(clean_audio) - len(noise_audio)
        noise_audio = np.pad(noise_audio, (0, pad_length), mode='wrap')
    
    noise_audio = noise_audio[:len(clean_audio)]

    # 3. Calculate signal power
    clean_power = calculate_signal_power(clean_audio)
    noise_power = calculate_signal_power(noise_audio)

    # Prevent division by zero errors on silent files
    if noise_power == 0:
        return clean_audio, clean_audio, noise_audio 

    # 4. Scale the noise to hit the target Decibel (dB) ratio
    required_noise_power = clean_power / (10 ** (target_snr_db / 10))
    scaling_factor = np.sqrt(required_noise_power / noise_power)

    # 5. Mix them together
    scaled_noise = noise_audio * scaling_factor
    noisy_mix = clean_audio + scaled_noise

    return noisy_mix, clean_audio, scaled_noise

def extract_features_for_ml(audio_array, sample_rate=16000):
    """
    Converts a 1D audio waveform into a 2D Spectrogram (frequency over time).
    """
    # Compute the Short-Time Fourier Transform (STFT)
    stft = librosa.stft(audio_array, n_fft=512, hop_length=256)
    
    # Extract the magnitude (volume) of the frequencies
    magnitude = np.abs(stft)
    
    # Convert to Decibel scale for the neural network
    spectrogram_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # --- THE FIX ---
    # Crop the frequency dimension from 257 to 256 so it divides evenly in the CNN
    spectrogram_db = spectrogram_db[:256, :]
    
    return spectrogram_db