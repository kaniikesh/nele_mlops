import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
from google.colab import files
from IPython.display import Audio, display

# CONFIG
SR = 16000
N_FFT = 512
HOP_LENGTH = 128

# LOAD MODEL
model = tf.keras.models.load_model("final_model.h5")

# ==========================================
# ENHANCEMENT FUNCTION (FIXED)
# ==========================================
def enhance_audio(noisy_audio, model):
    noisy_stft = librosa.stft(noisy_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)

    mag = np.abs(noisy_stft)
    phase = np.angle(noisy_stft)

    log_mag = np.log1p(mag).T

    pred_log_mag = model.predict(np.expand_dims(log_mag, 0), verbose=0)[0]

    # Convert back
    enhanced_mag = np.expm1(pred_log_mag.T)

    # Clamp values
    enhanced_mag = np.maximum(enhanced_mag, 0)

    enhanced_audio = librosa.istft(
        enhanced_mag * np.exp(1j * phase),
        hop_length=HOP_LENGTH
    )

    # Normalize
    enhanced_audio = enhanced_audio / (np.max(np.abs(enhanced_audio)) + 1e-8)

    return enhanced_audio

# ==========================================
# UPLOAD AUDIO
# ==========================================
print("Upload noisy audio:")
uploaded = files.upload()

file_path = list(uploaded.keys())[0]

audio, _ = librosa.load(file_path, sr=SR)

# ==========================================
# ENHANCE
# ==========================================
enhanced = enhance_audio(audio, model)

# SAVE
sf.write("enhanced.wav", enhanced, SR)

# PLAY
print("Original:")
display(Audio(audio, rate=SR))

print("Enhanced:")
display(Audio(enhanced, rate=SR))

# DOWNLOAD
files.download("enhanced.wav")