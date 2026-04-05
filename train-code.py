# ==========================================
# 0. INSTALL & IMPORTS
# ==========================================
#!pip install librosa soundfile kagglehub tqdm -q

import os
import random
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
import kagglehub
from tqdm import tqdm

# ==========================================
# 1. CONFIG
# ==========================================
SR = 16000
N_FFT = 512
HOP_LENGTH = 128
N_BINS = N_FFT // 2 + 1

TARGET_SECONDS = 2
TARGET_SAMPLES = SR * TARGET_SECONDS
FRAMES = int(np.ceil(TARGET_SAMPLES / HOP_LENGTH))

NUM_SAMPLES = 2000  

# ==========================================
# 2. DATA
# ==========================================
musan_path = kagglehub.dataset_download("nhattruongdev/musan-noise")
libri_path = kagglehub.dataset_download("pypiahmad/librispeech-asr-corpus")

def get_audio_files(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith(".wav") or f.endswith(".flac"):
                files.append(os.path.join(root, f))
    return files

clean_files = get_audio_files(libri_path)
noise_files = get_audio_files(musan_path)

# ==========================================
# 3. FEATURE EXTRACTION (IMPROVED)
# ==========================================
def extract_features(clean, noise):
    snr = random.uniform(0, 10)

    clean_power = np.mean(clean**2)
    noise_power = np.mean(noise**2)

    if noise_power > 0:
        noise_scaler = np.sqrt((clean_power / (10**(snr/10))) / noise_power)
        noisy = clean + noise * noise_scaler
    else:
        noisy = clean

    clean_stft = librosa.stft(clean, n_fft=N_FFT, hop_length=HOP_LENGTH)
    noisy_stft = librosa.stft(noisy, n_fft=N_FFT, hop_length=HOP_LENGTH)

    clean_mag = np.abs(clean_stft)
    noisy_mag = np.abs(noisy_stft)

    
    log_noisy = np.log1p(noisy_mag)
    log_clean = np.log1p(clean_mag)

    return log_noisy.T, log_clean.T

# ==========================================
# 4. DATASET
# ==========================================
X, Y = [], []

print("Preparing data...")

for _ in tqdm(range(NUM_SAMPLES)):
    clean, _ = librosa.load(random.choice(clean_files), sr=SR)
    noise, _ = librosa.load(random.choice(noise_files), sr=SR)

    clean = clean[:TARGET_SAMPLES] if len(clean) > TARGET_SAMPLES else np.pad(clean, (0, TARGET_SAMPLES - len(clean)))
    noise = noise[:TARGET_SAMPLES] if len(noise) > TARGET_SAMPLES else np.pad(noise, (0, TARGET_SAMPLES - len(noise)))

    features, target = extract_features(clean, noise)

    if features.shape[0] >= FRAMES:
        X.append(features[:FRAMES])
        Y.append(target[:FRAMES])

X = np.array(X)
Y = np.array(Y)

print("Shape:", X.shape)

# ==========================================
# 5. BiLSTM MODEL (STRONGER)
# ==========================================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(FRAMES, N_BINS)),

    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True)
    ),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True)
    ),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(N_BINS)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.Huber()  
)

model.summary()

# ==========================================
# 6. TRAIN
# ==========================================
model.fit(
    X, Y,
    epochs=25, 
    batch_size=32,
    validation_split=0.1
)

# ==========================================
# 7. SAVE (FIXED VERSION)
# ==========================================

# Save only weights
model.save_weights("model_weights.h5")

# Rebuild clean model (no extra config)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Input

model_fixed = Sequential([
    Input(shape=(FRAMES, N_BINS)),

    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),

    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),

    Dense(N_BINS)
])

# Load weights
model_fixed.load_weights("model_weights.h5")

# Save clean model
model.save("final_model.h5", include_optimizer=False)

print("Model saved successfully!")
