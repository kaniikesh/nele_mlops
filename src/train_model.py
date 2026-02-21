import os
import numpy as np
from sklearn.linear_model import Ridge
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def load_data_subset(noisy_dir, clean_dir, max_files=None, keep_ratio=0.15):
    """Loads the extracted .npy spectrograms but only keeps a random subset of frames to save RAM."""
    noisy_path = Path(noisy_dir)
    clean_path = Path(clean_dir)
    
    X_list = [] 
    y_list = [] 
    
    noisy_files = list(noisy_path.glob("*.npy"))
    if max_files is not None:
        noisy_files = noisy_files[:max_files]
        
    total_files = len(noisy_files)
    print(f"Loading {total_files} files. Keeping {keep_ratio*100}% of frames to save RAM...")
    
    for i, noisy_file in enumerate(noisy_files):
        if i % 1000 == 0 and i > 0:
            print(f"Processed {i}/{total_files} files...")
            
        clean_file = clean_path / noisy_file.name
        if clean_file.exists():
            noisy_mag = np.load(noisy_file).T
            clean_mag = np.load(clean_file).T
            
            # --- RAM SAVER: Randomly sample frames ---
            num_frames = noisy_mag.shape[0]
            sample_size = max(1, int(num_frames * keep_ratio)) 
            
            # Pick random row indices without replacement
            indices = np.random.choice(num_frames, sample_size, replace=False)
            
            # Only append the randomly selected frames
            X_list.append(noisy_mag[indices])
            y_list.append(clean_mag[indices])
            
    print("Stacking matrices... (This should now easily fit in memory)")
    X = np.vstack(X_list)
    y = np.vstack(y_list)
    
    return X, y

def train_baseline_model(X, y, model_save_path):
    """Trains a fast regression model and saves it to disk."""
    print(f"Data shape: {X.shape[0]} time frames, {X.shape[1]} frequency bins.")
    print("Scaling data and training Ridge Regression baseline model...")
    
    # NEW: Create a pipeline that scales the data FIRST, then applies Ridge
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    
    # Train the model
    model.fit(X, y)
    print("Training complete!")
    
    # Save the model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"Model saved successfully to {model_save_path}")

if __name__ == "__main__":
    # --- 1. SET YOUR PATHS ---
    noisy_processed_dir = r"processed\noisy_train" 
    clean_processed_dir = r"processed\clean_train"
    model_output_path = r"models\baseline_ridge_model_1.pkl"
    
    # --- 2. RUN FULL PIPELINE ---
    # Setting max_files=None forces it to load the entire dataset
    X_train, y_train = load_data_subset(noisy_processed_dir, clean_processed_dir, max_files=None)
    
    train_baseline_model(X_train, y_train, model_output_path)