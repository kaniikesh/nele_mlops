import os
import numpy as np
from sklearn.linear_model import Ridge
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def load_data_subset(noisy_dir, clean_dir, max_files=100):
    """Loads a subset of the extracted .npy spectrograms and flattens them for ML."""
    noisy_path = Path(noisy_dir)
    clean_path = Path(clean_dir)
    
    X_list = [] # Will hold noisy frames
    y_list = [] # Will hold clean frames
    
    # Get a list of noisy files, limited by max_files to save RAM during the baseline test
    noisy_files = list(noisy_path.glob("*.npy"))[:max_files]
    
    print(f"Loading {len(noisy_files)} files for baseline training...")
    
    for noisy_file in noisy_files:
        # Find the matching clean file
        clean_file = clean_path / noisy_file.name
        
        if clean_file.exists():
            # Load the numpy arrays
            noisy_mag = np.load(noisy_file)
            clean_mag = np.load(clean_file)
            
            # Spectrograms are shape (frequency_bins, time_frames). 
            # We transpose them to (time_frames, frequency_bins) so each frame is a "row" of data
            X_list.append(noisy_mag.T)
            y_list.append(clean_mag.T)
            
    # Stack all frames together into massive 2D matrices
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
    # Point these to the "processed" folders you just created in the last step
    noisy_processed_dir = r"processed\noisy_train" 
    clean_processed_dir = r"processed\clean_train"
    
    # Where to save the trained model
    model_output_path = r"models\baseline_ridge_model.pkl"
    
    # --- 2. RUN PIPELINE ---
    # We are loading just 100 files first to ensure it works without crashing your RAM
    X_train, y_train = load_data_subset(noisy_processed_dir, clean_processed_dir, max_files=100)
    
    train_baseline_model(X_train, y_train, model_output_path)