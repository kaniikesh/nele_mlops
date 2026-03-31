import torch
import torch.optim as optim
import torch.nn as nn
import mlflow
import mlflow.pytorch

# Import your custom model and the new real-data loader
from model import AudioEnhancementCNN
from dataset import AudioEnhancementDataset 
from torch.utils.data import DataLoader

def train_model():
    # 1. Set up MLflow tracking
    mlflow.set_experiment("Near-End-Listening-Enhancement")
    
    # Hyperparameters
    epochs = 10
    learning_rate = 0.001
    batch_size = 8 # Processes 8 audio files at a time
    
    model = AudioEnhancementCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- UPDATED: Load Real Data ---
    print("Scanning audio folders and setting up dataset...")
    # These paths must match the folder structure where you saved LibriSpeech and ESC-50/TBUS
    dataset = AudioEnhancementDataset(clean_dir="data/clean_speech", noise_dir="data/noise")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Total training batches per epoch: {len(dataloader)}")

    # 2. Start the MLflow run
    with mlflow.start_run():
        # Log parameters so you can compare experiments later
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("architecture", "CNN_Encoder_Decoder")

        print("Starting training loop...")
        for epoch in range(epochs):
            running_loss = 0.0
            
            # --- UPDATED: Iterate over real audio batches ---
            for batch_idx, (noisy_inputs, clean_targets) in enumerate(dataloader):
                
                # Standard PyTorch Training Step
                optimizer.zero_grad()
                
                # Feed the real noisy market audio spectrogram into the model
                outputs = model(noisy_inputs)
                
                # Compare the model's output mask to the real clean voice spectrogram
                loss = criterion(outputs, clean_targets)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # Print progress every batch to see it working in real-time
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(dataloader)}] | Loss: {loss.item():.4f}")
            
            # Calculate average loss for the entire epoch and log it to MLflow
            avg_epoch_loss = running_loss / len(dataloader)
            mlflow.log_metric("train_loss", avg_epoch_loss, step=epoch)
            print(f"--- Epoch {epoch+1} Completed | Average Loss: {avg_epoch_loss:.4f} ---")

       # 3. Save the trained model to the MLflow Model Registry
        mlflow.pytorch.log_model(model, "audio_enhancement_model")
        print("Model saved to MLflow successfully!")
        
        # --- ADD THIS LINE ---
        torch.save(model.state_dict(), "audio_model_weights.pth")
        print("Raw PyTorch weights saved as audio_model_weights.pth!")

if __name__ == "__main__":
    train_model()
