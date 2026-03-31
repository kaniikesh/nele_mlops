import torch
import torch.nn as nn

class AudioEnhancementCNN(nn.Module):
    def __init__(self):
        super(AudioEnhancementCNN, self).__init__()
        
        # Encoder: Extracts patterns from the audio features
        # UPDATED: in_channels=1 because we pass a single Noisy Spectrogram map
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Decoder: Reconstructs the enhanced audio mask
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2, stride=2),
            nn.Sigmoid() # Outputs a mask with values between 0 and 1
        )

    def forward(self, x):
        features = self.encoder(x)
        mask = self.decoder(features)
        return mask