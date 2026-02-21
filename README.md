nele_mlops/
├── data/
│   ├── raw/                 # Put the downloaded VoiceBank .wav files here
│   └── processed/           # Extracted .npy spectrograms will go here
├── src/
│   ├── extract_features.py  # Our feature extraction script
│   └── train_model.py       # Where we will build the regressor later
├── requirements.txt
└── README.md
