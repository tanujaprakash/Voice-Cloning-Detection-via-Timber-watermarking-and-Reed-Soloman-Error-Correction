import librosa
import numpy as np

def extract_features(audio_path: str) -> np.ndarray:
    """
    Extracts acoustic features (like MFCCs) used for the original AI model prediction.
    Timbre/Spectral analysis (MFCCs, Spectral Centroid, etc.) already happens here.
    """
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        # Extract 20 MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        # Flatten and return the mean of each coefficient
        mean_mfccs = np.mean(mfccs, axis=1)
        return mean_mfccs
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        # Return an array of NaNs if extraction fails
        return np.full((20,), np.nan)