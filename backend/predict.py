import pickle
import sys
import os
from feature_extraction import extract_features

# ✅ Path to saved model
MODEL_PATH = r"C:\Users\DELL\Downloads\voice-cloning-detection\models\model.pkl"

def predict(audio_file):
    if not os.path.exists(MODEL_PATH):
        print("❌ Model file not found. Please train the model first.")
        return

    # Load trained model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Extract features from new audio
    features = extract_features(audio_file).reshape(1, -1)

    # Predict
    prediction = model.predict(features)[0]
    if prediction == 1:
        print(f"✅ {audio_file} → REAL voice")
    else:
        print(f"❌ {audio_file} → FAKE (cloned) voice")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio.wav>")
    else:
        audio_path = sys.argv[1]
        if not os.path.exists(audio_path):
            print(f"❌ File not found: {audio_path}")
        else:
            predict(audio_path)
