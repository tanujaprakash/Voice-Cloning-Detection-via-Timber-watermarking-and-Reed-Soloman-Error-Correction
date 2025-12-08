import os
import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from feature_extraction import extract_features

# --- Prepare dataset ---
def prepare_dataset(real_path, fake_path):
    features, labels = [], []

    print(f"[LOAD] Loading REAL samples from: {real_path}")
    real_files = [f for f in os.listdir(real_path) if f.endswith((".wav", ".flac"))]
    real_count = 0
    
    for file in real_files:
        try:
            feat = extract_features(os.path.join(real_path, file))
            if not np.isnan(feat).any():
                features.append(feat)
                labels.append(1)
                real_count += 1
                if real_count % 10 == 0:
                    print(f"  Loaded {real_count} real samples...", end='\r')
        except Exception as e:
            print(f"  [WARN]  Warning: Failed to process {file}: {e}")
    print(f"  Loaded {real_count} real samples.          ")

    print(f"[LOAD] Loading FAKE samples from: {fake_path}")
    fake_files = [f for f in os.listdir(fake_path) if f.endswith((".wav", ".flac"))]
    
    # Balance dataset: Limit fake samples to number of real samples
    if len(fake_files) > real_count:
        print(f"  [INFO] Balancing dataset: Limiting fake samples to {real_count} (same as real)")
        random.shuffle(fake_files)
        fake_files = fake_files[:real_count]
    
    fake_count = 0
    for file in fake_files:
        try:
            feat = extract_features(os.path.join(fake_path, file))
            if not np.isnan(feat).any():
                features.append(feat)
                labels.append(0)
                fake_count += 1
                if fake_count % 10 == 0:
                    print(f"  Loaded {fake_count} fake samples...", end='\r')
        except Exception as e:
            print(f"  [WARN]  Warning: Failed to process {file}: {e}")
    print(f"  Loaded {fake_count} fake samples.          ")

    print(f"\n[OK] Loaded {real_count} REAL and {fake_count} FAKE samples")
    return np.array(features), np.array(labels)

# --- Main training ---
if __name__ == "__main__":
    print("=" * 70)
    print("Voice Cloning Detection - Model Training")
    print("=" * 70)
    print("\n[INFO] NO GPU REQUIRED - RandomForest runs on CPU")
    print("\n" + "=" * 70)
    
    # Use relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    real_path = os.path.join(project_root, "data", "real")
    fake_path = os.path.join(project_root, "data", "fake")

    # Validate paths exist
    if not os.path.exists(real_path):
        print(f"[ERROR] ERROR: Real samples directory not found: {real_path}")
        print("\nPlease run one of these first:")
        print("  - python generate_samples.py (quick synthetic samples)")
        print("  - python download_datasets.py (real ASVspoof dataset)")
        exit(1)
    
    if not os.path.exists(fake_path):
        print(f"[ERROR] ERROR: Fake samples directory not found: {fake_path}")
        print("\nPlease run one of these first:")
        print("  - python generate_samples.py (quick synthetic samples)")
        print("  - python download_datasets.py (real ASVspoof dataset)")
        exit(1)

    print("\n[DATA] Preparing dataset...")
    X, y = prepare_dataset(real_path, fake_path)
    
    # Validate minimum samples
    if len(y) < 10:
        print(f"\n[WARN]  WARNING: Only {len(y)} samples found!")
        print("This is too small for reliable training.")
        print("Recommended: At least 50 samples (25 real + 25 fake)")
        print("\nContinuing anyway, but expect poor accuracy...")
    elif len(y) < 50:
        print(f"\n[WARN]  WARNING: Only {len(y)} samples found.")
        print("Recommended: At least 50 samples for decent accuracy")
    
    print(f"\n[STATS] Dataset Statistics:")
    print(f"  Total samples: {len(y)}")
    print(f"  Real samples: {np.sum(y == 1)}")
    print(f"  Fake samples: {np.sum(y == 0)}")
    print(f"  Features per sample: {X.shape[1]}")
    
    # Check class balance
    real_ratio = np.sum(y == 1) / len(y) * 100
    print(f"  Class balance: {real_ratio:.1f}% real, {100-real_ratio:.1f}% fake")
    if real_ratio < 30 or real_ratio > 70:
        print("  [WARN]  Warning: Imbalanced dataset may affect accuracy")

    print("\n[SPLIT] Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Training samples: {len(y_train)}")
    print(f"  Test samples: {len(y_test)}")

    print("\n[TRAIN] Training RandomForestClassifier (CPU-based)...")
    print("  This may take a few minutes depending on dataset size...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)
    rf.fit(X_train, y_train)

    print("\n[DATA] Evaluating model...")
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)
    print(f"\n[OK] Accuracy: {accuracy * 100:.2f}%")
    
    print("\n[MATRIX] Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"              Predicted")
    print(f"              FAKE  REAL")
    print(f"Actual FAKE   {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"       REAL   {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    print("\n[DATA] Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))

    # --- Save model ---
    model_dict = {
        "model": rf,
        "scaler": None,  # No scaler used
        "accuracy": accuracy,
        "num_samples": len(y),
        "num_features": X.shape[1]
    }

    model_path = os.path.join(project_root, "models", "model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pickle.dump(model_dict, open(model_path, "wb"))
    
    print("\n" + "=" * 70)
    print(f"[OK] Model saved at: {model_path}")
    print("=" * 70)
    print("\n[INFO] Model Information:")
    print(f"  - Accuracy: {accuracy * 100:.2f}%")
    print(f"  - Training samples: {len(y)}")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Algorithm: RandomForest (CPU-based, no GPU needed)")
    print("\n[NEXT] Next step: Test at http://127.0.0.1:5000/")

