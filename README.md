# 🎤 Voice Cloning Detection via Timbre Watermarking and Reed-Solomon Error Correction

A dual-layer voice protection system that detects AI-generated voice clones and embeds imperceptible watermarks into audio files for ownership verification.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![Flask](https://img.shields.io/badge/flask-2.0+-red.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [API Endpoints](#api-endpoints)
- [License](#license)

---

## 🎯 Overview

With the rise of AI voice cloning technology, there's an increasing threat of voice-based fraud and misinformation. This project provides **two layers of protection**:

1. **AI Voice Detection**: Machine learning-based classification to identify synthetic/cloned voices
2. **Timbre Watermarking**: Imperceptible audio watermarking using spectral domain embedding

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **AI Detection** | Classifies audio as REAL or FAKE using MFCC features and RandomForest |
| 🔐 **Watermark Embedding** | Hides ownership data in audio's frequency spectrum (timbre) |
| 📖 **Watermark Extraction** | Recovers hidden data with Reed-Solomon error correction |
| 🌐 **Web Interface** | Modern dark-mode UI with amplifier-themed design |
| ⚡ **Real-time Processing** | Fast analysis via Flask REST API |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (HTML/JS)                       │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ AI Detection │ │ Embed Tab    │ │ Verify Tab   │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   FLASK REST API                            │
│  /predict          /embed           /verify                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   BACKEND PROCESSING                        │
│  ┌──────────────────┐  ┌─────────────────────────────────┐ │
│  │ MFCC Extraction  │  │ Timbre Watermarking             │ │
│  │ + RandomForest   │  │ (FFT + Reed-Solomon)            │ │
│  └──────────────────┘  └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/Voice-Cloning-Detection-via-Timber-watermarking-and-Reed-Soloman-Error-Correction.git
cd Voice-Cloning-Detection-via-Timber-watermarking-and-Reed-Soloman-Error-Correction
```

2. **Create virtual environment**
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install flask flask-cors numpy librosa soundfile scikit-learn reedsolo
```

4. **Run the server**
```bash
python app.py
```

5. **Open the frontend**
```
Open frontend/index.html in your browser
# Or navigate to http://127.0.0.1:5000
```

---

## 💻 Usage

### 1. AI Voice Detection
- Upload a WAV audio file
- Click "Analyze Audio"
- View REAL/FAKE verdict with confidence score

### 2. Watermark Embedding
- Upload audio file
- Enter watermark text (e.g., your ID or ownership info)
- Click "Embed & Download"
- Save the watermarked audio (sounds identical to original!)

### 3. Watermark Extraction
- Upload watermarked audio
- Click "Extract Watermark"
- View the recovered ownership data

---

## 🔬 Technical Details

### AI Detection Pipeline
- **Feature Extraction**: 20 MFCCs (Mel-Frequency Cepstral Coefficients)
- **Classifier**: RandomForest with 100 estimators
- **Sample Rate**: 22050 Hz

### Timbre Watermarking
- **Technique**: FFT-based spectral domain embedding
- **Error Correction**: Reed-Solomon RS(15,11)
- **Redundancy**: 5x with majority voting
- **Target Bins**: 1.5 kHz and 3 kHz frequency ranges
- **Chunk Size**: 1024 samples

### Why "Timbre" Watermarking?
Timbre is the quality that distinguishes sounds of the same pitch - determined by the frequency spectrum. We embed data by modifying the relative energy between frequency bins, creating imperceptible changes in the audio's timbre.

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Classify audio as REAL or FAKE |
| `/embed` | POST | Embed watermark into audio |
| `/verify` | POST | Extract watermark from audio |

### Example Request
```bash
curl -X POST -F "audio=@sample.wav" http://127.0.0.1:5000/predict
```

---

## 📁 Project Structure

```
├── backend/
│   ├── app.py                 # Flask API server
│   ├── feature_extraction.py  # MFCC extraction
│   ├── watermarking.py        # Timbre watermarking logic
│   ├── train_model.py         # Model training script
│   └── uploads/               # Temporary file storage
├── frontend/
│   ├── index.html             # Main web interface
│   ├── workflow_diagram.html  # System diagrams
│   ├── presentation_notes.html# Project documentation
│   └── presenter_script.html  # Presentation guide
├── models/
│   └── model.pkl              # Trained RandomForest model
└── data/
    ├── real/                  # Real voice samples
    └── fake/                  # AI-generated samples
```

---

## 🔮 Future Scope

- Deep learning models (CNN/Transformer) for higher accuracy
- Real-time browser extension for instant verification
- Blockchain integration for tamper-proof ownership records
- Mobile application support
- Multi-format support (MP3, FLAC)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Team

Academic Project - Voice Signal Integrity System

---

## 🙏 Acknowledgments

- librosa for audio analysis
- scikit-learn for machine learning
- reedsolo for Reed-Solomon encoding
