# Medical AI Assistant Platform

A comprehensive AI-powered medical assistant platform providing preliminary healthcare diagnosis through multiple analysis modalities including skin disease detection, lab report interpretation, respiratory sound analysis, and medical chatbot consultation.

---

## Table of Contents

- [Trained Models](#trained-models)
- [Features](#features)
- [Backend Overview](#backend-overview)
- [Frontend Overview](#frontend-overview)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Model Verification](#model-verification)
- [Datasets](#datasets)
- [Tech Stack](#tech-stack)
- [Disclaimer](#disclaimer)

---

## Trained Models

The following trained models are available in the `checkpoints/` directory:

| Model File | Purpose | Input | Output | Size | Dependencies |
|------------|---------|-------|--------|------|--------------|
| `skin_model_cls_seg_trained.pth` | Skin disease classification & segmentation | Skin images (JPG/PNG) | Diagnosis, confidence, treatment plan, severity | 92.76 MB | PyTorch, OpenCV, PIL |
| `skin_model_full.pth` | Comprehensive skin analysis (full model) | Skin images (JPG/PNG) | Extended diagnosis with detailed medications | 88.95 MB | PyTorch, OpenCV, PIL |
| `lab_model_cls_seg_trained.pth` | Lab report OCR & analysis | Lab report images | Extracted values, abnormality detection | 0.08 MB | PyTorch, Tesseract OCR |
| `sound_model_cls_seg_trained.pth` | Respiratory sound classification | Audio files (WAV) | Respiratory condition diagnosis | 12.71 MB | PyTorch, Librosa |
| `chatbot_model_cls_seg_trained.pth` | Medical chatbot NLP | Text queries | Symptom analysis, medical advice | 7.20 MB | PyTorch, Transformers |

---

## Detailed Model Descriptions

### 1. Skin Analysis Model (`skin_model_cls_seg_trained.pth`)

**Task Type:** Multi-class Classification + Segmentation

**Intended Role:** Analyzes skin images to detect and classify various dermatological conditions including acne, eczema, psoriasis, melanoma, dermatitis, rosacea, and fungal infections.

**Supported Conditions:**
- Healthy Skin
- Acne
- Eczema
- Psoriasis
- Melanoma (requires immediate medical attention)
- Dermatitis
- Rosacea
- Fungal Infection

**Performance Notes:**
- Input images are resized to 224x224 pixels
- Uses HSV and LAB color space analysis for accurate detection
- Confidence scores typically range from 60-95%
- Best results with well-lit, close-up images

**Preprocessing:**
```python
import cv2
import numpy as np

def preprocess_skin_image(image_path):
    """Preprocess skin image for model input."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img, axis=0)
```

**Example Usage:**
```python
from backend.models.skin_analyzer import SkinAnalyzer

# Initialize analyzer
analyzer = SkinAnalyzer()

# Analyze skin image
result = analyzer.analyze('path/to/skin_image.jpg')

print(f"Diagnosis: {result['diagnosis']}")
print(f"Confidence: {result['confidence']}%")
print(f"Severity: {result['severity']}")
print(f"Treatment: {result['treatment']}")
print(f"Medications: {result['medications']}")
```

**Output Structure:**
```python
{
    'diagnosis': str,           # e.g., 'Acne', 'Eczema', etc.
    'confidence': float,        # 0.0 - 100.0
    'treatment': str,           # Comprehensive treatment plan
    'severity': str,            # 'none', 'mild', 'moderate', 'severe'
    'medications': list,        # List of recommended medications
    'symptoms': list,           # Associated symptoms
    'recommendations': list,    # Care recommendations
    'analysis_details': dict    # Feature analysis details
}
```

---

### 2. Skin Model Full (`skin_model_full.pth`)

**Task Type:** Comprehensive Classification + Segmentation

**Intended Role:** Extended version of the skin analyzer with additional features for comprehensive skin analysis, including detailed medication dosages and treatment timelines.

**Performance Notes:**
- Same preprocessing as the standard skin model
- Provides more detailed treatment recommendations
- Includes severity-based medication adjustments

**Example Usage:**
```python
from backend.models.skin_analyzer import SkinAnalyzer

analyzer = SkinAnalyzer()
result = analyzer.analyze('path/to/skin_image.jpg')

# Access comprehensive treatment
print(result['treatment'])  # Detailed medication dosages included
```

---

### 3. Lab Report Analyzer (`lab_model_cls_seg_trained.pth`)

**Task Type:** OCR + Value Extraction + Classification

**Intended Role:** Extracts and analyzes lab test values from medical report images using OCR technology, detecting abnormal values and providing medical recommendations.

**Supported Lab Tests:**
- Glucose (Blood Sugar)
- Total Cholesterol
- HDL / LDL Cholesterol
- Triglycerides
- Hemoglobin
- WBC (White Blood Cells)
- RBC (Red Blood Cells)
- Platelets
- Creatinine
- ALT / AST (Liver enzymes)

**Performance Notes:**
- Best results with clear, high-resolution images
- Adaptive thresholding for various image qualities
- Automatic deskewing for rotated images

**Preprocessing:**
```python
import cv2
import numpy as np

def preprocess_lab_image(image_path):
    """Preprocess lab report image for OCR."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    return thresh
```

**Example Usage:**
```python
from backend.models.lab_analyzer import LabAnalyzer

# Initialize analyzer
analyzer = LabAnalyzer()

# Analyze lab report
result = analyzer.analyze('path/to/lab_report.jpg')

print(f"Diagnosis: {result['diagnosis']}")
print(f"Lab Values: {result['lab_values']}")
print(f"Abnormal Values: {result['abnormal_values']}")
print(f"Recommendations: {result['recommendations']}")
```

**Output Structure:**
```python
{
    'diagnosis': str,              # Summary of findings
    'treatment': str,              # Treatment recommendations
    'severity': str,               # 'none', 'mild', 'moderate', 'severe'
    'lab_values': dict,            # Extracted lab values
    'abnormal_values': list,       # List of abnormal findings
    'recommendations': list        # Health recommendations
}
```

---

### 4. Sound Analyzer (`sound_model_cls_seg_trained.pth`)

**Task Type:** Audio Classification

**Intended Role:** Analyzes respiratory sounds (breathing, coughing) to detect respiratory conditions using audio feature extraction.

**Supported Conditions:**
- Healthy Breathing
- Asthma
- Bronchitis
- Pneumonia
- COPD (Chronic Obstructive Pulmonary Disease)
- Whooping Cough

**Performance Notes:**
- Sample rate: 22050 Hz
- Uses MFCCs (Mel-frequency cepstral coefficients)
- Spectral features for pattern recognition
- Best results with clear recordings (minimal background noise)

**Preprocessing:**
```python
import librosa
import numpy as np

def extract_audio_features(audio_path):
    """Extract audio features for model input."""
    y, sr = librosa.load(audio_path, sr=22050)
    
    features = {}
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['mfcc_mean'] = np.mean(mfccs, axis=1)
    
    # Spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = np.mean(zcr)
    
    # RMS Energy
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = np.mean(rms)
    
    return features
```

**Example Usage:**
```python
from backend.models.sound_analyzer import SoundAnalyzer

# Initialize analyzer
analyzer = SoundAnalyzer()

# Analyze respiratory sound
result = analyzer.analyze('path/to/breathing_sound.wav')

print(f"Diagnosis: {result['diagnosis']}")
print(f"Confidence: {result['confidence']}%")
print(f"Severity: {result['severity']}")
print(f"Recommendations: {result['recommendations']}")
```

**Output Structure:**
```python
{
    'diagnosis': str,              # e.g., 'Asthma', 'Bronchitis'
    'confidence': float,           # 0.0 - 100.0
    'treatment': str,              # Treatment plan
    'severity': str,               # 'none', 'moderate', 'severe'
    'recommendations': list,       # Health recommendations
    'audio_features': dict         # Extracted audio features
}
```

---

### 5. Medical Chatbot (`chatbot_model_cls_seg_trained.pth`)

**Task Type:** Natural Language Processing / Symptom Analysis

**Intended Role:** Provides medical advice based on symptom descriptions, medication recommendations, and general health guidance.

**Capabilities:**
- Symptom detection and analysis
- Multi-condition diagnosis prediction
- Medication recommendations with dosages
- Severity assessment
- Treatment timeline estimation
- Emergency detection

**Supported Symptoms:**
- Fever, headache, cough, sore throat
- Chest pain, shortness of breath
- Stomach/abdominal pain, nausea, vomiting, diarrhea
- Dizziness, fatigue, weakness
- Rash, itching
- Back pain, joint pain, muscle pain
- Runny nose, congestion, sneezing

**Example Usage:**
```python
from backend.models.chatbot import MedicalChatbot

# Initialize chatbot
chatbot = MedicalChatbot()

# Get medical advice
response = chatbot.get_response("I have fever and headache for 2 days")

print(response)
```

**Output Example:**
```
🏥 MEDICAL ANALYSIS REPORT
==================================================

🔍 SYMPTOMS DETECTED: FEVER, HEADACHE

📋 MOST LIKELY CONDITIONS:
   1. Common Cold (Confidence: 75%)
   2. Flu (Confidence: 70%)
   3. Viral Infection (Confidence: 65%)

⚠️ SEVERITY LEVEL: 🟢 MILD

💊 RECOMMENDED MEDICATIONS:
   1. Acetaminophen 500mg
      → For: fever, headache
      → Dosage: 500mg per dose
      → Timing: Every 4-6 hours as needed
      → Duration: Maximum 3000mg per day, up to 7 days

📝 DETAILED MEDICAL ADVICE:
1. Rest, stay hydrated, take fever reducers...
```

---

## Features

- **Skin Analysis**: AI-powered skin disease detection and classification with 8 condition types
- **Lab Report Analysis**: OCR-based extraction and interpretation of medical lab reports
- **Sound Analysis**: Respiratory sound classification for health monitoring
- **Medical Chatbot**: AI assistant for medical queries, symptom analysis, and health information
- **Health Records**: Track and manage your health analysis history
- **User Authentication**: Secure user registration and login system with JWT tokens
- **Dashboard**: Overview of all health analyses and statistics

---

## Backend Overview

### Architecture

The backend is built with **Flask** and provides RESTful APIs for all analysis services.

### Services

| Service | Endpoint | Description |
|---------|----------|-------------|
| Authentication | `/api/register`, `/api/login` | User registration and JWT-based authentication |
| Profile | `/api/profile` | User profile management |
| Skin Analysis | `/api/analyze/skin` | Upload and analyze skin images |
| Lab Analysis | `/api/analyze/lab` | Upload and analyze lab reports |
| Sound Analysis | `/api/analyze/sound` | Upload and analyze respiratory sounds |
| Chatbot | `/api/chatbot` | Medical Q&A chatbot |
| Health Records | `/api/records` | Retrieve analysis history |
| Dashboard | `/api/dashboard/stats` | User statistics and recent records |

### File Structure

```
backend/
├── app.py                 # Main Flask application
├── setup_project.py       # Project setup utilities
├── __init__.py
├── database/
│   ├── db.py              # SQLite database configuration
│   └── __init__.py
├── models/
│   ├── skin_analyzer.py   # Skin analysis model
│   ├── lab_analyzer.py    # Lab report analyzer
│   ├── sound_analyzer.py  # Sound analysis model
│   ├── chatbot.py         # Medical chatbot
│   └── __init__.py
└── utils/
    ├── gpu_config.py      # GPU configuration
    ├── helpers.py         # Utility functions
    └── __init__.py
```

---

## Frontend Overview

### Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Modern UI**: Clean, intuitive interface
- **Real-time Analysis**: Instant feedback on uploads
- **Secure**: JWT token-based authentication

### Pages

| Page | File | Description |
|------|------|-------------|
| Home | `index.html` | Landing page with feature overview |
| Dashboard | `dashboard.html` | User statistics and recent analyses |
| Skin Analysis | `skin-analysis.html` | Upload and analyze skin images |
| Lab Analysis | `lab-analysis.html` | Upload and analyze lab reports |
| Sound Analysis | `sound-analysis.html` | Upload and analyze respiratory sounds |
| Chatbot | `chatbot.html` | Medical Q&A interface |
| Health Records | `health-records.html` | View analysis history |
| Profile | `profile.html` | User profile management |
| About | `about.html` | About the platform |
| Contact | `contact.html` | Contact information |

### File Structure

```
frontend/
├── index.html             # Landing page
├── dashboard.html         # User dashboard
├── skin-analysis.html     # Skin analysis page
├── lab-analysis.html      # Lab analysis page
├── sound-analysis.html    # Sound analysis page
├── chatbot.html           # Medical chatbot
├── health-records.html    # Health records
├── profile.html           # User profile
├── about.html             # About page
├── contact.html           # Contact page
├── css/
│   └── styles.css         # Main stylesheet
└── js/
    └── app.js             # Main JavaScript
```

---

## Installation

### Prerequisites

- Python 3.11+
- pip (Python package manager)
- Git LFS (for downloading model files)
- Tesseract OCR (optional, for lab report analysis)

### Step 1: Clone the Repository

```bash
git clone https://github.com/tarekmhmd/Medical-AI-Assistant-Platform.git
cd Medical-AI-Assistant-Platform
```

### Step 2: Install Git LFS and Download Models

```bash
git lfs install
git lfs pull
```

This downloads the large model files (.pth) from Git LFS.

### Step 3: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Configure Environment Variables

Copy the example environment file:

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

Edit `.env` and set your configuration:

```env
# Flask Configuration
FLASK_APP=backend/app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-change-this-in-production

# Database
DATABASE_URL=sqlite:///medical_assistant.db
```

### Step 6: (Optional) Install Tesseract OCR

For lab report OCR functionality:

```bash
# Windows - Download installer from:
# https://github.com/UB-Mannheim/tesseract/wiki

# Or run the provided batch file:
download_tesseract_portable.bat
```

---

## Usage

### Start the Backend Server

```bash
cd backend
python app.py
```

The server will start at: `http://localhost:5000`

### Access the Frontend

Open your browser and navigate to:
```
http://localhost:5000
```

Or open `frontend/index.html` directly.

### Using Docker (Optional)

```bash
docker-compose up --build
```

---

## API Reference

### Authentication

#### Register User
```http
POST /api/register
Content-Type: application/json

{
    "email": "user@example.com",
    "password": "securepassword",
    "name": "John Doe",
    "age": 30,
    "address": "123 Main St"
}
```

#### Login
```http
POST /api/login
Content-Type: application/json

{
    "email": "user@example.com",
    "password": "securepassword"
}
```

**Response:**
```json
{
    "token": "eyJhbGciOiJIUzI1NiIs...",
    "user": {
        "id": 1,
        "email": "user@example.com",
        "name": "John Doe"
    }
}
```

### Analysis Endpoints

All analysis endpoints require authentication. Include the JWT token in the header:

```http
Authorization: Bearer <your_jwt_token>
```

#### Skin Analysis
```http
POST /api/analyze/skin
Content-Type: multipart/form-data

image: <skin_image_file>
```

#### Lab Analysis
```http
POST /api/analyze/lab
Content-Type: multipart/form-data

image: <lab_report_image>
```

#### Sound Analysis
```http
POST /api/analyze/sound
Content-Type: multipart/form-data

audio: <audio_file.wav>
```

#### Chatbot
```http
POST /api/chatbot
Content-Type: application/json
Authorization: Bearer <token>

{
    "message": "I have fever and headache"
}
```

---

## Model Verification

### Verify Models Are Loaded

Start the server and check for model loading messages:

```bash
cd backend
python app.py
```

Expected output:
```
==================================================
Medical AI Assistant Server
==================================================
Skin analysis model loaded (demo mode)
Lab analysis model loaded (demo mode)
Chatbot model loaded (demo mode)
Sound analysis model loaded (demo mode)
Server running at: http://localhost:5000
==================================================
```

### Test Skin Analysis

```python
import requests

# Login first
response = requests.post('http://localhost:5000/api/login', json={
    'email': 'user@example.com',
    'password': 'password'
})
token = response.json()['token']

# Test skin analysis
with open('test_skin_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/analyze/skin',
        headers={'Authorization': f'Bearer {token}'},
        files={'image': f}
    )
print(response.json())
```

### Test Chatbot

```bash
curl -X POST http://localhost:5000/api/chatbot \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the symptoms of diabetes?"}'
```

### Test via Frontend

1. Register a new account at `http://localhost:5000`
2. Login with your credentials
3. Navigate to each analysis page
4. Upload test files and verify results

---

## Datasets

**Important:** Datasets are **NOT** included in this repository due to their large size.

The following datasets were used for training:

| Model | Dataset | Source |
|-------|---------|--------|
| Skin Analysis | ISIC 2016/2018/2019, HAM10000 | [ISIC Archive](https://www.isic-archive.com/) |
| Lab Analysis | Medical lab report images | Custom dataset |
| Sound Analysis | Respiratory Sound Database | [PhysioNet](https://physionet.org/) |
| Chatbot | Medical Q&A datasets | Custom curated dataset |

### Downloading Datasets

To obtain the datasets:

1. **Skin Images (ISIC):**
   ```bash
   python download_isic2016_task1.py
   python download_and_unzip_ham10000.py
   ```

2. **Other datasets:** Contact the repository owner or refer to original sources.

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | Flask, Python 3.11 |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Database** | SQLite (SQLAlchemy) |
| **AI/ML** | PyTorch, TensorFlow, Transformers |
| **Computer Vision** | OpenCV, PIL |
| **Audio Processing** | Librosa, SoundFile |
| **OCR** | Tesseract (pytesseract) |
| **Authentication** | JWT, bcrypt |
| **Containerization** | Docker, Docker Compose |

---

## Project Structure

```
Medical-AI-Assistant-Platform/
├── backend/                    # Backend source code
│   ├── app.py                  # Main Flask application
│   ├── database/               # Database configuration
│   ├── models/                 # AI model wrappers
│   └── utils/                  # Utility functions
├── frontend/                   # Frontend source code
│   ├── *.html                  # HTML pages
│   ├── css/                    # Stylesheets
│   └── js/                     # JavaScript files
├── checkpoints/                # Trained model files (.pth)
│   ├── skin_model_cls_seg_trained.pth
│   ├── skin_model_full.pth
│   ├── lab_model_cls_seg_trained.pth
│   ├── sound_model_cls_seg_trained.pth
│   └── chatbot_model_cls_seg_trained.pth
├── requirements.txt            # Python dependencies
├── .env.example                # Environment configuration template
├── docker-compose.yml          # Docker configuration
├── Dockerfile                  # Docker image definition
├── .gitignore                  # Git ignore rules
├── .gitattributes              # Git LFS tracking
└── README.md                   # This file
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Author

Developed by [tarekmhmd](https://github.com/tarekmhmd)

---

## Disclaimer

⚠️ **IMPORTANT MEDICAL DISCLAIMER**

This application is for **educational and research purposes only**. It should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment.

- Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
- Never disregard professional medical advice or delay in seeking it because of something you have read or received from this application.
- If you think you may have a medical emergency, call your doctor or emergency services immediately.
- The AI models provide preliminary analysis only and should not be considered definitive diagnosis.

The developers and contributors of this project are not responsible for any decisions made based on the output of this application.