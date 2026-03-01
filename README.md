# 🏥 MediSense AI

**A comprehensive AI-powered medical assistant platform for multi-modal health analysis**

[![Python](https://img.shields.io/badge/Python-3.11.4-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [AI Models & Modules](#ai-models--modules)
- [Technology Stack](#technology-stack)
- [Installation & Setup](#installation--setup)
- [Running the Project](#running-the-project)
- [Verification & Testing](#verification--testing)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Important Notes](#important-notes)
- [Disclaimer](#disclaimer)

---

## 🎯 Project Overview

**MediSense AI** is an intelligent medical assistant platform that leverages artificial intelligence to provide preliminary health analysis across multiple modalities. The platform integrates four specialized AI analyzers to help users understand their health conditions through image, text, and audio analysis.

### Purpose
- Provide accessible preliminary health information
- Assist in understanding medical reports and symptoms
- Track personal health records over time
- Offer educational health insights

### Target Users
- General public seeking preliminary health information
- Patients wanting to understand their lab results
- Individuals tracking their health conditions
- Healthcare students for educational purposes

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔬 **Skin Analysis** | AI-powered skin condition detection from images |
| 🧪 **Lab Report Analysis** | OCR-based extraction and interpretation of lab results |
| 💬 **Medical Chatbot** | Symptom-based disease diagnosis and medical advice |
| 🎤 **Respiratory Sound Analysis** | Audio analysis for respiratory conditions |
| 📋 **Health Records** | Track and monitor health history |
| 👤 **User Profiles** | Personalized health tracking with secure authentication |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (HTML/CSS/JS)                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │  Login   │ │Dashboard │ │  Skin    │ │   Lab    │ │ Chatbot  │      │
│  │  Page    │ │   Page   │ │ Analysis │ │ Analysis │ │   Page   │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ HTTP/REST API
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        BACKEND (Flask - Python)                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    API Routes & Authentication                   │   │
│  │         (JWT Tokens, User Management, File Uploads)             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                  │                                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │    Skin     │ │     Lab     │ │   Chatbot   │ │    Sound    │       │
│  │  Analyzer   │ │  Analyzer   │ │   Model     │ │  Analyzer   │       │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘       │
│                                  │                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    SQLite Database                               │   │
│  │         (Users, Health Records, Chat History)                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🤖 AI Models & Modules

### 1. Skin Analyzer (`backend/models/skin_analyzer.py`)

**Purpose:** Analyzes skin images to detect dermatological conditions.

**Architecture:** UNet-based classifier with segmentation head

**Supported Conditions (8 classes):**
| Class | Condition | Description |
|-------|-----------|-------------|
| 0 | Healthy Skin | Normal skin appearance |
| 1 | Acne | Inflammatory skin condition |
| 2 | Eczema | Atopic dermatitis |
| 3 | Psoriasis | Autoimmune skin disease |
| 4 | Melanoma | Skin cancer (urgent referral) |
| 5 | Dermatitis | Contact dermatitis |
| 6 | Rosacea | Chronic facial redness |
| 7 | Fungal Infection | Ringworm, tinea |

**Analysis Features:**
- Multi-color space analysis (HSV, LAB)
- Texture analysis using Local Binary Patterns
- Edge detection and shape analysis
- Color uniformity assessment
- Border regularity detection
- Symmetry analysis

**Output:**
```json
{
  "diagnosis": "Acne",
  "confidence": 85.5,
  "severity": "mild",
  "treatment": "Comprehensive treatment plan...",
  "medications": ["Benzoyl Peroxide 5%", "Salicylic Acid 2%"],
  "symptoms": ["Red bumps", "Pustules", "Oily skin"],
  "recommendations": ["Gentle cleansing", "Avoid triggers"]
}
```

---

### 2. Lab Analyzer (`backend/models/lab_analyzer.py`)

**Purpose:** Extracts and interprets laboratory test results from report images.

**Architecture:** OCR-based extraction with MLP classifier

**Supported Tests:**
| Test | Normal Range | Unit |
|------|-------------|------|
| Glucose | 70-100 | mg/dL |
| Total Cholesterol | 0-200 | mg/dL |
| HDL | 40+ | mg/dL |
| LDL | 0-100 | mg/dL |
| Triglycerides | 0-150 | mg/dL |
| Hemoglobin | 12-17 | g/dL |
| WBC | 4000-11000 | cells/mcL |
| RBC | 4.5-5.5 | million/mcL |
| Platelets | 150000-400000 | cells/mcL |
| Creatinine | 0.6-1.2 | mg/dL |
| ALT | 7-56 | U/L |
| AST | 10-40 | U/L |

**Analysis Process:**
1. Image preprocessing (denoising, thresholding, deskewing)
2. OCR text extraction using Tesseract
3. Pattern matching for lab values
4. Normal range comparison
5. Severity assessment
6. Treatment recommendation generation

**Output:**
```json
{
  "diagnosis": "High Glucose Detected",
  "severity": "moderate",
  "lab_values": {"glucose": 145, "cholesterol": 180},
  "abnormal_values": [{"test": "glucose", "value": 145, "status": "high"}],
  "treatment": "Detailed treatment plan with medications...",
  "recommendations": ["Monitor blood sugar", "Follow up with doctor"]
}
```

---

### 3. Medical Chatbot (`backend/models/chatbot.py`)

**Purpose:** Provides symptom-based disease diagnosis and medical advice.

**Architecture:** LSTM-based with word attention mechanism

**Symptom Database:** 25+ symptoms mapped to conditions

**Supported Symptoms:**
- Fever, headache, cough, sore throat
- Chest pain, shortness of breath
- Stomach pain, nausea, vomiting, diarrhea
- Dizziness, fatigue, weakness
- Rash, itching, back pain, joint pain
- Runny nose, congestion, sneezing

**Analysis Process:**
1. Symptom detection from user message
2. Condition matching with confidence scoring
3. Severity assessment
4. Medication recommendation with dosages
5. Treatment timeline estimation

**Output:**
```json
{
  "response": "🏥 MEDICAL ANALYSIS REPORT\n...\n💊 RECOMMENDED MEDICATIONS:\n1. Acetaminophen 500mg\n   → Dosage: Every 4-6 hours\n...",
  "conditions": ["Common Cold", "Flu"],
  "severity": "mild"
}
```

---

### 4. Sound Analyzer (`backend/models/sound_analyzer.py`)

**Purpose:** Analyzes respiratory sounds to detect lung conditions.

**Architecture:** CNN-based audio classifier with attention maps

**Supported Conditions (6 classes):**
| Class | Condition | Description |
|-------|-----------|-------------|
| 0 | Healthy Breathing | Normal respiratory sounds |
| 1 | Asthma | Airway obstruction (wheezing) |
| 2 | Bronchitis | Airway inflammation |
| 3 | Pneumonia | Lung infection |
| 4 | COPD | Chronic obstructive disease |
| 5 | Whooping Cough | Bacterial infection |

**Audio Features Extracted:**
- MFCCs (Mel-frequency cepstral coefficients)
- Spectral centroid
- Spectral rolloff
- Zero crossing rate
- RMS energy

**Output:**
```json
{
  "diagnosis": "Asthma",
  "confidence": 75.0,
  "severity": "moderate",
  "treatment": "Use prescribed inhaler...",
  "recommendations": ["Avoid triggers", "Keep rescue inhaler available"],
  "audio_features": {
    "spectral_centroid": 2500.5,
    "rms_energy": 0.045,
    "zero_crossing_rate": 0.12
  }
}
```

---

## 🛠️ Technology Stack

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| HTML5 | - | Structure |
| CSS3 | - | Styling |
| JavaScript | ES6+ | Client-side logic |
| Font Awesome | 6.x | Icons |

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.11.4 | Programming language |
| Flask | 3.0.0 | Web framework |
| Flask-CORS | 4.0.0 | Cross-origin support |
| SQLite | - | Database |
| PyJWT | 2.8.0 | Authentication tokens |
| Werkzeug | 3.0.1 | Password hashing |

### AI/ML Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| TensorFlow | 2.15.0 | Deep learning framework |
| PyTorch | 2.1.0 | Deep learning framework |
| OpenCV | 4.8.1.78 | Image processing |
| Librosa | 0.10.1 | Audio processing |
| Pytesseract | 0.3.10 | OCR engine |
| NumPy | 1.24.3 | Numerical computing |
| scikit-learn | 1.3.2 | Machine learning utilities |

---

## 📥 Installation & Setup

### Prerequisites

- **Python 3.11.4** (recommended) or 3.9+
- **pip** (Python package manager)
- **Git** (for cloning)
- **Tesseract OCR** (optional, for lab report analysis)

### Step 1: Clone/Download Project

```cmd
git clone https://github.com/your-username/medisense-ai.git
cd medisense-ai
```

### Step 2: Run Setup Script

```cmd
setup.bat
```

This will:
1. ✅ Check Python version
2. ✅ Create virtual environment (`venv/`)
3. ✅ Install all dependencies
4. ✅ Initialize SQLite database
5. ✅ (Optional) Install Tesseract OCR

### Step 3: Install Tesseract OCR (Optional - for Lab Analysis)

If you skipped Tesseract during setup:

```cmd
setup_tesseract.bat
```

Or download manually from:
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- Default path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

---

## 🚀 Running the Project

### Start the Application

```cmd
run.bat
```

### Access the Application

Open your browser and navigate to:
```
http://localhost:5000
```

### Default Behavior

1. **Registration:** Create a new account with email/password
2. **Login:** Access the dashboard
3. **Analysis:** Use any of the four analysis tools
4. **Records:** View your health history

---

## ✅ Verification & Testing

### Test 1: Server Startup

Run the server and verify:
```
✓ Server starts without errors
✓ Database initializes successfully
✓ All models load correctly
✓ Server accessible at http://localhost:5000
```

Expected console output:
```
==================================================
Medical AI Assistant Server
==================================================
Server running at: http://localhost:5000
Press Ctrl+C to stop
==================================================
Skin analysis model loaded (demo mode)
Lab analysis model loaded (demo mode)
Chatbot model loaded (demo mode)
Sound analysis model loaded (demo mode)
Database initialized successfully!
```

### Test 2: User Authentication

1. Navigate to `http://localhost:5000`
2. Click "Register" and create an account
3. Verify successful registration message
4. Login with credentials
5. Verify redirect to dashboard

### Test 3: Skin Analysis

1. Go to Dashboard → Skin Analysis
2. Upload a skin image (JPG/PNG)
3. Verify analysis results include:
   - Diagnosis
   - Confidence score
   - Severity level
   - Treatment recommendations
   - Medications list

### Test 4: Lab Report Analysis

1. Go to Dashboard → Lab Analysis
2. Upload a lab report image
3. Verify extracted values and analysis

> **Note:** If Tesseract is not installed, demo data will be used.

### Test 5: Medical Chatbot

1. Go to Dashboard → Medical Chatbot
2. Enter symptoms: "I have fever and headache"
3. Verify response includes:
   - Possible conditions
   - Recommended medications with dosages
   - Severity assessment
   - Medical advice

### Test 6: Sound Analysis

1. Go to Dashboard → Sound Analysis
2. Upload an audio file (WAV/MP3)
3. Verify respiratory analysis results

### Test 7: Health Records

1. Go to Dashboard → Health Records
2. Verify all previous analyses are listed
3. Check record details (type, date, diagnosis)

### Automated Testing

Run the test script:
```cmd
python test_all_models.py
```

This will test all four AI models and generate a report.

---

## 📡 API Documentation

### Authentication Endpoints

#### Register User
```http
POST /api/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword",
  "name": "John Doe",
  "age": 30
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

Response:
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "user": {"id": 1, "email": "user@example.com", "name": "John Doe"}
}
```

### Analysis Endpoints

#### Skin Analysis
```http
POST /api/analyze/skin
Authorization: Bearer <token>
Content-Type: multipart/form-data

image: <skin_image.jpg>
```

#### Lab Analysis
```http
POST /api/analyze/lab
Authorization: Bearer <token>
Content-Type: multipart/form-data

image: <lab_report.jpg>
```

#### Sound Analysis
```http
POST /api/analyze/sound
Authorization: Bearer <token>
Content-Type: multipart/form-data

audio: <breathing_sound.wav>
```

#### Chatbot
```http
POST /api/chatbot
Authorization: Bearer <token>
Content-Type: application/json

{
  "message": "I have fever and headache"
}
```

### Data Endpoints

#### Get Health Records
```http
GET /api/records
Authorization: Bearer <token>
```

#### Get Dashboard Stats
```http
GET /api/dashboard/stats
Authorization: Bearer <token>
```

---

## 📁 Project Structure

```
medisense-ai/
├── backend/
│   ├── app.py                 # Main Flask application
│   ├── setup_project.py       # Project setup script
│   ├── database/
│   │   └── db.py              # Database initialization
│   ├── models/
│   │   ├── skin_analyzer.py   # Skin condition analysis
│   │   ├── lab_analyzer.py    # Lab report analysis
│   │   ├── chatbot.py         # Medical chatbot
│   │   └── sound_analyzer.py  # Respiratory sound analysis
│   ├── utils/                 # Helper functions
│   └── uploads/               # Uploaded files storage
├── frontend/
│   ├── index.html             # Login/Register page
│   ├── dashboard.html         # Main dashboard
│   ├── profile.html           # User profile
│   ├── skin-analysis.html     # Skin analysis page
│   ├── lab-analysis.html      # Lab analysis page
│   ├── chatbot.html           # Medical chatbot page
│   ├── sound-analysis.html    # Sound analysis page
│   ├── health-records.html    # Health records page
│   ├── about.html             # About page
│   ├── contact.html           # Contact page
│   ├── css/
│   │   └── styles.css         # Main stylesheet
│   └── js/
│       └── app.js             # Client-side JavaScript
├── data/                      # Knowledge bases
│   ├── diseases/              # Disease database
│   ├── skin_images/           # Skin condition database
│   ├── lab_results/           # Lab test database
│   └── respiratory_sounds/    # Respiratory database
├── checkpoints/               # Trained model weights
├── models_pretrained/         # Pre-trained models
├── tessdata/                  # Tesseract language data
├── venv/                      # Virtual environment
├── requirements.txt           # Python dependencies
├── setup.bat                  # Setup script
├── run.bat                    # Run script
└── README.md                  # This file
```

---

## ⚠️ Important Notes

### Dependencies

1. **Tesseract OCR** is required for lab report analysis
   - Install via `setup_tesseract.bat`
   - Or download from: https://github.com/UB-Mannheim/tesseract/wiki

2. **Virtual Environment** must be activated before running
   - `run.bat` handles this automatically

3. **Port 5000** must be available
   - Change in `backend/app.py` if needed

### Model Status

All AI models are fully functional in **demo mode** with rule-based analysis:

| Model | Status | Mode |
|-------|--------|------|
| Skin Analyzer | ✅ Working | Demo (rule-based + image features) |
| Lab Analyzer | ✅ Working | Demo (OCR + pattern matching) |
| Medical Chatbot | ✅ Working | Demo (symptom matching) |
| Sound Analyzer | ✅ Working | Demo (audio feature analysis) |

To use deep learning models:
1. Place trained models in `models_pretrained/`
2. Update model loading code in each analyzer

### Database

- SQLite database (`medical_assistant.db`) is created automatically
- Contains: users, health_records, chat_history tables
- No manual setup required

---

## ⚖️ Disclaimer

> **IMPORTANT: This application is for educational and informational purposes only.**

This application provides preliminary health information and should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

**Key Points:**
- ❌ Not a replacement for professional medical consultation
- ❌ Not intended for emergency medical situations
- ❌ Cannot diagnose with 100% accuracy
- ✅ Useful for educational purposes
- ✅ Provides general health information
- ✅ Helps track personal health records

**If you are experiencing a medical emergency, call emergency services (911) immediately.**

---

## 📄 License

This project is for educational purposes. See [LICENSE](LICENSE) for details.

---

## 📞 Support

For questions or issues:
- Email: support@medisense-ai.com
- GitHub Issues: [Project Issues Page]

---

<div align="center">

**Built with ❤️ for better healthcare accessibility**

</div>