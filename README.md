# Medical AI Assistant Platform

An AI-powered medical assistant platform that provides skin disease analysis, lab report interpretation, respiratory sound analysis, and a medical chatbot for health-related queries.

---

## Trained Models

The following trained models are available in the `checkpoints/` directory:

| Model File | Purpose | How to Use | Size |
|------------|---------|------------|------|
| `skin_model_cls_seg_trained.pth` | Skin disease classification and segmentation from images | Upload skin images via the Skin Analysis page | 92.76 MB |
| `skin_model_full.pth` | Full skin analysis model (comprehensive) | Upload skin images via the Skin Analysis page | 88.95 MB |
| `lab_model_cls_seg_trained.pth` | Lab report analysis and interpretation | Upload lab report images via the Lab Analysis page | 0.08 MB |
| `sound_model_cls_seg_trained.pth` | Respiratory sound classification (cough, breathing patterns) | Upload audio files via the Sound Analysis page | 12.71 MB |
| `chatbot_model_cls_seg_trained.pth` | Medical chatbot for health-related Q&A | Interact via the Chatbot page | 7.20 MB |

---

## Features

- **Skin Analysis**: AI-powered skin disease detection and classification
- **Lab Report Analysis**: OCR and interpretation of medical lab reports
- **Sound Analysis**: Respiratory sound classification for health monitoring
- **Medical Chatbot**: AI assistant for medical queries and health information
- **Health Records**: Track and manage your health analysis history
- **User Authentication**: Secure user registration and login system

---

## Prerequisites

- Python 3.11+
- pip (Python package manager)
- Git LFS (for downloading model files)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/tarekmhmd/Medical-AI-Assistant-Platform.git
cd Medical-AI-Assistant-Platform
```

### 2. Install Git LFS (if not already installed)

```bash
git lfs install
git lfs pull
```

This will download the large model files (.pth) from Git LFS.

### 3. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Copy the example environment file and configure it:

```bash
copy .env.example .env
```

Edit `.env` and set your secret key:

```
SECRET_KEY=your-secure-secret-key-here
```

---

## Running the Application

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

Or open `frontend/index.html` directly in your browser.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/register` | POST | Register a new user |
| `/api/login` | POST | User login |
| `/api/profile` | GET/PUT | Get/update user profile |
| `/api/analyze/skin` | POST | Analyze skin image |
| `/api/analyze/lab` | POST | Analyze lab report image |
| `/api/analyze/sound` | POST | Analyze respiratory sound |
| `/api/chatbot` | POST | Chat with medical assistant |
| `/api/records` | GET | Get health records history |
| `/api/dashboard/stats` | GET | Get dashboard statistics |

---

## Verifying Model Functionality

### 1. Check Models are Loaded

Start the server and verify no model loading errors appear in the console.

### 2. Test Skin Analysis

```bash
# Using curl (replace YOUR_TOKEN with actual JWT token)
curl -X POST http://localhost:5000/api/analyze/skin \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "image=@path/to/skin_image.jpg"
```

### 3. Test Chatbot

```bash
curl -X POST http://localhost:5000/api/chatbot \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the symptoms of diabetes?"}'
```

### 4. Test via Frontend

1. Register a new account at `http://localhost:5000`
2. Login with your credentials
3. Navigate to each analysis page and test with sample files

---

## Project Structure

```
Medical-AI-Assistant-Platform/
├── backend/
│   ├── app.py                 # Main Flask application
│   ├── database/              # Database configuration
│   ├── models/                # AI model wrappers
│   └── utils/                 # Utility functions
├── frontend/
│   ├── index.html             # Landing page
│   ├── dashboard.html         # User dashboard
│   ├── skin-analysis.html     # Skin analysis page
│   ├── lab-analysis.html      # Lab analysis page
│   ├── sound-analysis.html    # Sound analysis page
│   ├── chatbot.html           # Medical chatbot
│   ├── css/                   # Stylesheets
│   └── js/                    # JavaScript files
├── checkpoints/               # Trained model files (.pth)
├── requirements.txt           # Python dependencies
├── .env.example               # Environment configuration template
├── docker-compose.yml         # Docker configuration
└── README.md                  # This file
```

---

## Datasets

**Note:** Datasets are NOT included in this repository due to their large size. 

The following datasets were used for training but must be downloaded separately:

- **Skin Analysis**: ISIC 2016/2018/2019 datasets, HAM10000
- **Lab Analysis**: Medical lab report images
- **Sound Analysis**: Respiratory sound database
- **Chatbot**: Medical Q&A datasets

To obtain the datasets, please refer to the original sources or contact the repository owner.

---

## Docker Deployment (Optional)

```bash
# Build and run with Docker Compose
docker-compose up --build
```

---

## Tech Stack

- **Backend**: Flask, Python, TensorFlow, PyTorch
- **Frontend**: HTML5, CSS3, JavaScript
- **Database**: SQLite (SQLAlchemy)
- **AI/ML**: TensorFlow, PyTorch, Transformers, OpenCV
- **Audio Processing**: Librosa, SoundFile
- **OCR**: Tesseract (pytesseract)

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Author

Developed by [tarekmhmd](https://github.com/tarekmhmd)

---

## Disclaimer

This application is for educational and research purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
