# Medical AI Assistant Platform - System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MEDICAL AI ASSISTANT PLATFORM                         │
│                         Multi-Modal Healthcare Diagnosis System                  │
└─────────────────────────────────────────────────────────────────────────────────┘

                                    ┌──────────────┐
                                    │   FRONTEND   │
                                    │  (Web Interface) │
                                    │  HTML/CSS/JS  │
                                    └──────┬───────┘
                                           │ HTTP/REST API
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              BACKEND SERVER (Flask)                             │
│                                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Auth      │  │   User      │  │   Health    │  │   File      │            │
│  │  Module     │  │  Profile    │  │  Records    │  │  Upload     │            │
│  │  (JWT)      │  │  Management │  │  Database   │  │  Handler    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                                  │
│                              ┌──────────────┐                                   │
│                              │  SQLite DB   │                                   │
│                              │  (Users,     │                                   │
│                              │   Records)   │                                   │
│                              └──────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            AI ANALYSIS MODULES                                   │
│                                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐ │
│  │  SKIN ANALYZER  │  │  LAB ANALYZER   │  │ SOUND ANALYZER  │  │  CHATBOT   │ │
│  │                 │  │                 │  │                 │  │            │ │
│  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌────────┐ │ │
│  │ │ UNet        │ │  │ │ MLP         │ │  │ │ CNN         │ │  │ │ LSTM   │ │ │
│  │ │ Classifier  │ │  │ │ Segmenter   │ │  │ │ Segmenter   │ │  │ │Segmentr│ │ │
│  │ │             │ │  │ │             │ │  │ │             │ │  │ │        │ │ │
│  │ │ Seg: 84.4%  │ │  │ │ Acc: 73.8%  │ │  │ │ Acc: 100%   │ │  │ │Acc:100%│ │ │
│  │ │ Dice: 91.4% │ │  │ │ AUC: 83.5%  │ │  │ │ Att: 100%   │ │  │ │Att:100%│ │ │
│  │ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │  │ └────────┘ │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └────────────┘ │
│                                                                                  │
│                            UNIFIED MODEL ARCHITECTURE                            │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                     Classification + Segmentation Head                      │ │
│  │                                                                             │ │
│  │   Input          Encoder           Bottleneck         Decoder      Output   │ │
│  │  ┌─────┐      ┌─────────┐       ┌───────────┐     ┌─────────┐   ┌────────┐ │ │
│  │  │Image│  ──▶ │  Conv   │  ──▶  │   Dense   │ ──▶ │  Trans  │──▶│Mask+Cls│ │ │
│  │  │Audio│      │  LSTM   │       │   Layer   │     │  Conv   │   │        │ │ │
│  │  │Text│       │  MLP    │       │           │     │         │   │        │ │ │
│  │  │Lab │       │         │       │           │     │         │   │        │ │ │
│  │  └─────┘      └─────────┘       └───────────┘     └─────────┘   └────────┘ │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              KNOWLEDGE BASES                                     │
│                                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Disease   │  │    Skin     │  │     Lab     │  │ Respiratory │            │
│  │  Database   │  │  Conditions │  │    Tests    │  │  Conditions │            │
│  │             │  │  Database   │  │  Reference  │  │  Database   │            │
│  │ 10+ diseases│  │ 7 conditions│  │ 8 test types│  │ 6 conditions│            │
│  │ 40+ meds    │  │ 25+ meds    │  │ 30+ meds    │  │ 20+ meds    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING DATA                                      │
│                                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   ISIC      │  │  Respiratory│  │   Diabetes  │  │  Medical    │            │
│  │  2018/2019  │  │   Sounds    │  │   Dataset   │  │    QA       │            │
│  │             │  │             │  │             │  │  Dataset    │            │
│  │ 11,735      │  │   Audio     │  │   Tabular   │  │  Symptoms   │            │
│  │  images     │  │   samples   │  │   data      │  │  + Diseases │            │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Model Architectures

### 1. UNetClassifier (Skin Analysis)
```
Input: [B, 3, 256, 256] RGB Image
         │
         ▼
┌─────────────────────────────────────┐
│           ENCODER PATH              │
│  Conv3x3 → BN → ReLU → Conv3x3 → BN │
│         ↓ MaxPool2x2                │
│  32 → 64 → 128 → 256 → 512          │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│           BOTTLENECK                │
│  Conv3x3 → BN → ReLU → Conv3x3 → BN │
│  512 → 1024 channels                │
│         ↓                           │
│  ┌─────────────────┐                │
│  │ Classification  │                │
│  │ GlobalPool→FC   │                │
│  │ 1024→512→128→8  │                │
│  └─────────────────┘                │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│           DECODER PATH              │
│  UpConv2x2 → Concat → Conv          │
│  512 → 256 → 128 → 64 → 32          │
│         ↓                           │
│  ┌─────────────────┐                │
│  │ Segmentation    │                │
│  │ Conv1x1 → Sigmoid│               │
│  │ Output: [B,1,H,W]│               │
│  └─────────────────┘                │
└─────────────────────────────────────┘

Output: (Mask [B,1,256,256], Class [B,8])
```

### 2. MLPSegmenter (Lab Analysis)
```
Input: [B, 8] Lab Features
         │
         ▼
┌─────────────────────────────────────┐
│        SHARED FEATURES              │
│  Linear(8→128) → BN → ReLU → Dropout│
│  Linear(128→64) → BN → ReLU → Drop  │
│  Linear(64→32) → BN → ReLU → Drop   │
└─────────────────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────────┐
│ Class  │ │ Importance │
│ Head   │ │    Map     │
│ 32→2   │ │ 32→64→64   │
│        │ │ [B,1,8,8]  │
└────────┘ └────────────┘

Output: (Importance Map [B,1,8,8], Class [B,2])
```

### 3. CNNSegmenter (Sound Analysis)
```
Input: [B, 1, 64, 128] Mel-Spectrogram
         │
         ▼
┌─────────────────────────────────────┐
│           ENCODER                    │
│  Conv3x3 → BN → ReLU → MaxPool       │
│  32 → 64 → 128 → 256 channels        │
│  Skip connections stored             │
└─────────────────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────────┐
│ Class  │ │  Decoder   │
│ Head   │ │  UpConv    │
│ 256→   │ │  + Skip    │
│ 512→   │ │  128→64→32 │
│ 128→6  │ │  Conv1x1   │
└────────┘ └────────────┘

Output: (Attention Map [B,1,64,128], Class [B,6])
```

### 4. LSTMSegmenter (Chatbot)
```
Input: [B, seq_len] Word Indices
         │
         ▼
┌─────────────────────────────────────┐
│        EMBEDDING LAYER              │
│  Embedding(vocab_size, 128)         │
│  Output: [B, seq_len, 128]          │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│        BIDIRECTIONAL LSTM           │
│  2 layers, hidden_dim=256           │
│  Output: [B, seq_len, 256]          │
└─────────────────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────────┐
│ Class  │ │   Word     │
│ Head   │ │ Attention  │
│ Attn   │ │ Linear→1   │
│ Pool   │ │ Sigmoid    │
│ FC→num │ │ [B,1,seq]  │
│ classes│ │            │
└────────┘ └────────────┘

Output: (Attention Map [B,1,seq_len], Class [B,num_classes])
```

## Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           USER INTERACTION                               │
│                                                                          │
│  1. User uploads image/audio/text through web interface                  │
│  2. Backend receives file and stores in uploads/ directory               │
│  3. Appropriate analyzer module is invoked                               │
│  4. Preprocessing: resize, normalize, feature extraction                 │
│  5. Model inference: classification + segmentation                       │
│  6. Knowledge base lookup for medications and treatments                 │
│  7. Response generation with diagnosis, treatment, severity              │
│  8. Results stored in database and returned to user                      │
└──────────────────────────────────────────────────────────────────────────┘
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | HTML5, CSS3, JavaScript, Bootstrap |
| Backend | Python 3.11, Flask, Flask-CORS |
| Database | SQLite with Flask-SQLAlchemy |
| Authentication | JWT (JSON Web Tokens) |
| AI Framework | PyTorch 2.x |
| Image Processing | OpenCV, PIL |
| Audio Processing | Librosa, SoundFile |
| OCR | Tesseract |
| Deployment | Docker, Docker Compose |
