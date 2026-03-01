"""
Respiratory Sound CNN Classifier Training Script
================================================
This script trains a CNN model to classify respiratory sounds (e.g., healthy vs unhealthy)
using Mel-Spectrograms as input features.

Steps:
1. Load WAV files from train/val/test folders
2. Convert audio to Mel-Spectrogram
3. Normalize and preprocess spectrograms
4. Train CNN classifier
5. Validate and save model
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wave
import struct
import math

# Try to import torchaudio, fall back to librosa or manual processing
try:
    import torchaudio
    import torchaudio.transforms as T
    AUDIO_BACKEND = "torchaudio"
except ImportError:
    torchaudio = None
    try:
        import librosa
        AUDIO_BACKEND = "librosa"
    except ImportError:
        librosa = None
        AUDIO_BACKEND = "manual"
        print("⚠️ torchaudio and librosa not found. Using basic WAV loading.")

print(f"🎵 Using audio backend: {AUDIO_BACKEND}")

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "sample_rate": 22050,       # Target sample rate
    "n_mels": 64,               # Number of mel bands
    "n_fft": 1024,              # FFT window size
    "hop_length": 512,          # Hop length for STFT
    "target_width": 128,        # Fixed width for spectrogram (time dimension)
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 0.001,
    "num_classes": 2,           # Binary classification (healthy vs unhealthy)
    "dropout": 0.3,
}

# =============================================================================
# AUDIO PROCESSING UTILITIES
# =============================================================================

def load_wav_manual(filepath, target_sr=22050):
    """
    Load WAV file manually without external libraries.
    Returns waveform as numpy array and sample rate.
    """
    try:
        with wave.open(filepath, 'rb') as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            
            # Read raw audio data
            raw_data = wav_file.readframes(n_frames)
            
            # Convert to numpy array based on sample width
            if sample_width == 1:
                dtype = np.uint8
            elif sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                dtype = np.int16
            
            audio = np.frombuffer(raw_data, dtype=dtype)
            
            # Convert to float and normalize
            if dtype == np.uint8:
                audio = (audio.astype(np.float32) - 128) / 128.0
            else:
                audio = audio.astype(np.float32) / np.iinfo(dtype).max
            
            # Convert stereo to mono by averaging channels
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)
            
            # Resample if needed (simple linear interpolation)
            if sample_rate != target_sr:
                ratio = target_sr / sample_rate
                new_length = int(len(audio) * ratio)
                indices = np.linspace(0, len(audio) - 1, new_length)
                audio = np.interp(indices, np.arange(len(audio)), audio)
            
            return audio, target_sr
            
    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None, None


def compute_mel_spectrogram(audio, sample_rate, n_mels=64, n_fft=1024, hop_length=512):
    """
    Compute Mel-Spectrogram manually using numpy/scipy.
    """
    try:
        # Ensure audio is long enough
        if len(audio) < n_fft:
            audio = np.pad(audio, (0, n_fft - len(audio)))
        
        # Compute STFT manually using sliding window
        n_frames = 1 + (len(audio) - n_fft) // hop_length
        stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)
        
        # Create Hann window
        window = np.hanning(n_fft)
        
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + n_fft] * window
            spectrum = np.fft.rfft(frame)
            stft_matrix[:, i] = spectrum
        
        # Compute magnitude spectrogram
        magnitude = np.abs(stft_matrix)
        
        # Create mel filter bank
        low_freq = 0
        high_freq = sample_rate / 2
        
        # Convert Hz to Mel
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)
        
        mel_low = hz_to_mel(low_freq)
        mel_high = hz_to_mel(high_freq)
        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # Convert Hz points to FFT bin indices
        bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
        
        # Create mel filter bank
        filter_bank = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            
            for j in range(left, center):
                if j < n_fft // 2 + 1:
                    filter_bank[i, j] = (j - left) / (center - left) if center != left else 0
            for j in range(center, right):
                if j < n_fft // 2 + 1:
                    filter_bank[i, j] = (right - j) / (right - center) if right != center else 0
        
        # Apply mel filter bank
        mel_spec = np.dot(filter_bank, magnitude)
        
        # Convert to log scale (dB)
        mel_spec_db = 20 * np.log10(mel_spec + 1e-10)
        
        # Normalize to [0, 1]
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-10)
        
        return mel_spec_db.astype(np.float32)
        
    except Exception as e:
        print(f"❌ Error computing mel spectrogram: {e}")
        return None


def process_audio_file(filepath, config):
    """
    Process a single audio file and return mel-spectrogram tensor.
    Handles different audio backends and error cases.
    """
    try:
        if AUDIO_BACKEND == "torchaudio":
            # Use torchaudio for processing
            waveform, sr = torchaudio.load(filepath)
            
            # Resample if needed
            if sr != config["sample_rate"]:
                resampler = T.Resample(sr, config["sample_rate"])
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Compute mel spectrogram
            mel_transform = T.MelSpectrogram(
                sample_rate=config["sample_rate"],
                n_mels=config["n_mels"],
                n_fft=config["n_fft"],
                hop_length=config["hop_length"]
            )
            mel_spec = mel_transform(waveform)
            
            # Convert to dB scale
            amplitude_to_db = T.AmplitudeToDB()
            mel_spec_db = amplitude_to_db(mel_spec)
            
            # Normalize
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-10)
            
        elif AUDIO_BACKEND == "librosa":
            # Use librosa for processing
            audio, sr = librosa.load(filepath, sr=config["sample_rate"], mono=True)
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=config["sample_rate"],
                n_mels=config["n_mels"],
                n_fft=config["n_fft"],
                hop_length=config["hop_length"]
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-10)
            mel_spec_db = torch.from_numpy(mel_spec_db).unsqueeze(0)
            
        else:
            # Use manual processing
            audio, sr = load_wav_manual(filepath, config["sample_rate"])
            if audio is None:
                return None
            
            mel_spec_db = compute_mel_spectrogram(
                audio, sr,
                n_mels=config["n_mels"],
                n_fft=config["n_fft"],
                hop_length=config["hop_length"]
            )
            if mel_spec_db is None:
                return None
            mel_spec_db = torch.from_numpy(mel_spec_db).unsqueeze(0)
        
        # Resize to fixed dimensions
        current_height = mel_spec_db.shape[-2]
        current_width = mel_spec_db.shape[-1]
        target_height = config["n_mels"]
        target_width = config["target_width"]
        
        # Resize using interpolation
        mel_spec_db = F.interpolate(
            mel_spec_db.unsqueeze(0) if mel_spec_db.dim() == 3 else mel_spec_db.unsqueeze(0).unsqueeze(0),
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        )
        
        # Squeeze back to [1, n_mels, target_width]
        if mel_spec_db.dim() == 4:
            mel_spec_db = mel_spec_db.squeeze(0)
        
        return mel_spec_db.float()
        
    except Exception as e:
        print(f"⚠️ Failed to process {filepath}: {e}")
        return None


# =============================================================================
# DATASET CLASS
# =============================================================================

class RespiratorySoundDataset(Dataset):
    """
    Custom Dataset for respiratory sound classification.
    Loads WAV files, converts to mel-spectrograms, and assigns labels.
    """
    
    def __init__(self, folder, config, augment=False):
        """
        Args:
            folder: Path to folder containing WAV files
            config: Configuration dictionary
            augment: Whether to apply data augmentation
        """
        self.folder = folder
        self.config = config
        self.augment = augment
        
        # Find all WAV files
        self.files = []
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.lower().endswith('.wav'):
                    self.files.append(os.path.join(folder, f))
        
        print(f"📁 Found {len(self.files)} WAV files in {os.path.basename(folder)}")
        
        # Validate files (remove corrupted ones)
        self.valid_files = []
        for f in self.files:
            try:
                with wave.open(f, 'rb') as wf:
                    wf.getnframes()  # Just try to read something
                self.valid_files.append(f)
            except:
                print(f"⚠️ Skipping corrupted file: {os.path.basename(f)}")
        
        print(f"✅ {len(self.valid_files)} valid files")
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        filepath = self.valid_files[idx]
        
        # Process audio file
        mel_spec = process_audio_file(filepath, self.config)
        
        # If processing failed, return a zero tensor
        if mel_spec is None:
            mel_spec = torch.zeros(1, self.config["n_mels"], self.config["target_width"])
        
        # Assign label based on filename
        # Common patterns: "healthy", "normal", "crackle", "wheeze", etc.
        filename_lower = os.path.basename(filepath).lower()
        if any(keyword in filename_lower for keyword in ['healthy', 'normal', 'healthy_', 'n_']):
            label = 0  # Healthy
        else:
            label = 1  # Unhealthy (default for respiratory sounds dataset)
        
        # Data augmentation (optional)
        if self.augment:
            # Random time shift
            if torch.rand(1).item() > 0.5:
                shift = torch.randint(0, 10, (1,)).item()
                mel_spec = torch.roll(mel_spec, shift, dims=-1)
            
            # Random volume change
            if torch.rand(1).item() > 0.5:
                scale = 0.8 + torch.rand(1).item() * 0.4  # 0.8 to 1.2
                mel_spec = mel_spec * scale
        
        return mel_spec, label


# =============================================================================
# CNN MODEL
# =============================================================================

class RespiratorySoundCNN(nn.Module):
    """
    CNN architecture for respiratory sound classification.
    Designed for mel-spectrogram input [batch, 1, n_mels, time].
    """
    
    def __init__(self, num_classes=2, dropout=0.3):
        super(RespiratorySoundCNN, self).__init__()
        
        # Convolutional blocks with batch normalization
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Input: [batch, 1, n_mels, time]
        
        # Block 1: Conv -> BN -> ReLU -> Pool
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        
        # Block 3
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        
        # Block 4
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        # Adaptive pooling to handle variable input sizes
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch and return average loss."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (mel_specs, labels) in enumerate(dataloader):
        mel_specs = mel_specs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(mel_specs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model and return loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for mel_specs, labels in dataloader:
            mel_specs = mel_specs.to(device)
            labels = labels.to(device)
            
            outputs = model(mel_specs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    avg_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


def train_model(train_dir, val_dir, config, save_path="checkpoints/sound_model.pth"):
    """
    Main training function.
    
    Args:
        train_dir: Path to training data folder
        val_dir: Path to validation data folder
        config: Configuration dictionary
        save_path: Path to save the trained model
    """
    print("=" * 60)
    print("🎙️ RESPIRATORY SOUND CLASSIFICATION TRAINING")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")
    
    # Create datasets
    print("\n📦 Loading datasets...")
    train_dataset = RespiratorySoundDataset(train_dir, config, augment=True)
    val_dataset = RespiratorySoundDataset(val_dir, config, augment=False)
    
    # Check if datasets are not empty
    if len(train_dataset) == 0:
        print("❌ No training data found!")
        return None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0
    ) if len(val_dataset) > 0 else None
    
    # Create model
    print("\n🏗️ Building model...")
    model = RespiratorySoundCNN(
        num_classes=config["num_classes"],
        dropout=config["dropout"]
    ).to(device)
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    print("\n🚀 Starting training...")
    print("-" * 60)
    
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(1, config["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        print(f"📈 Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        
        # Validate
        if val_loader is not None:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            print(f"📉 Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': config
                }, save_path)
                print(f"💾 Best model saved! (Val Acc: {val_acc*100:.2f}%)")
        else:
            # Save model every epoch if no validation
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, save_path)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print(f"📁 Model saved to: {save_path}")
    print(f"🏆 Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print("=" * 60)
    
    return model, history


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Define paths
    BASE_DIR = r"D:\project 2"
    TRAIN_DIR = os.path.join(BASE_DIR, "data", "sound", "train")
    VAL_DIR = os.path.join(BASE_DIR, "data", "sound", "val")
    SAVE_PATH = os.path.join(BASE_DIR, "checkpoints", "sound_model.pth")
    
    # Run training
    model, history = train_model(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        config=CONFIG,
        save_path=SAVE_PATH
    )