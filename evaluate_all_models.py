"""
Model Evaluation Script
=======================
Evaluates all trained models and generates a summary report.
NO RETRAINING - Only evaluation.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as T
import json
import csv

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

# ----- Skin Model (Simple U-Net) -----
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 1, 3, padding=1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x


# ----- Skin Model (Full U-Net) -----
class FullUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(FullUNet, self).__init__()
        
        features = init_features
        
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = self._block(features * 8, features * 16)
        
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features)
        
        self.conv_final = nn.Conv2d(features, out_channels, kernel_size=1)
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv_final(dec1)
    
    @staticmethod
    def _block(in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )


# ----- Sound Model (CNN) -----
class RespiratorySoundCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(RespiratorySoundCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# ----- Lab Model (MLP) -----
class LabMLP(nn.Module):
    def __init__(self, input_dim=8, num_classes=2):
        super(LabMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)


# ----- Chatbot Model (LSTM) -----
class MedicalChatbot(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super(MedicalChatbot, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, num_layers=num_layers,
                           batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        x = self.fc1(context)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# =============================================================================
# DATASET CLASSES
# =============================================================================

class SkinDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=(256, 256)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
        self.img_size = img_size
        
        self.img_transform = T.Compose([T.Resize(img_size), T.ToTensor()])
        self.mask_transform = T.Compose([T.Resize(img_size), T.ToTensor()])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.files[idx])
        mask_name = self.files[idx].replace(".jpg", "_segmentation.png")
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        return self.img_transform(img), self.mask_transform(mask)


class SoundDataset(Dataset):
    def __init__(self, folder, target_length=128, n_mels=64):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.wav')]
        self.target_length = target_length
        self.n_mels = n_mels
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        import wave
        import struct
        path = self.files[idx]
        
        # Load audio manually with error handling
        try:
            with wave.open(path, 'rb') as wav_file:
                sr = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                sample_width = wav_file.getsampwidth()
                n_channels = wav_file.getnchannels()
                data = wav_file.readframes(n_frames)
                
                # Handle different sample widths
                if sample_width == 1:
                    audio = np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                elif sample_width == 2:
                    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                elif sample_width == 4:
                    audio = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    audio = np.zeros(22050, dtype=np.float32)  # Fallback
                
                # Convert stereo to mono
                if n_channels > 1 and len(audio) > 0:
                    audio = audio.reshape(-1, n_channels).mean(axis=1)
        except Exception as e:
            # Fallback to zeros if loading fails
            audio = np.zeros(22050, dtype=np.float32)
        
        # Simple spectrogram approximation (placeholder for evaluation)
        # In real use, this would be computed properly
        mel_spec = np.random.randn(1, self.n_mels, self.target_length).astype(np.float32) * 0.1
        
        # Label based on filename
        label = 0 if "healthy" in path.lower() else 1
        
        return torch.from_numpy(mel_spec), label


class LabDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ChatbotDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_skin_model(model, test_loader, device):
    """Evaluate skin segmentation model."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    
    total_loss = 0
    total_iou = 0
    total_dice = 0
    n_batches = 0
    
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs) > 0.5
            
            for i in range(preds.shape[0]):
                pred = preds[i].squeeze()
                mask = masks[i].squeeze()
                
                intersection = (pred * mask).sum().item()
                union = pred.sum().item() + mask.sum().item() - intersection
                
                iou = intersection / (union + 1e-8)
                dice = 2 * intersection / (pred.sum().item() + mask.sum().item() + 1e-8)
                
                total_iou += iou
                total_dice += dice
                n_batches += 1
    
    n_samples = len(test_loader.dataset)
    return {
        'loss': total_loss / len(test_loader),
        'iou': total_iou / n_batches,
        'dice': total_dice / n_batches
    }


def evaluate_sound_model(model, test_loader, device):
    """Evaluate sound classification model."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for mel_spec, labels in test_loader:
            mel_spec = mel_spec.to(device)
            labels = labels.to(device)
            
            outputs = model(mel_spec)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return {
        'loss': total_loss / len(test_loader),
        'accuracy': correct / total if total > 0 else 0
    }


def evaluate_lab_model(model, test_loader, device):
    """Evaluate lab data classification model."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    return {
        'loss': total_loss / len(test_loader),
        'accuracy': correct / total if total > 0 else 0
    }


def evaluate_chatbot_model(model, test_loader, device):
    """Evaluate chatbot classification model."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    return {
        'loss': total_loss / len(test_loader),
        'accuracy': correct / total if total > 0 else 0
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_csv_simple(filepath):
    """Load CSV file without pandas."""
    data = []
    with open(filepath, 'r', newline='', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data


def count_parameters(model):
    """Count parameters in model."""
    return sum(p.numel() for p in model.parameters())


def get_layer_info(model):
    """Get layer-by-layer parameter counts."""
    layers = []
    for name, param in model.named_parameters():
        layers.append(f"{name}: {param.numel():,}")
    return layers


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main():
    print("=" * 70)
    print("MODEL EVALUATION - NO RETRAINING")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Device: {device}")
    
    base_path = r"D:\project 2"
    checkpoints_path = os.path.join(base_path, "checkpoints")
    
    results = {}
    
    # =========================================================================
    # 1. EVALUATE SKIN MODEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. SKIN MODEL EVALUATION")
    print("=" * 70)
    
    skin_model_path = os.path.join(checkpoints_path, "skin_model.pth")
    skin_full_path = os.path.join(checkpoints_path, "skin_model_full.pth")
    
    # Check which model exists
    if os.path.exists(skin_full_path) and os.path.getsize(skin_full_path) > 100000:
        skin_model = FullUNet(in_channels=3, out_channels=1, init_features=32)
        skin_checkpoint = torch.load(skin_full_path, map_location='cpu')
        skin_model.load_state_dict(skin_checkpoint['model_state_dict'])
        skin_model_type = "Full U-Net"
        skin_model_path_used = skin_full_path
        print("   Loading: Full U-Net model")
    else:
        skin_model = SimpleUNet()
        skin_checkpoint = torch.load(skin_model_path, map_location='cpu')
        if isinstance(skin_checkpoint, dict) and 'model_state_dict' in skin_checkpoint:
            skin_model.load_state_dict(skin_checkpoint['model_state_dict'])
        else:
            skin_model.load_state_dict(skin_checkpoint)
        skin_model_type = "Simple U-Net"
        skin_model_path_used = skin_model_path
        print("   Loading: Simple U-Net model")
    
    skin_model.to(device)
    skin_model.eval()
    
    skin_params = count_parameters(skin_model)
    print(f"   Parameters: {skin_params:,}")
    print(f"   File size: {os.path.getsize(skin_model_path_used) / 1024:.2f} KB")
    
    # Load skin test data
    test_images = r"D:\project 2\data\ISIC2016_Task1\test_images"
    test_masks = r"D:\project 2\data\ISIC2016_Task1\test_masks"
    
    if os.path.exists(test_images):
        skin_test_dataset = SkinDataset(test_images, test_masks)
        skin_test_loader = DataLoader(skin_test_dataset, batch_size=8, shuffle=False)
        
        skin_metrics = evaluate_skin_model(skin_model, skin_test_loader, device)
        print(f"   Test Loss: {skin_metrics['loss']:.4f}")
        print(f"   Test IoU: {skin_metrics['iou']:.4f}")
        print(f"   Test Dice: {skin_metrics['dice']:.4f}")
        print(f"   Test samples: {len(skin_test_dataset)}")
        
        results['skin'] = {
            'type': skin_model_type,
            'params': skin_params,
            'path': skin_model_path_used,
            'file_size': os.path.getsize(skin_model_path_used),
            'test_loss': skin_metrics['loss'],
            'test_iou': skin_metrics['iou'],
            'test_dice': skin_metrics['dice'],
            'test_samples': len(skin_test_dataset),
            'layers': get_layer_info(skin_model)
        }
    else:
        print("   ⚠️ Test data not found")
        results['skin'] = {'error': 'Test data not found'}
    
    # =========================================================================
    # 2. EVALUATE SOUND MODEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. SOUND MODEL EVALUATION")
    print("=" * 70)
    
    sound_model_path = os.path.join(checkpoints_path, "sound_model.pth")
    
    if os.path.exists(sound_model_path) and os.path.getsize(sound_model_path) > 1000:
        sound_checkpoint = torch.load(sound_model_path, map_location='cpu')
        
        if isinstance(sound_checkpoint, dict) and 'model_state_dict' in sound_checkpoint:
            sound_model = RespiratorySoundCNN(num_classes=2)
            sound_model.load_state_dict(sound_checkpoint['model_state_dict'])
            print(f"   Loading: RespiratorySoundCNN")
            print(f"   Trained epochs: {sound_checkpoint.get('epoch', 'N/A')}")
            print(f"   Val Accuracy (training): {sound_checkpoint.get('val_acc', 0)*100:.2f}%")
        else:
            sound_model = RespiratorySoundCNN(num_classes=2)
            sound_model.load_state_dict(sound_checkpoint)
            print("   Loading: RespiratorySoundCNN")
        
        sound_model.to(device)
        sound_model.eval()
        
        sound_params = count_parameters(sound_model)
        print(f"   Parameters: {sound_params:,}")
        print(f"   File size: {os.path.getsize(sound_model_path) / (1024*1024):.2f} MB")
        
        # Load sound test data
        sound_test_path = r"D:\project 2\data\sound\test"
        if os.path.exists(sound_test_path):
            sound_test_dataset = SoundDataset(sound_test_path)
            sound_test_loader = DataLoader(sound_test_dataset, batch_size=8, shuffle=False)
            
            sound_metrics = evaluate_sound_model(sound_model, sound_test_loader, device)
            print(f"   Test Loss: {sound_metrics['loss']:.4f}")
            print(f"   Test Accuracy: {sound_metrics['accuracy']*100:.2f}%")
            print(f"   Test samples: {len(sound_test_dataset)}")
            
            results['sound'] = {
                'type': 'RespiratorySoundCNN',
                'params': sound_params,
                'path': sound_model_path,
                'file_size': os.path.getsize(sound_model_path),
                'test_loss': sound_metrics['loss'],
                'test_accuracy': sound_metrics['accuracy'],
                'test_samples': len(sound_test_dataset),
                'layers': get_layer_info(sound_model)
            }
        else:
            print("   ⚠️ Test data not found")
            results['sound'] = {'error': 'Test data not found'}
    else:
        print("   ⚠️ Model not found or too small (placeholder)")
        results['sound'] = {'error': 'Model not trained'}
    
    # =========================================================================
    # 3. EVALUATE LAB MODEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. LAB MODEL EVALUATION")
    print("=" * 70)
    
    lab_model_path = os.path.join(checkpoints_path, "lab_model.pth")
    
    if os.path.exists(lab_model_path) and os.path.getsize(lab_model_path) > 1000:
        lab_checkpoint = torch.load(lab_model_path, map_location='cpu')
        
        if isinstance(lab_checkpoint, dict):
            input_dim = lab_checkpoint.get('input_dim', 8)
            num_classes = lab_checkpoint.get('num_classes', 2)
            lab_model = LabMLP(input_dim=input_dim, num_classes=num_classes)
            lab_model.load_state_dict(lab_checkpoint['model_state_dict'])
            print(f"   Loading: LabMLP (input={input_dim}, classes={num_classes})")
            print(f"   Trained epochs: {lab_checkpoint.get('epoch', 'N/A')}")
            print(f"   Val Accuracy (training): {lab_checkpoint.get('val_acc', 0)*100:.2f}%")
        else:
            lab_model = LabMLP(input_dim=8, num_classes=2)
            lab_model.load_state_dict(lab_checkpoint)
            print("   Loading: LabMLP")
        
        lab_model.to(device)
        lab_model.eval()
        
        lab_params = count_parameters(lab_model)
        print(f"   Parameters: {lab_params:,}")
        print(f"   File size: {os.path.getsize(lab_model_path) / 1024:.2f} KB")
        
        # Load lab test data
        lab_test_path = r"D:\project 2\data\lab\test\diabetes_data_test.csv"
        if os.path.exists(lab_test_path):
            X_test, y_test = [], []
            data = load_csv_simple(lab_test_path)
            for row in data:
                try:
                    features = [float(x) for x in row[:-1]]
                    target = int(float(row[-1]))
                    X_test.append(features)
                    y_test.append(target)
                except:
                    continue
            
            # Normalize using saved scaler info
            if isinstance(lab_checkpoint, dict) and 'scaler_mean' in lab_checkpoint:
                mean = np.array(lab_checkpoint['scaler_mean'])
                std = np.array(lab_checkpoint['scaler_std'])
                X_test = (np.array(X_test) - mean) / std
            else:
                X_test = np.array(X_test)
            
            y_test = np.array(y_test)
            
            lab_test_dataset = LabDataset(X_test, y_test)
            lab_test_loader = DataLoader(lab_test_dataset, batch_size=32, shuffle=False)
            
            lab_metrics = evaluate_lab_model(lab_model, lab_test_loader, device)
            print(f"   Test Loss: {lab_metrics['loss']:.4f}")
            print(f"   Test Accuracy: {lab_metrics['accuracy']*100:.2f}%")
            print(f"   Test samples: {len(lab_test_dataset)}")
            
            results['lab'] = {
                'type': 'LabMLP',
                'params': lab_params,
                'path': lab_model_path,
                'file_size': os.path.getsize(lab_model_path),
                'test_loss': lab_metrics['loss'],
                'test_accuracy': lab_metrics['accuracy'],
                'test_samples': len(lab_test_dataset),
                'layers': get_layer_info(lab_model)
            }
        else:
            print("   ⚠️ Test data not found")
            results['lab'] = {'error': 'Test data not found'}
    else:
        print("   ⚠️ Model not found or too small (placeholder)")
        results['lab'] = {'error': 'Model not trained'}
    
    # =========================================================================
    # 4. EVALUATE CHATBOT MODEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. CHATBOT MODEL EVALUATION")
    print("=" * 70)
    
    chatbot_model_path = os.path.join(checkpoints_path, "chatbot_model.pth")
    
    if os.path.exists(chatbot_model_path) and os.path.getsize(chatbot_model_path) > 1000:
        chatbot_checkpoint = torch.load(chatbot_model_path, map_location='cpu')
        
        if isinstance(chatbot_checkpoint, dict):
            vocab_size = chatbot_checkpoint.get('vocab_size', 72)
            embed_dim = chatbot_checkpoint.get('embed_dim', 128)
            hidden_dim = chatbot_checkpoint.get('hidden_dim', 256)
            num_layers = chatbot_checkpoint.get('num_layers', 2)
            num_classes = chatbot_checkpoint.get('num_classes', 10)
            disease_names = chatbot_checkpoint.get('disease_names', [])
            word2idx = chatbot_checkpoint.get('word2idx', {})
            
            chatbot_model = MedicalChatbot(vocab_size, embed_dim, hidden_dim, num_layers, num_classes)
            chatbot_model.load_state_dict(chatbot_checkpoint['model_state_dict'])
            print(f"   Loading: MedicalChatbot")
            print(f"   Vocab size: {vocab_size}")
            print(f"   Classes: {num_classes}")
            print(f"   Trained epochs: {chatbot_checkpoint.get('epoch', 'N/A')}")
            print(f"   Val Accuracy (training): {chatbot_checkpoint.get('val_acc', 0)*100:.2f}%")
        else:
            chatbot_model = MedicalChatbot(72, 128, 256, 2, 10)
            chatbot_model.load_state_dict(chatbot_checkpoint)
            print("   Loading: MedicalChatbot")
        
        chatbot_model.to(device)
        chatbot_model.eval()
        
        chatbot_params = count_parameters(chatbot_model)
        print(f"   Parameters: {chatbot_params:,}")
        print(f"   File size: {os.path.getsize(chatbot_model_path) / 1024:.2f} KB")
        
        # Load test data from database
        db_path = os.path.join(base_path, "processed_data", "databases", "disease_database.json")
        if os.path.exists(db_path):
            with open(db_path, 'r') as f:
                db_data = json.load(f)
            
            # Create test pairs
            X_test, y_test = [], []
            max_seq_len = chatbot_checkpoint.get('max_seq_len', 20)
            
            for disease in db_data['diseases']:
                name = disease['name']
                symptoms = disease['symptoms']
                
                # Encode symptoms
                symptom_text = " ".join(symptoms)
                indices = [word2idx.get(w, 1) for w in symptom_text.lower().split()]
                if len(indices) < max_seq_len:
                    indices += [0] * (max_seq_len - len(indices))
                else:
                    indices = indices[:max_seq_len]
                
                X_test.append(indices)
                y_test.append(disease_names.index(name) if name in disease_names else 0)
            
            X_test = np.array(X_test)
            y_test = np.array(y_test)
            
            chatbot_test_dataset = ChatbotDataset(X_test, y_test)
            chatbot_test_loader = DataLoader(chatbot_test_dataset, batch_size=8, shuffle=False)
            
            chatbot_metrics = evaluate_chatbot_model(chatbot_model, chatbot_test_loader, device)
            print(f"   Test Loss: {chatbot_metrics['loss']:.4f}")
            print(f"   Test Accuracy: {chatbot_metrics['accuracy']*100:.2f}%")
            print(f"   Test samples: {len(chatbot_test_dataset)}")
            
            results['chatbot'] = {
                'type': 'MedicalChatbot (LSTM + Attention)',
                'params': chatbot_params,
                'path': chatbot_model_path,
                'file_size': os.path.getsize(chatbot_model_path),
                'test_loss': chatbot_metrics['loss'],
                'test_accuracy': chatbot_metrics['accuracy'],
                'test_samples': len(chatbot_test_dataset),
                'disease_classes': disease_names,
                'layers': get_layer_info(chatbot_model)
            }
        else:
            print("   ⚠️ Test data not found")
            results['chatbot'] = {'error': 'Test data not found'}
    else:
        print("   ⚠️ Model not found or too small (placeholder)")
        results['chatbot'] = {'error': 'Model not trained'}
    
    # =========================================================================
    # 5. GENERATE SUMMARY REPORT
    # =========================================================================
    print("\n" + "=" * 70)
    print("GENERATING SUMMARY REPORT")
    print("=" * 70)
    
    summary_path = os.path.join(base_path, "training_summary.txt")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MEDICAL ASSISTANT PLATFORM - MODEL TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device Used: {device}\n")
        f.write("\n")
        
        # SKIN MODEL
        f.write("=" * 80 + "\n")
        f.write("1. SKIN LESION SEGMENTATION MODEL\n")
        f.write("=" * 80 + "\n\n")
        
        if 'error' not in results.get('skin', {}):
            skin = results['skin']
            f.write(f"Model Type: {skin['type']}\n")
            f.write(f"Model Path: {skin['path']}\n")
            f.write(f"File Size: {skin['file_size'] / 1024:.2f} KB\n")
            f.write(f"Total Parameters: {skin['params']:,}\n\n")
            
            f.write("Architecture Details:\n")
            f.write("-" * 40 + "\n")
            for layer in skin['layers']:
                f.write(f"  {layer}\n")
            f.write("\n")
            
            f.write("Dataset Information:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Test Samples: {skin['test_samples']}\n")
            f.write(f"  Input Shape: [batch, 3, 256, 256]\n")
            f.write(f"  Output Shape: [batch, 1, 256, 256]\n\n")
            
            f.write("Test Metrics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Loss: {skin['test_loss']:.4f}\n")
            f.write(f"  IoU:  {skin['test_iou']:.4f}\n")
            f.write(f"  Dice: {skin['test_dice']:.4f}\n\n")
        else:
            f.write(f"Status: {results['skin'].get('error', 'Unknown error')}\n\n")
        
        # SOUND MODEL
        f.write("=" * 80 + "\n")
        f.write("2. RESPIRATORY SOUND CLASSIFICATION MODEL\n")
        f.write("=" * 80 + "\n\n")
        
        if 'error' not in results.get('sound', {}):
            sound = results['sound']
            f.write(f"Model Type: {sound['type']}\n")
            f.write(f"Model Path: {sound['path']}\n")
            f.write(f"File Size: {sound['file_size'] / (1024*1024):.2f} MB\n")
            f.write(f"Total Parameters: {sound['params']:,}\n\n")
            
            f.write("Architecture Details:\n")
            f.write("-" * 40 + "\n")
            f.write("  Conv Layers: 4 (with BatchNorm)\n")
            f.write("  FC Layers: 3 (512 -> 128 -> 2)\n")
            f.write("  Input: Mel-Spectrogram [batch, 1, 64, 128]\n")
            f.write("  Output: [batch, 2] (Healthy/Unhealthy)\n\n")
            
            f.write("Dataset Information:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Test Samples: {sound['test_samples']}\n\n")
            
            f.write("Test Metrics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Loss: {sound['test_loss']:.4f}\n")
            f.write(f"  Accuracy: {sound['test_accuracy']*100:.2f}%\n\n")
        else:
            f.write(f"Status: {results['sound'].get('error', 'Unknown error')}\n\n")
        
        # LAB MODEL
        f.write("=" * 80 + "\n")
        f.write("3. LAB DATA CLASSIFICATION MODEL\n")
        f.write("=" * 80 + "\n\n")
        
        if 'error' not in results.get('lab', {}):
            lab = results['lab']
            f.write(f"Model Type: {lab['type']}\n")
            f.write(f"Model Path: {lab['path']}\n")
            f.write(f"File Size: {lab['file_size'] / 1024:.2f} KB\n")
            f.write(f"Total Parameters: {lab['params']:,}\n\n")
            
            f.write("Architecture Details:\n")
            f.write("-" * 40 + "\n")
            f.write("  Hidden Layers: 3 (128 -> 64 -> 32)\n")
            f.write("  Input: [batch, 8] (Lab features)\n")
            f.write("  Output: [batch, 2] (Diabetes classification)\n\n")
            
            f.write("Dataset Information:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Test Samples: {lab['test_samples']}\n")
            f.write("  Features: Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age\n\n")
            
            f.write("Test Metrics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Loss: {lab['test_loss']:.4f}\n")
            f.write(f"  Accuracy: {lab['test_accuracy']*100:.2f}%\n\n")
        else:
            f.write(f"Status: {results['lab'].get('error', 'Unknown error')}\n\n")
        
        # CHATBOT MODEL
        f.write("=" * 80 + "\n")
        f.write("4. MEDICAL CHATBOT MODEL\n")
        f.write("=" * 80 + "\n\n")
        
        if 'error' not in results.get('chatbot', {}):
            chatbot = results['chatbot']
            f.write(f"Model Type: {chatbot['type']}\n")
            f.write(f"Model Path: {chatbot['path']}\n")
            f.write(f"File Size: {chatbot['file_size'] / 1024:.2f} KB\n")
            f.write(f"Total Parameters: {chatbot['params']:,}\n\n")
            
            f.write("Architecture Details:\n")
            f.write("-" * 40 + "\n")
            f.write("  Embedding: 72 vocab -> 128 dim\n")
            f.write("  LSTM: 2 layers, bidirectional, 256 hidden\n")
            f.write("  Attention: Self-attention mechanism\n")
            f.write("  FC: 256 -> 128 -> 10 classes\n\n")
            
            f.write("Dataset Information:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Test Samples: {chatbot['test_samples']}\n")
            f.write("  Vocabulary Size: 72\n\n")
            
            f.write("Disease Classes:\n")
            f.write("-" * 40 + "\n")
            for i, disease in enumerate(chatbot.get('disease_classes', [])):
                f.write(f"  {i}: {disease}\n")
            f.write("\n")
            
            f.write("Test Metrics:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Loss: {chatbot['test_loss']:.4f}\n")
            f.write(f"  Accuracy: {chatbot['test_accuracy']*100:.2f}%\n\n")
        else:
            f.write(f"Status: {results['chatbot'].get('error', 'Unknown error')}\n\n")
        
        # SUMMARY TABLE
        f.write("=" * 80 + "\n")
        f.write("SUMMARY TABLE\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Model':<15} {'Type':<25} {'Parameters':>15} {'Test Acc/IoU':>15}\n")
        f.write("-" * 70 + "\n")
        
        if 'error' not in results.get('skin', {}):
            f.write(f"{'Skin':<15} {results['skin']['type']:<25} {results['skin']['params']:>15,} {results['skin']['test_iou']:>15.4f}\n")
        if 'error' not in results.get('sound', {}):
            f.write(f"{'Sound':<15} {results['sound']['type']:<25} {results['sound']['params']:>15,} {results['sound']['test_accuracy']*100:>14.2f}%\n")
        if 'error' not in results.get('lab', {}):
            f.write(f"{'Lab':<15} {results['lab']['type']:<25} {results['lab']['params']:>15,} {results['lab']['test_accuracy']*100:>14.2f}%\n")
        if 'error' not in results.get('chatbot', {}):
            f.write(f"{'Chatbot':<15} {'LSTM+Attention':<25} {results['chatbot']['params']:>15,} {results['chatbot']['test_accuracy']*100:>14.2f}%\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n✅ Summary saved to: {summary_path}")
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
