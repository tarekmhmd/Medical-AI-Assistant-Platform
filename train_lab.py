"""
Lab Data Training Script (No External Dependencies)
===================================================
Trains an MLP model on lab/medical CSV data for classification tasks.
Uses only standard library + torch + numpy.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# MLP MODEL ARCHITECTURE
# =============================================================================

class LabMLP(nn.Module):
    """
    Multi-Layer Perceptron for lab data classification.
    
    Architecture:
    - Input layer: variable size based on features
    - Hidden layers: 128 -> 64 -> 32
    - Output layer: variable size based on classes
    - Batch normalization and dropout for regularization
    """
    
    def __init__(self, input_dim, num_classes, hidden_sizes=[128, 64, 32], dropout=0.3):
        super(LabMLP, self).__init__()
        
        layers = []
        prev_size = input_dim
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


# =============================================================================
# DATASET CLASS
# =============================================================================

class LabDataset(Dataset):
    """Dataset for lab CSV data."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# DATA PREPARATION (No pandas)
# =============================================================================

def load_csv_simple(filepath):
    """Load CSV file without pandas."""
    data = []
    with open(filepath, 'r', newline='', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data


def prepare_diabetes_data(filepath):
    """Prepare diabetes dataset for training."""
    print(f"\n📊 Loading diabetes data from: {filepath}")
    
    data = load_csv_simple(filepath)
    
    # Convert to numpy
    X = []
    y = []
    
    for row in data:
        try:
            # All values should be numeric
            features = [float(x) for x in row[:-1]]
            target = int(float(row[-1]))
            X.append(features)
            y.append(target)
        except (ValueError, IndexError):
            continue
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    print(f"   Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"   Classes: {np.unique(y)}")
    
    return X, y


def normalize_data(X_train, X_val, X_test):
    """Normalize features using z-score normalization."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std
    
    return X_train_norm, X_val_norm, X_test_norm, mean, std


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    
    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    return total_loss / len(dataloader), correct / total


def train_lab_model():
    """Main training function."""
    print("=" * 70)
    print("LAB DATA MLP TRAINING")
    print("=" * 70)
    
    # Paths
    train_path = r"D:\project 2\data\lab\train\diabetes_data_train.csv"
    val_path = r"D:\project 2\data\lab\val\diabetes_data_val.csv"
    test_path = r"D:\project 2\data\lab\test\diabetes_data_test.csv"
    save_path = r"D:\project 2\checkpoints\lab_model.pth"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Load data
    X_train, y_train = prepare_diabetes_data(train_path)
    X_val, y_val = prepare_diabetes_data(val_path)
    X_test, y_test = prepare_diabetes_data(test_path)
    
    # Normalize features
    X_train, X_val, X_test, mean, std = normalize_data(X_train, X_val, X_test)
    
    # Dataset info
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    print(f"\n📈 Dataset Summary:")
    print(f"   Input features: {input_dim}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Create datasets and dataloaders
    train_dataset = LabDataset(X_train, y_train)
    val_dataset = LabDataset(X_val, y_val)
    test_dataset = LabDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Device: {device}")
    
    # Create model
    print("\n🏗️ Building MLP model...")
    model = LabMLP(input_dim=input_dim, num_classes=num_classes, 
                   hidden_sizes=[128, 64, 32], dropout=0.3).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training
    print("\n🚀 Starting training...")
    print("-" * 70)
    
    epochs = 50
    best_val_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'input_dim': input_dim,
                'num_classes': num_classes,
                'val_acc': val_acc,
                'scaler_mean': mean.tolist(),
                'scaler_std': std.tolist(),
            }, save_path)
    
    print("-" * 70)
    
    # Test evaluation
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\n✅ Training complete!")
    print(f"📁 Model saved to: {save_path}")
    print(f"🏆 Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"📊 Test Accuracy: {test_acc*100:.2f}%")
    
    return model, input_dim, num_classes


if __name__ == "__main__":
    train_lab_model()