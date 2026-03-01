"""
Medical Chatbot Training Script
===============================
Trains an LSTM-based model for medical symptom-to-disease classification.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from collections import Counter

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "embed_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 0.001,
    "max_seq_len": 20,
    "min_word_freq": 1,
}

# =============================================================================
# DATA PREPARATION
# =============================================================================

class Vocabulary:
    """Simple vocabulary for text processing."""
    
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<EOS>": 2}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<EOS>"}
        self.word_counts = Counter()
        self.vocab_size = 3
    
    def add_sentence(self, sentence):
        """Add words from a sentence to vocabulary."""
        for word in sentence.lower().split():
            self.word_counts[word] += 1
    
    def build_vocab(self, min_freq=1):
        """Build vocabulary from word counts."""
        for word, count in self.word_counts.items():
            if count >= min_freq and word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
    
    def encode(self, sentence, max_len=20):
        """Encode sentence to indices."""
        indices = [self.word2idx.get(w, 1) for w in sentence.lower().split()]
        # Pad or truncate
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        return indices
    
    def decode(self, indices):
        """Decode indices to sentence."""
        words = [self.idx2word.get(i, "<UNK>") for i in indices if i != 0]
        return " ".join(words)


def load_medical_data(base_path):
    """Load medical database and create training pairs."""
    print("=" * 70)
    print("MEDICAL CHATBOT TRAINING")
    print("=" * 70)
    
    # Load disease database
    db_path = os.path.join(base_path, "processed_data", "databases", "disease_database.json")
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return None
    
    with open(db_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n📊 Loading medical database...")
    
    # Create training pairs: (symptoms, disease)
    training_pairs = []
    disease_names = []
    
    for disease in data['diseases']:
        name = disease['name']
        disease_names.append(name)
        symptoms = disease['symptoms']
        medications = disease.get('medications', [])
        severity = disease.get('severity', 'unknown')
        duration = disease.get('duration', 'unknown')
        
        # Create multiple training examples per disease
        # 1. Full symptom list -> disease
        symptom_text = " ".join(symptoms)
        training_pairs.append((symptom_text, name))
        
        # 2. Individual symptoms -> disease
        for symptom in symptoms:
            training_pairs.append((symptom, name))
        
        # 3. Pairs of symptoms -> disease
        for i in range(len(symptoms)):
            for j in range(i+1, len(symptoms)):
                pair_text = f"{symptoms[i]} {symptoms[j]}"
                training_pairs.append((pair_text, name))
        
        # 4. Questions about disease
        training_pairs.append((f"what are the symptoms of {name}", name))
        training_pairs.append((f"symptoms of {name}", name))
        training_pairs.append((f"i have {symptoms[0]}", name))
    
    print(f"   Diseases: {len(disease_names)}")
    print(f"   Training pairs: {len(training_pairs)}")
    
    return training_pairs, disease_names


def prepare_datasets(training_pairs, disease_names, vocab, max_len=20):
    """Prepare datasets for training."""
    # Build vocabulary
    for symptoms, _ in training_pairs:
        vocab.add_sentence(symptoms)
    
    vocab.build_vocab(min_freq=1)
    print(f"\n📖 Vocabulary size: {vocab.vocab_size}")
    
    # Create label encoder
    disease2idx = {name: i for i, name in enumerate(disease_names)}
    num_classes = len(disease_names)
    
    # Encode data
    X = []
    y = []
    
    for symptoms, disease in training_pairs:
        encoded = vocab.encode(symptoms, max_len)
        X.append(encoded)
        y.append(disease2idx[disease])
    
    X = np.array(X, dtype=np.int64)
    y = np.array(y, dtype=np.int64)
    
    # Split data
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"\n📈 Dataset splits:")
    print(f"   Training: {len(X_train)}")
    print(f"   Validation: {len(X_val)}")
    print(f"   Test: {len(X_test)}")
    print(f"   Classes: {num_classes}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), num_classes


# =============================================================================
# DATASET CLASS
# =============================================================================

class ChatbotDataset(Dataset):
    """Dataset for chatbot training."""
    
    def __init__(self, X, y):
        self.X = torch.LongTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class MedicalChatbot(nn.Module):
    """
    LSTM-based model for medical symptom classification.
    
    Architecture:
    - Embedding layer: converts word indices to dense vectors
    - Bidirectional LSTM: processes sequences in both directions
    - Attention mechanism: focuses on important words
    - Fully connected layers: classification head
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super(MedicalChatbot, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim // 2,  # Bidirectional will double this
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Classification head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x):
        # Embedding: [batch, seq_len] -> [batch, seq_len, embed_dim]
        embedded = self.embedding(x)
        
        # LSTM: [batch, seq_len, embed_dim] -> [batch, seq_len, hidden_dim]
        lstm_out, _ = self.lstm(embedded)
        
        # Attention: compute attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Apply attention
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Classification
        x = self.fc1(context)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


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


def train_chatbot():
    """Main training function."""
    # Paths
    base_path = r"D:\project 2"
    save_path = r"D:\project 2\checkpoints\chatbot_model.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Load data
    result = load_medical_data(base_path)
    if result is None:
        return None
    
    training_pairs, disease_names = result
    
    # Create vocabulary
    vocab = Vocabulary()
    
    # Prepare datasets
    (X_train, y_train), (X_val, y_val), (X_test, y_test), num_classes = prepare_datasets(
        training_pairs, disease_names, vocab, CONFIG["max_seq_len"]
    )
    
    # Create datasets
    train_dataset = ChatbotDataset(X_train, y_train)
    val_dataset = ChatbotDataset(X_val, y_val)
    test_dataset = ChatbotDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"])
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Device: {device}")
    
    # Create model
    print("\n🏗️ Building Medical Chatbot model...")
    model = MedicalChatbot(
        vocab_size=vocab.vocab_size,
        embed_dim=CONFIG["embed_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_layers"],
        num_classes=num_classes,
        dropout=CONFIG["dropout"]
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training
    print("\n🚀 Starting training...")
    print("-" * 70)
    
    best_val_acc = 0.0
    epochs = CONFIG["epochs"]
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        if epoch % 20 == 0 or epoch == 1:
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
                'vocab_size': vocab.vocab_size,
                'embed_dim': CONFIG["embed_dim"],
                'hidden_dim': CONFIG["hidden_dim"],
                'num_layers': CONFIG["num_layers"],
                'num_classes': num_classes,
                'val_acc': val_acc,
                'word2idx': vocab.word2idx,
                'disease_names': disease_names,
                'max_seq_len': CONFIG["max_seq_len"],
            }, save_path)
    
    print("-" * 70)
    
    # Test evaluation
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\n✅ Training complete!")
    print(f"📁 Model saved to: {save_path}")
    print(f"🏆 Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"📊 Test Accuracy: {test_acc*100:.2f}%")
    
    return model


if __name__ == "__main__":
    train_chatbot()
