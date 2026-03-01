"""
Medical Chatbot Model Viewer
============================
Loads and displays the architecture of the trained LSTM-based medical chatbot model.

This script:
1. Rebuilds the MedicalChatbot architecture
2. Loads trained weights from checkpoints/chatbot_model.pth
3. Prints full architecture
4. Prints parameters per layer
5. Prints total parameters

Run this script independently to view model details without retraining.
"""

import torch
import torch.nn as nn

# =============================================================================
# MODEL ARCHITECTURE: Medical Chatbot (LSTM with Attention)
# =============================================================================

class MedicalChatbot(nn.Module):
    """
    LSTM-based model for medical symptom classification.
    
    Architecture:
    - Embedding layer: converts word indices to dense vectors
    - Bidirectional LSTM: processes sequences in both directions
    - Attention mechanism: focuses on important words in the input
    - Fully connected layers: classification head
    
    Input: Token indices [batch, seq_len]
    Output: Disease class logits [batch, num_classes]
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        """
        Initialize MedicalChatbot model.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            num_classes: Number of output classes (diseases)
            dropout: Dropout probability
        """
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
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor [batch, seq_len] of token indices
            
        Returns:
            Output logits [batch, num_classes]
        """
        # Embedding: [batch, seq_len] -> [batch, seq_len, embed_dim]
        embedded = self.embedding(x)
        
        # LSTM: [batch, seq_len, embed_dim] -> [batch, seq_len, hidden_dim]
        lstm_out, _ = self.lstm(embedded)
        
        # Attention: compute attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Apply attention to get context vector
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Classification
        x = self.fc1(context)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def count_parameters(model):
    """Count and display parameters per layer."""
    print("\n" + "=" * 70)
    print("PARAMETERS PER LAYER")
    print("=" * 70)
    print(f"{'Layer Name':<50} {'Parameters':>15}")
    print("-" * 70)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        print(f"{name:<50} {param_count:>15,}")
    
    print("-" * 70)
    print(f"{'TOTAL PARAMETERS':<50} {total_params:>15,}")
    print(f"{'TRAINABLE PARAMETERS':<50} {trainable_params:>15,}")
    print("=" * 70)
    
    return total_params, trainable_params


def get_model_size(model):
    """Calculate model size in memory."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    return total_size


def print_layer_details(model):
    """Print detailed information about each layer type."""
    print("\n" + "=" * 70)
    print("LAYER TYPE COUNTS")
    print("=" * 70)
    
    layer_counts = {
        'Embedding': 0,
        'LSTM': 0,
        'Linear': 0,
        'BatchNorm1d': 0,
        'Dropout': 0
    }
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            layer_counts['Embedding'] += 1
        elif isinstance(module, nn.LSTM):
            layer_counts['LSTM'] += 1
        elif isinstance(module, nn.Linear):
            layer_counts['Linear'] += 1
        elif isinstance(module, nn.BatchNorm1d):
            layer_counts['BatchNorm1d'] += 1
        elif isinstance(module, nn.Dropout):
            layer_counts['Dropout'] += 1
    
    for layer_type, count in layer_counts.items():
        print(f"{layer_type}: {count} layers")


def main():
    print("=" * 70)
    print("MEDICAL CHATBOT MODEL VIEWER")
    print("=" * 70)
    
    # Model path
    model_path = r"D:\project 2\checkpoints\chatbot_model.pth"
    
    # Check file existence and size
    import os
    print(f"\n\U0001F4C2 Loading model from: {model_path}")
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        print(f"   File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    else:
        print("   \u274C Model file not found!")
        return
    
    # Load checkpoint to get model configuration
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            vocab_size = checkpoint.get('vocab_size', 72)
            embed_dim = checkpoint.get('embed_dim', 128)
            hidden_dim = checkpoint.get('hidden_dim', 256)
            num_layers = checkpoint.get('num_layers', 2)
            num_classes = checkpoint.get('num_classes', 10)
            max_seq_len = checkpoint.get('max_seq_len', 20)
            disease_names = checkpoint.get('disease_names', [])
            
            print(f"\n\U0001F4CB Model Configuration:")
            print(f"   Vocabulary size: {vocab_size}")
            print(f"   Embedding dimension: {embed_dim}")
            print(f"   Hidden dimension: {hidden_dim}")
            print(f"   LSTM layers: {num_layers}")
            print(f"   Output classes: {num_classes}")
            print(f"   Max sequence length: {max_seq_len}")
            
            # Create model instance
            model = MedicalChatbot(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=0.3
            )
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            print("\n\u2705 Model weights loaded successfully!")
            
            # Print training info if available
            if 'epoch' in checkpoint:
                print(f"   Trained for {checkpoint['epoch']} epochs")
            if 'val_acc' in checkpoint:
                print(f"   Validation Accuracy: {checkpoint['val_acc']*100:.2f}%")
        else:
            # Old format - assume default config
            model = MedicalChatbot(vocab_size=72, embed_dim=128, hidden_dim=256, 
                                   num_layers=2, num_classes=10)
            model.load_state_dict(checkpoint)
            print("\n\u2705 Model weights loaded successfully!")
            
    except Exception as e:
        print(f"\n\u26A0\uFE0F Could not load weights: {e}")
        print("   Displaying untrained model architecture...")
        model = MedicalChatbot(vocab_size=72, embed_dim=128, hidden_dim=256, 
                               num_layers=2, num_classes=10)
    
    # Set to evaluation mode
    model.eval()
    
    # Print full architecture
    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE")
    print("=" * 70)
    print(model)
    
    # Print parameters per layer
    total_params, trainable_params = count_parameters(model)
    
    # Calculate model size
    model_size = get_model_size(model)
    
    # Print layer details
    print_layer_details(model)
    
    # Print model summary
    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(f"{'Model Type:':<25} MedicalChatbot (LSTM + Attention)")
    print(f"{'Input Shape:':<25} [batch, {max_seq_len if 'max_seq_len' in dir() else 20}] (Token Indices)")
    print(f"{'Output Shape:':<25} [batch, {num_classes}] (Disease Classes)")
    print(f"{'Embedding Dim:':<25} {embed_dim}")
    print(f"{'Hidden Dim:':<25} {hidden_dim}")
    print(f"{'LSTM Layers:':<25} {num_layers}")
    print(f"{'Bidirectional:':<25} Yes")
    print(f"{'Total Parameters:':<25} {total_params:,}")
    print(f"{'Trainable Parameters:':<25} {trainable_params:,}")
    print(f"{'Model Size:':<25} {model_size/1024:.2f} KB (float32)")
    print(f"{'Task:':<25} Symptom-to-Disease Classification")
    print(f"{'Loss Function:':<25} CrossEntropyLoss")
    print("=" * 70)
    
    # Print architecture diagram
    print("\n" + "=" * 70)
    print("ARCHITECTURE DIAGRAM")
    print("=" * 70)
    print("""
    Input: [batch, seq_len] (Token Indices)
           │
    ┌──────┴──────┐
    │  Embedding  │ vocab_size → embed_dim
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │ Bi-LSTM     │ 2 layers, hidden_dim
    │ (Forward)   │
    │ Bi-LSTM     │
    │ (Backward)  │
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │ Attention   │ context = Σ(αᵢ × hᵢ)
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │  Linear     │ hidden_dim → hidden_dim/2
    │  BatchNorm  │
    │  ReLU       │
    │  Dropout    │
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │  Linear     │ hidden_dim/2 → num_classes
    └──────┬──────┘
           │
    Output: [batch, num_classes] (Disease Logits)
    """)
    
    # Print disease classes
    if isinstance(checkpoint, dict) and 'disease_names' in checkpoint:
        print("=" * 70)
        print("DISEASE CLASSES")
        print("=" * 70)
        for i, name in enumerate(checkpoint['disease_names']):
            print(f"   {i}: {name}")
    
    print("\n" + "=" * 70)
    print("\u2705 MODEL VIEWER COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()