"""
Sound Model Viewer
==================
Loads and displays the architecture of the trained respiratory sound CNN classifier.

Run this script to view the model architecture, layer parameters, and total parameter count.
"""

import torch
import torch.nn as nn

# =============================================================================
# MODEL ARCHITECTURE (Respiratory Sound CNN Classifier)
# =============================================================================

class RespiratorySoundCNN(nn.Module):
    """
    CNN architecture for respiratory sound classification.
    Input: Mel-spectrogram [batch, 1, n_mels, time]
    Output: Class probabilities [batch, num_classes]
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


def count_parameters(model):
    """Count parameters per layer and total."""
    print("\n" + "=" * 70)
    print("PARAMETERS PER LAYER")
    print("=" * 70)
    print(f"{'Layer Name':<40} {'Parameters':>15}")
    print("-" * 70)
    
    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        print(f"{name:<40} {param_count:>15,}")
    
    print("-" * 70)
    print(f"{'TOTAL PARAMETERS':<40} {total_params:>15,}")
    print("=" * 70)
    
    return total_params


def main():
    print("=" * 70)
    print("RESPIRATORY SOUND CNN MODEL VIEWER")
    print("=" * 70)
    
    # Model path
    model_path = r"D:\project 2\checkpoints\sound_model.pth"
    
    # Create model instance
    model = RespiratorySoundCNN(num_classes=2, dropout=0.3)
    
    # Load weights
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\n✅ Model loaded successfully from: {model_path}")
            if 'val_acc' in checkpoint:
                print(f"   Validation Accuracy: {checkpoint['val_acc']*100:.2f}%")
            if 'epoch' in checkpoint:
                print(f"   Trained for {checkpoint['epoch']} epochs")
        else:
            model.load_state_dict(checkpoint)
            print(f"\n✅ Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"\n⚠️ Could not load weights: {e}")
        print("   Displaying untrained model architecture...")
    
    # Print full architecture
    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE")
    print("=" * 70)
    print(model)
    
    # Print parameters per layer
    total_params = count_parameters(model)
    
    # Print model summary
    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(f"Model Type:       RespiratorySoundCNN")
    print(f"Input Shape:      [batch, 1, 64, 128] (Mel-Spectrogram)")
    print(f"Output Shape:     [batch, 2] (Class Logits)")
    print(f"Total Parameters: {total_params:,}")
    print(f"Model Size:       {total_params * 4 / (1024*1024):.2f} MB (float32)")
    print(f"Classes:          2 (Healthy vs Unhealthy)")
    print("=" * 70)


if __name__ == "__main__":
    main()
