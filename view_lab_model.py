"""
Lab MLP Model Viewer
====================
Loads and displays the architecture of the trained MLP model for lab data classification.

This script:
1. Rebuilds the LabMLP architecture
2. Loads trained weights from checkpoints/lab_model.pth
3. Prints full architecture
4. Prints parameters per layer
5. Prints total parameters

Run this script independently to view model details without retraining.
"""

import torch
import torch.nn as nn

# =============================================================================
# MODEL ARCHITECTURE: MLP for Lab Data Classification
# =============================================================================

class LabMLP(nn.Module):
    """
    Multi-Layer Perceptron for lab data classification.
    
    Architecture:
    - Input layer: accepts lab test features
    - Hidden layers: 128 -> 64 -> 32 neurons
    - Output layer: classification logits
    - Batch normalization after each hidden layer
    - Dropout for regularization (0.3)
    - ReLU activation functions
    
    Input: Feature vector [batch, input_dim]
    Output: Class logits [batch, num_classes]
    """
    
    def __init__(self, input_dim=8, num_classes=2, hidden_sizes=[128, 64, 32], dropout=0.3):
        """
        Initialize LabMLP model.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout probability
        """
        super(LabMLP, self).__init__()
        
        layers = []
        prev_size = input_dim
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            # Batch normalization
            layers.append(nn.BatchNorm1d(hidden_size))
            # ReLU activation
            layers.append(nn.ReLU())
            # Dropout for regularization
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer (no activation - CrossEntropyLoss applies softmax)
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor [batch, input_dim]
            
        Returns:
            Output logits [batch, num_classes]
        """
        return self.network(x)


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
        'Linear': 0,
        'BatchNorm1d': 0,
        'ReLU': 0,
        'Dropout': 0
    }
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layer_counts['Linear'] += 1
        elif isinstance(module, nn.BatchNorm1d):
            layer_counts['BatchNorm1d'] += 1
        elif isinstance(module, nn.ReLU):
            layer_counts['ReLU'] += 1
        elif isinstance(module, nn.Dropout):
            layer_counts['Dropout'] += 1
    
    for layer_type, count in layer_counts.items():
        print(f"{layer_type}: {count} layers")


def main():
    print("=" * 70)
    print("LAB DATA MLP MODEL VIEWER")
    print("=" * 70)
    
    # Model path
    model_path = r"D:\project 2\checkpoints\lab_model.pth"
    
    # Check file existence and size
    import os
    print(f"\n📂 Loading model from: {model_path}")
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        print(f"   File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    else:
        print("   ❌ Model file not found!")
        return
    
    # Load checkpoint to get model configuration
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            input_dim = checkpoint.get('input_dim', 8)
            num_classes = checkpoint.get('num_classes', 2)
            
            print(f"\n📋 Model Configuration:")
            print(f"   Input features: {input_dim}")
            print(f"   Output classes: {num_classes}")
            
            # Create model instance
            model = LabMLP(input_dim=input_dim, num_classes=num_classes, 
                          hidden_sizes=[128, 64, 32], dropout=0.3)
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            print("\n✅ Model weights loaded successfully!")
            
            # Print training info if available
            if 'epoch' in checkpoint:
                print(f"   Trained for {checkpoint['epoch']} epochs")
            if 'val_acc' in checkpoint:
                print(f"   Validation Accuracy: {checkpoint['val_acc']*100:.2f}%")
        else:
            # Old format - assume default config
            model = LabMLP(input_dim=8, num_classes=2)
            model.load_state_dict(checkpoint)
            print("\n✅ Model weights loaded successfully!")
            
    except Exception as e:
        print(f"\n⚠️ Could not load weights: {e}")
        print("   Displaying untrained model architecture...")
        model = LabMLP(input_dim=8, num_classes=2)
    
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
    print(f"{'Model Type:':<25} LabMLP (Multi-Layer Perceptron)")
    print(f"{'Input Shape:':<25} [batch, {model.network[0].in_features}] (Features)")
    print(f"{'Output Shape:':<25} [batch, {model.network[-1].out_features}] (Classes)")
    print(f"{'Hidden Layers:':<25} 3 (128 -> 64 -> 32)")
    print(f"{'Total Parameters:':<25} {total_params:,}")
    print(f"{'Trainable Parameters:':<25} {trainable_params:,}")
    print(f"{'Model Size:':<25} {model_size/1024:.2f} KB (float32)")
    print(f"{'Task:':<25} Binary Classification")
    print(f"{'Loss Function:':<25} CrossEntropyLoss")
    print(f"{'Dataset:':<25} Diabetes Dataset")
    print("=" * 70)
    
    # Print architecture diagram
    print("\n" + "=" * 70)
    print("ARCHITECTURE DIAGRAM")
    print("=" * 70)
    print("""
    Input: [batch, 8] (Lab Features)
           │
    ┌──────┴──────┐
    │  Linear     │ 8 → 128
    │  BatchNorm  │
    │  ReLU       │
    │  Dropout    │ (0.3)
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │  Linear     │ 128 → 64
    │  BatchNorm  │
    │  ReLU       │
    │  Dropout    │ (0.3)
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │  Linear     │ 64 → 32
    │  BatchNorm  │
    │  ReLU       │
    │  Dropout    │ (0.3)
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │  Linear     │ 32 → 2
    └──────┬──────┘
           │
    Output: [batch, 2] (Class Logits)
    """)
    
    # Print input features description
    print("=" * 70)
    print("INPUT FEATURES (Diabetes Dataset)")
    print("=" * 70)
    print("""
    Feature 0: Number of pregnancies
    Feature 1: Plasma glucose concentration
    Feature 2: Diastolic blood pressure (mm Hg)
    Feature 3: Triceps skin fold thickness (mm)
    Feature 4: 2-Hour serum insulin (mu U/ml)
    Feature 5: Body mass index (BMI)
    Feature 6: Diabetes pedigree function
    Feature 7: Age (years)
    
    Target: Diabetes diagnosis (0 = No, 1 = Yes)
    """)
    
    print("=" * 70)
    print("✅ MODEL VIEWER COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()