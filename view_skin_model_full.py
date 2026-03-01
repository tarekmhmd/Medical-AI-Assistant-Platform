"""
Full U-Net Model Viewer for Skin Lesion Segmentation
=====================================================
Loads and displays the architecture of the trained Full U-Net model.

This script:
1. Rebuilds the Full U-Net architecture
2. Loads trained weights from checkpoints/skin_model_full.pth
3. Prints full architecture
4. Prints parameters per layer
5. Prints total parameters

Run this script independently to view model details without retraining.
"""

import torch
import torch.nn as nn

# =============================================================================
# MODEL ARCHITECTURE: Full U-Net for Skin Lesion Segmentation
# =============================================================================

class FullUNet(nn.Module):
    """
    Complete U-Net architecture for skin lesion segmentation.
    
    Architecture Details:
    - Encoder: 4 blocks with MaxPool downsampling
    - Bottleneck: Deepest layer
    - Decoder: 4 blocks with ConvTranspose upsampling
    - Skip connections between encoder and decoder
    - Batch normalization for stable training
    
    Input: RGB image [batch, 3, H, W]
    Output: Segmentation mask logits [batch, 1, H, W]
    """
    
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        """
        Initialize Full U-Net model.
        
        Args:
            in_channels: Number of input channels (3 for RGB images)
            out_channels: Number of output channels (1 for binary mask)
            init_features: Number of features in first encoder block
        """
        super(FullUNet, self).__init__()
        
        features = init_features
        
        # ===== Encoder (Contracting Path) =====
        # Each encoder block: 2x (Conv -> BatchNorm -> ReLU)
        # Followed by MaxPool 2x2
        
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ===== Bottleneck =====
        # Deepest part of the network
        self.bottleneck = self._block(features * 8, features * 16)
        
        # ===== Decoder (Expansive Path) =====
        # Each decoder block: Upsample -> Concat with skip -> 2x (Conv -> BN -> ReLU)
        
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features)
        
        # ===== Final Output Layer =====
        # 1x1 Conv to produce segmentation map
        self.conv_final = nn.Conv2d(features, out_channels, kernel_size=1)
        
    def forward(self, x):
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor [batch, 3, H, W]
            
        Returns:
            Output tensor [batch, 1, H, W] with segmentation logits
        """
        # Encoder path - store outputs for skip connections
        enc1 = self.encoder1(x)                           # [batch, 32, H, W]
        enc2 = self.encoder2(self.pool1(enc1))            # [batch, 64, H/2, W/2]
        enc3 = self.encoder3(self.pool2(enc2))            # [batch, 128, H/4, W/4]
        enc4 = self.encoder4(self.pool3(enc3))            # [batch, 256, H/8, W/8]
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))    # [batch, 512, H/16, W/16]
        
        # Decoder path with skip connections
        dec4 = self.upconv4(bottleneck)                   # [batch, 256, H/8, W/8]
        dec4 = torch.cat((dec4, enc4), dim=1)             # [batch, 512, H/8, W/8]
        dec4 = self.decoder4(dec4)                        # [batch, 256, H/8, W/8]
        
        dec3 = self.upconv3(dec4)                         # [batch, 128, H/4, W/4]
        dec3 = torch.cat((dec3, enc3), dim=1)             # [batch, 256, H/4, W/4]
        dec3 = self.decoder3(dec3)                        # [batch, 128, H/4, W/4]
        
        dec2 = self.upconv2(dec3)                         # [batch, 64, H/2, W/2]
        dec2 = torch.cat((dec2, enc2), dim=1)             # [batch, 128, H/2, W/2]
        dec2 = self.decoder2(dec2)                        # [batch, 64, H/2, W/2]
        
        dec1 = self.upconv1(dec2)                         # [batch, 32, H, W]
        dec1 = torch.cat((dec1, enc1), dim=1)             # [batch, 64, H, W]
        dec1 = self.decoder1(dec1)                        # [batch, 32, H, W]
        
        # Final output
        return self.conv_final(dec1)                      # [batch, 1, H, W]
    
    @staticmethod
    def _block(in_channels, features):
        """
        Create a convolutional block.
        
        Structure: Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU
        
        Args:
            in_channels: Number of input channels
            features: Number of output features
            
        Returns:
            nn.Sequential block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )


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
    """Print detailed information about each layer."""
    print("\n" + "=" * 70)
    print("LAYER DETAILS")
    print("=" * 70)
    
    layer_counts = {
        'Conv2d': 0,
        'ConvTranspose2d': 0,
        'BatchNorm2d': 0,
        'MaxPool2d': 0,
        'ReLU': 0
    }
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and not isinstance(module, nn.ConvTranspose2d):
            layer_counts['Conv2d'] += 1
        elif isinstance(module, nn.ConvTranspose2d):
            layer_counts['ConvTranspose2d'] += 1
        elif isinstance(module, nn.BatchNorm2d):
            layer_counts['BatchNorm2d'] += 1
        elif isinstance(module, nn.MaxPool2d):
            layer_counts['MaxPool2d'] += 1
        elif isinstance(module, nn.ReLU):
            layer_counts['ReLU'] += 1
    
    for layer_type, count in layer_counts.items():
        print(f"{layer_type}: {count} layers")


def main():
    print("=" * 70)
    print("FULL U-NET MODEL VIEWER - SKIN LESION SEGMENTATION")
    print("=" * 70)
    
    # Model path
    model_path = r"D:\project 2\checkpoints\skin_model_full.pth"
    
    # Check file existence and size
    import os
    print(f"\n📂 Loading model from: {model_path}")
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        print(f"   File size: {file_size:,} bytes ({file_size/(1024*1024):.2f} MB)")
    else:
        print("   ❌ Model file not found!")
        return
    
    # Create model instance
    model = FullUNet(in_channels=3, out_channels=1, init_features=32)
    
    # Load weights
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint['model_state_dict'])
            print("\n✅ Model weights loaded successfully!")
            
            # Print training info if available
            if 'epoch' in checkpoint:
                print(f"   Trained for {checkpoint['epoch']} epochs")
            if 'val_loss' in checkpoint:
                print(f"   Validation Loss: {checkpoint['val_loss']:.4f}")
            if 'val_iou' in checkpoint:
                print(f"   Validation IoU: {checkpoint['val_iou']:.4f}")
            if 'val_dice' in checkpoint:
                print(f"   Validation Dice: {checkpoint['val_dice']:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print("\n✅ Model weights loaded successfully!")
    except Exception as e:
        print(f"\n⚠️ Could not load weights: {e}")
        print("   Displaying untrained model architecture...")
    
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
    print(f"{'Model Type:':<25} Full U-Net")
    print(f"{'Input Shape:':<25} [batch, 3, 256, 256] (RGB Image)")
    print(f"{'Output Shape:':<25} [batch, 1, 256, 256] (Segmentation Mask)")
    print(f"{'Encoder Blocks:':<25} 4")
    print(f"{'Decoder Blocks:':<25} 4")
    print(f"{'Initial Features:':<25} 32")
    print(f"{'Total Parameters:':<25} {total_params:,}")
    print(f"{'Trainable Parameters:':<25} {trainable_params:,}")
    print(f"{'Model Size:':<25} {model_size/(1024*1024):.2f} MB (float32)")
    print(f"{'Task:':<25} Binary Semantic Segmentation")
    print(f"{'Loss Function:':<25} BCEWithLogitsLoss")
    print(f"{'Dataset:':<25} ISIC 2016 Task 1")
    print("=" * 70)
    
    # Print architecture diagram
    print("\n" + "=" * 70)
    print("ARCHITECTURE DIAGRAM")
    print("=" * 70)
    print("""
    Input: [batch, 3, 256, 256]
           │
    ┌──────┴──────┐
    │   Encoder 1 │ → 32 features → Skip Connection 1
    └──────┬──────┘
           │ MaxPool 2x2
    ┌──────┴──────┐
    │   Encoder 2 │ → 64 features → Skip Connection 2
    └──────┬──────┘
           │ MaxPool 2x2
    ┌──────┴──────┐
    │   Encoder 3 │ → 128 features → Skip Connection 3
    └──────┬──────┘
           │ MaxPool 2x2
    ┌──────┴──────┐
    │   Encoder 4 │ → 256 features → Skip Connection 4
    └──────┬──────┘
           │ MaxPool 2x2
    ┌──────┴──────┐
    │  Bottleneck │ → 512 features
    └──────┬──────┘
           │ Upsample 2x2
    ┌──────┴──────┐     Skip Connection 4
    │   Decoder 4 │ ←────────────────────
    └──────┬──────┘ → 256 features
           │ Upsample 2x2
    ┌──────┴──────┐     Skip Connection 3
    │   Decoder 3 │ ←────────────────────
    └──────┬──────┘ → 128 features
           │ Upsample 2x2
    ┌──────┴──────┐     Skip Connection 2
    │   Decoder 2 │ ←────────────────────
    └──────┬──────┘ → 64 features
           │ Upsample 2x2
    ┌──────┴──────┐     Skip Connection 1
    │   Decoder 1 │ ←────────────────────
    └──────┬──────┘ → 32 features
           │
    ┌──────┴──────┐
    │  Final Conv │ → 1 channel
    └──────┬──────┘
           │
    Output: [batch, 1, 256, 256]
    """)
    
    print("=" * 70)
    print("✅ MODEL VIEWER COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
