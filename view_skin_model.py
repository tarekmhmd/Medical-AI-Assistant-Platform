"""
Skin Segmentation Model Viewer
==============================
Loads and displays the architecture of the trained U-Net model for skin lesion segmentation.

This script:
1. Rebuilds the U-Net architecture
2. Loads trained weights from checkpoints/skin_model.pth
3. Prints full architecture
4. Prints parameters per layer
5. Prints total parameters

Run this script independently to view model details without retraining.
"""

import torch
import torch.nn as nn

# =============================================================================
# MODEL ARCHITECTURE: U-Net for Skin Lesion Segmentation
# =============================================================================

class SkinUNet(nn.Module):
    """
    U-Net architecture for skin lesion segmentation.
    
    Architecture:
    - Encoder: Conv layers that downsample the input
    - Decoder: Upsampling layers that reconstruct the segmentation mask
    - Skip connections between encoder and decoder
    
    Input: RGB image [batch, 3, H, W]
    Output: Binary segmentation mask [batch, 1, H, W]
    """
    
    def __init__(self, in_channels=3, out_channels=1, init_features=16):
        """
        Initialize U-Net model.
        
        Args:
            in_channels: Number of input channels (3 for RGB images)
            out_channels: Number of output channels (1 for binary mask)
            init_features: Number of features in first conv layer
        """
        super(SkinUNet, self).__init__()
        
        features = init_features
        
        # Encoder (contracting path)
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")
        
        # Decoder (expansive path)
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")
        
        # Final output layer
        self.conv_final = nn.Conv2d(features, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder path with skip connections
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
        
        # Final output
        return torch.sigmoid(self.conv_final(dec1))
    
    @staticmethod
    def _block(in_channels, features, name):
        """Create a convolutional block with two conv layers."""
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )


class SimpleUNet(nn.Module):
    """
    Simplified U-Net for quick training (used in actual training).
    
    Input: RGB image [batch, 3, H, W]
    Output: Segmentation mask logits [batch, 1, H, W]
    """
    
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 1, 3, padding=1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
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


def main():
    print("=" * 70)
    print("SKIN LESION SEGMENTATION MODEL VIEWER")
    print("=" * 70)
    
    # Model path
    model_path = r"D:\project 2\checkpoints\skin_model.pth"
    
    # First, try to load the trained model
    print(f"\n📂 Loading model from: {model_path}")
    
    # Check file existence and size
    import os
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        print(f"   File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    
    # Create SimpleUNet (matches the trained model architecture)
    model = SimpleUNet()
    
    # Load weights
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
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
    
    # Print model summary
    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(f"{'Model Type:':<25} SimpleUNet (U-Net variant)")
    print(f"{'Input Shape:':<25} [batch, 3, 256, 256] (RGB Image)")
    print(f"{'Output Shape:':<25} [batch, 1, 256, 256] (Segmentation Mask)")
    print(f"{'Total Parameters:':<25} {total_params:,}")
    print(f"{'Trainable Parameters:':<25} {trainable_params:,}")
    print(f"{'Model Size:':<25} {model_size/1024:.2f} KB (float32)")
    print(f"{'Task:':<25} Binary Semantic Segmentation")
    print(f"{'Loss Function:':<25} BCEWithLogitsLoss")
    print("=" * 70)
    
    # Print layer details
    print("\n" + "=" * 70)
    print("LAYER DETAILS")
    print("=" * 70)
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            print(f"Conv2d: {name}")
            print(f"  - in_channels: {layer.in_channels}")
            print(f"  - out_channels: {layer.out_channels}")
            print(f"  - kernel_size: {layer.kernel_size}")
            print(f"  - stride: {layer.stride}")
            print(f"  - padding: {layer.padding}")
        elif isinstance(layer, nn.ReLU):
            print(f"ReLU: {name} (inplace={layer.inplace})")
    
    # Show alternative full U-Net architecture
    print("\n" + "=" * 70)
    print("ALTERNATIVE: FULL U-NET ARCHITECTURE (Not trained)")
    print("=" * 70)
    print("For better performance, consider using the full U-Net architecture:")
    full_unet = SkinUNet(in_channels=3, out_channels=1, init_features=16)
    full_params = sum(p.numel() for p in full_unet.parameters())
    print(f"Full U-Net Parameters: {full_params:,}")
    print(f"Full U-Net Size: {full_params * 4 / (1024*1024):.2f} MB (float32)")
    
    print("\n" + "=" * 70)
    print("✅ MODEL VIEWER COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()