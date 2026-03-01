
"""
Full U-Net Training Script for Skin Lesion Segmentation
========================================================
Trains a complete U-Net model on ISIC 2016 skin lesion dataset.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

# =============================================================================
# FULL U-NET ARCHITECTURE
# =============================================================================

class FullUNet(nn.Module):
    """
    Complete U-Net architecture for skin lesion segmentation.
    
    Features:
    - Encoder-decoder structure with skip connections
    - Batch normalization for stable training
    - 4 encoder blocks + bottleneck + 4 decoder blocks
    
    Input: RGB image [batch, 3, H, W]
    Output: Segmentation mask [batch, 1, H, W]
    """
    
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(FullUNet, self).__init__()
        
        features = init_features
        
        # ===== Encoder (Contracting Path) =====
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ===== Bottleneck =====
        self.bottleneck = self._block(features * 8, features * 16)
        
        # ===== Decoder (Expansive Path) =====
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features)
        
        # ===== Final Output =====
        self.conv_final = nn.Conv2d(features, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with skip connections
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
        """Convolutional block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )


# =============================================================================
# DATASET
# =============================================================================

class SkinDataset(Dataset):
    """Dataset for skin lesion segmentation."""
    
    def __init__(self, images_dir, masks_dir, img_size=(256, 256)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
        self.img_size = img_size
        
        self.img_transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor()
        ])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.files[idx])
        mask_name = self.files[idx].replace(".jpg", "_segmentation.png")
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        
        return img, mask


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for imgs, masks in dataloader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_iou = 0
    total_dice = 0
    
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            # Calculate IoU and Dice
            preds = torch.sigmoid(outputs) > 0.5
            intersection = (preds * masks).sum()
            union = preds.sum() + masks.sum() - intersection
            
            iou = intersection / (union + 1e-8)
            dice = 2 * intersection / (preds.sum() + masks.sum() + 1e-8)
            
            total_iou += iou.item()
            total_dice += dice.item()
    
    n = len(dataloader)
    return total_loss / n, total_iou / n, total_dice / n


def train_skin_full():
    """Main training function."""
    print("=" * 70)
    print("FULL U-NET TRAINING - SKIN LESION SEGMENTATION")
    print("=" * 70)
    
    # Paths
    train_images = r"D:\project 2\data\ISIC2016_Task1\train_images"
    train_masks = r"D:\project 2\data\ISIC2016_Task1\train_masks"
    val_images = r"D:\project 2\data\ISIC2016_Task1\test_images"
    val_masks = r"D:\project 2\data\ISIC2016_Task1\test_masks"
    save_path = r"D:\project 2\checkpoints\skin_model_full.pth"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Device: {device}")
    
    # Datasets
    print("\n📦 Loading datasets...")
    train_dataset = SkinDataset(train_images, train_masks)
    val_dataset = SkinDataset(val_images, val_masks)
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("❌ No training data found!")
        return
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Model
    print("\n🏗️ Building Full U-Net model...")
    model = FullUNet(in_channels=3, out_channels=1, init_features=32).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: {total_params * 4 / (1024*1024):.2f} MB")
    
    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training
    print("\n🚀 Starting training...")
    print("-" * 70)
    
    epochs = 20
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_iou, val_dice = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"IoU: {val_iou:.4f} | "
              f"Dice: {val_dice:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'val_dice': val_dice,
            }, save_path)
            print(f"   💾 Best model saved! (Val Loss: {val_loss:.4f})")
    
    print("-" * 70)
    print("\n✅ Training complete!")
    print(f"📁 Model saved to: {save_path}")
    print(f"🏆 Best Val Loss: {best_val_loss:.4f}")
    
    return model


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    train_skin_full()
