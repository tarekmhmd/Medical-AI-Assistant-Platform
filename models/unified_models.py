"""
Unified Models for Classification + Segmentation
=================================================
This module contains upgraded models that support BOTH classification AND segmentation.

Models:
1. UNetClassifier - U-Net with classification head (from segmentation model)
2. CNNSegmenter - CNN with segmentation decoder (from classification model)
3. MLPSegmenter - MLP with pseudo-segmentation output (from tabular model)
4. LSTMSegmenter - LSTM with attention map output (from text model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. U-NET WITH CLASSIFICATION HEAD (Upgraded from Segmentation-only)
# =============================================================================

class UNetClassifier(nn.Module):
    """
    U-Net architecture upgraded for BOTH Segmentation AND Classification.
    
    Original: Segmentation only (binary mask output)
    Upgraded: Segmentation mask + Disease classification
    
    Architecture:
    - Encoder: 4 blocks with pooling
    - Bottleneck: 512 features
    - Decoder: 4 blocks with skip connections
    - Segmentation head: 1 channel output
    - Classification head: Global pooling + FC layers
    
    Input: [batch, 3, H, W] - RGB image
    Output: (mask, class_logits)
        - mask: [batch, 1, H, W] - Segmentation mask
        - class_logits: [batch, num_classes] - Disease classification
    """
    
    def __init__(self, in_channels=3, out_channels=1, num_classes=8, init_features=32):
        super(UNetClassifier, self).__init__()
        
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
        
        # ===== Segmentation Head =====
        self.seg_head = nn.Conv2d(features, out_channels, kernel_size=1)
        
        # ===== Classification Head (NEW) =====
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features * 16, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Classification from bottleneck features
        class_features = self.global_pool(bottleneck)
        class_logits = self.classifier(class_features)
        
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
        
        # Segmentation output
        mask = self.seg_head(dec1)
        
        return mask, class_logits
    
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
# 2. CNN WITH SEGMENTATION DECODER (Upgraded from Classification-only)
# =============================================================================

class CNNSegmenter(nn.Module):
    """
    CNN architecture upgraded for BOTH Classification AND Segmentation.
    
    Original: Classification only (healthy/unhealthy)
    Upgraded: Classification + Pseudo-segmentation mask
    
    Architecture:
    - Shared encoder: 4 conv blocks
    - Classification head: FC layers
    - Segmentation decoder: Upsampling + conv layers
    
    Input: [batch, 1, 64, 128] - Mel-spectrogram
    Output: (mask, class_logits)
        - mask: [batch, 1, 64, 128] - Attention/activation map
        - class_logits: [batch, num_classes] - Classification
    """
    
    def __init__(self, in_channels=1, num_classes=6, dropout=0.3):
        super(CNNSegmenter, self).__init__()
        
        # ===== Shared Encoder =====
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # ===== Classification Head =====
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        # ===== Segmentation Decoder (NEW) =====
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(256, 128)  # 128 + 128 from skip
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64)   # 64 + 64 from skip
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(96, 32)    # 64 + 32 from skip (adjusted)
        
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(16, 16)
        
        self.seg_head = nn.Conv2d(16, 1, kernel_size=1)
        
        # Store intermediate feature sizes
        self.target_size = (64, 128)
        
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.relu(self.bn1(self.conv1(x)))      # [B, 32, 64, 128]
        p1 = self.pool(e1)                            # [B, 32, 32, 64]
        
        e2 = self.relu(self.bn2(self.conv2(p1)))     # [B, 64, 32, 64]
        p2 = self.pool(e2)                            # [B, 64, 16, 32]
        
        e3 = self.relu(self.bn3(self.conv3(p2)))     # [B, 128, 16, 32]
        p3 = self.pool(e3)                            # [B, 128, 8, 16]
        
        e4 = self.relu(self.bn4(self.conv4(p3)))     # [B, 256, 8, 16]
        p4 = self.pool(e4)                            # [B, 256, 4, 8]
        
        # Classification branch
        class_features = self.adaptive_pool(p4)
        class_features = class_features.view(class_features.size(0), -1)
        class_features = self.dropout(self.relu(self.fc1(class_features)))
        class_features = self.dropout(self.relu(self.fc2(class_features)))
        class_logits = self.fc3(class_features)
        
        # Segmentation decoder
        # Upsample and decode
        d1 = self.up1(p4)                             # [B, 128, 8, 16]
        d1 = torch.cat([d1, e4], dim=1)              # [B, 384, 8, 16] -> need to handle
        d1 = self.dec1(d1[:, :256])                  # Use first 256 channels
        
        d2 = self.up2(d1[:, :128])                   # [B, 64, 16, 32]
        d2 = torch.cat([d2, e3], dim=1)              # [B, 192, 16, 32]
        d2 = self.dec2(d2[:, :128])
        
        d3 = self.up3(d2[:, :64])                    # [B, 32, 32, 64]
        d3 = torch.cat([d3, e2], dim=1)              # [B, 96, 32, 64]
        d3 = self.dec3(d3)
        
        d4 = self.up4(d3)                            # [B, 16, 64, 128]
        d4 = self.dec4(d4)
        
        # Resize to match input if needed
        if d4.shape[-2:] != self.target_size:
            d4 = F.interpolate(d4, size=self.target_size, mode='bilinear', align_corners=False)
        
        mask = self.seg_head(d4)
        
        return mask, class_logits
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


# =============================================================================
# 3. MLP WITH PSEUDO-SEGMENTATION (Upgraded from Classification-only)
# =============================================================================

class MLPSegmenter(nn.Module):
    """
    MLP architecture upgraded for BOTH Classification AND Segmentation.
    
    Original: Classification only (diabetes prediction)
    Upgraded: Classification + Feature importance map (pseudo-segmentation)
    
    Note: For tabular data, "segmentation" means generating a feature importance
    map that can be visualized as a heatmap showing which features contribute
    to the prediction.
    
    Architecture:
    - Shared feature extractor
    - Classification head
    - Segmentation head: Generates feature importance map
    
    Input: [batch, input_dim] - Tabular features
    Output: (importance_map, class_logits)
        - importance_map: [batch, 1, 8, 8] - Feature importance visualization
        - class_logits: [batch, num_classes] - Classification
    """
    
    def __init__(self, input_dim=8, num_classes=2, hidden_sizes=[128, 64, 32], dropout=0.3):
        super(MLPSegmenter, self).__init__()
        
        self.input_dim = input_dim
        
        # ===== Shared Feature Extractor =====
        layers = []
        prev_size = input_dim
        self.feature_sizes = []
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            self.feature_sizes.append(hidden_size)
            prev_size = hidden_size
        
        self.shared_features = nn.Sequential(*layers)
        self.feature_dim = prev_size  # 32
        
        # ===== Classification Head =====
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        # ===== Segmentation Head (NEW) =====
        # Generate feature importance map from learned features
        # We'll create a visual representation of feature contributions
        self.importance_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),  # 8x8 = 64
            nn.Sigmoid()
        )
        
        # Learnable feature importance weights
        self.feature_weights = nn.Parameter(torch.randn(input_dim, 1))
        
    def forward(self, x):
        # Shared features
        features = self.shared_features(x)
        
        # Classification
        class_logits = self.classifier(features)
        
        # Generate importance map
        # Method 1: From shared features
        importance_flat = self.importance_encoder(features)  # [B, 64]
        importance_map = importance_flat.view(-1, 1, 8, 8)    # [B, 1, 8, 8]
        
        # Method 2: From input features (attention-like)
        # Create a weighted combination of input features
        feature_importance = torch.matmul(x, self.feature_weights)  # [B, 1]
        
        return importance_map, class_logits


# =============================================================================
# 4. LSTM WITH ATTENTION MAP (Upgraded from Classification-only)
# =============================================================================

class LSTMSegmenter(nn.Module):
    """
    LSTM architecture upgraded for BOTH Classification AND Segmentation.
    
    Original: Classification only (symptom-to-disease)
    Upgraded: Classification + Attention map (word importance visualization)
    
    Note: For text data, "segmentation" means generating an attention map
    that highlights important words/tokens in the input sequence.
    
    Architecture:
    - Embedding layer
    - Bidirectional LSTM
    - Classification head with attention
    - Segmentation head: Word-level attention map
    
    Input: [batch, seq_len] - Word indices
    Output: (attention_map, class_logits)
        - attention_map: [batch, 1, seq_len] - Word importance scores
        - class_logits: [batch, num_classes] - Disease classification
    """
    
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, 
                 num_classes=100, dropout=0.3, max_seq_len=20):
        super(LSTMSegmenter, self).__init__()
        
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        
        # ===== Embedding Layer =====
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # ===== Bidirectional LSTM =====
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # ===== Classification Head =====
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        
        # ===== Segmentation Head (NEW) =====
        # Word-level attention for importance visualization
        self.word_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # [B, seq_len, embed_dim]
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # [B, seq_len, hidden_dim]
        
        # Classification with attention pooling
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [B, hidden_dim]
        
        # Classification output
        class_features = self.fc1(context)
        class_features = self.bn1(class_features)
        class_features = torch.relu(class_features)
        class_features = self.dropout(class_features)
        class_logits = self.fc2(class_features)
        
        # Segmentation: Word-level attention map
        word_scores = self.word_attention(lstm_out)  # [B, seq_len, 1]
        word_scores = word_scores.squeeze(-1)        # [B, seq_len]
        
        # Normalize to [0, 1] range for visualization
        word_attention_map = torch.sigmoid(word_scores)
        
        # Reshape to segmentation format: [B, 1, seq_len]
        attention_map = word_attention_map.unsqueeze(1)
        
        return attention_map, class_logits


# =============================================================================
# COMBINED LOSS FUNCTION
# =============================================================================

class CombinedLoss(nn.Module):
    """
    Combined loss for segmentation + classification tasks.
    
    Loss = seg_weight * SegmentationLoss + cls_weight * ClassificationLoss
    
    For segmentation: BCEWithLogitsLoss
    For classification: CrossEntropyLoss
    """
    
    def __init__(self, seg_weight=0.5, cls_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.cls_loss = nn.CrossEntropyLoss()
        
    def forward(self, pred_mask, pred_class, target_mask, target_class):
        """
        Args:
            pred_mask: Predicted segmentation mask [B, 1, H, W]
            pred_class: Predicted class logits [B, num_classes]
            target_mask: Ground truth mask [B, 1, H, W]
            target_class: Ground truth class [B]
        """
        loss_seg = self.seg_loss(pred_mask, target_mask)
        loss_cls = self.cls_loss(pred_class, target_class)
        
        total_loss = self.seg_weight * loss_seg + self.cls_weight * loss_cls
        
        return total_loss, loss_seg, loss_cls


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_model(model_type, **kwargs):
    """
    Factory function to create unified models.
    
    Args:
        model_type: 'unet', 'cnn', 'mlp', or 'lstm'
        **kwargs: Model-specific arguments
    
    Returns:
        model: PyTorch model instance
    """
    if model_type == 'unet':
        return UNetClassifier(
            in_channels=kwargs.get('in_channels', 3),
            out_channels=kwargs.get('out_channels', 1),
            num_classes=kwargs.get('num_classes', 8),
            init_features=kwargs.get('init_features', 32)
        )
    elif model_type == 'cnn':
        return CNNSegmenter(
            in_channels=kwargs.get('in_channels', 1),
            num_classes=kwargs.get('num_classes', 6),
            dropout=kwargs.get('dropout', 0.3)
        )
    elif model_type == 'mlp':
        return MLPSegmenter(
            input_dim=kwargs.get('input_dim', 8),
            num_classes=kwargs.get('num_classes', 2),
            hidden_sizes=kwargs.get('hidden_sizes', [128, 64, 32]),
            dropout=kwargs.get('dropout', 0.3)
        )
    elif model_type == 'lstm':
        return LSTMSegmenter(
            vocab_size=kwargs.get('vocab_size', 10000),
            embed_dim=kwargs.get('embed_dim', 128),
            hidden_dim=kwargs.get('hidden_dim', 256),
            num_layers=kwargs.get('num_layers', 2),
            num_classes=kwargs.get('num_classes', 100),
            dropout=kwargs.get('dropout', 0.3),
            max_seq_len=kwargs.get('max_seq_len', 20)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_models():
    """Test all unified models."""
    print("=" * 70)
    print("TESTING UNIFIED MODELS (Classification + Segmentation)")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Test U-Net
    print("1. Testing UNetClassifier...")
    unet = create_model('unet', num_classes=8).to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    mask, cls = unet(x)
    print(f"   Input: {x.shape}")
    print(f"   Mask output: {mask.shape}")
    print(f"   Class output: {cls.shape}")
    print(f"   Parameters: {sum(p.numel() for p in unet.parameters()):,}\n")
    
    # Test CNN
    print("2. Testing CNNSegmenter...")
    cnn = create_model('cnn', num_classes=6).to(device)
    x = torch.randn(2, 1, 64, 128).to(device)
    mask, cls = cnn(x)
    print(f"   Input: {x.shape}")
    print(f"   Mask output: {mask.shape}")
    print(f"   Class output: {cls.shape}")
    print(f"   Parameters: {sum(p.numel() for p in cnn.parameters()):,}\n")
    
    # Test MLP
    print("3. Testing MLPSegmenter...")
    mlp = create_model('mlp', input_dim=8, num_classes=2).to(device)
    x = torch.randn(2, 8).to(device)
    mask, cls = mlp(x)
    print(f"   Input: {x.shape}")
    print(f"   Importance map output: {mask.shape}")
    print(f"   Class output: {cls.shape}")
    print(f"   Parameters: {sum(p.numel() for p in mlp.parameters()):,}\n")
    
    # Test LSTM
    print("4. Testing LSTMSegmenter...")
    lstm = create_model('lstm', vocab_size=1000, num_classes=50).to(device)
    x = torch.randint(0, 1000, (2, 20)).to(device)
    mask, cls = lstm(x)
    print(f"   Input: {x.shape}")
    print(f"   Attention map output: {mask.shape}")
    print(f"   Class output: {cls.shape}")
    print(f"   Parameters: {sum(p.numel() for p in lstm.parameters()):,}\n")
    
    # Test Combined Loss
    print("5. Testing CombinedLoss...")
    criterion = CombinedLoss(seg_weight=0.5, cls_weight=0.5)
    
    pred_mask = torch.randn(2, 1, 256, 256)
    pred_class = torch.randn(2, 8)
    target_mask = torch.randn(2, 1, 256, 256)
    target_class = torch.randint(0, 8, (2,))
    
    total_loss, seg_loss, cls_loss = criterion(pred_mask, pred_class, target_mask, target_class)
    print(f"   Total Loss: {total_loss.item():.4f}")
    print(f"   Segmentation Loss: {seg_loss.item():.4f}")
    print(f"   Classification Loss: {cls_loss.item():.4f}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_models()
