"""
Full System Validation Test
===========================
Comprehensive test of all models with random samples.
"""

import sys
sys.path.insert(0, 'models')
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import os
import json
from datetime import datetime

print('=' * 70)
print('FULL PIPELINE TEST - Inference on Random Samples')
print('=' * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print()

results = {'timestamp': datetime.now().isoformat(), 'device': str(device), 'tests': {}}

# ============ TEST 1: SKIN MODEL ============
print('TEST 1: Skin Lesion Segmentation')
print('-' * 50)
try:
    from train_skin_full import FullUNet
    model = FullUNet(in_channels=3, out_channels=1, init_features=32).to(device)
    ckpt = torch.load('checkpoints/skin_model_full.pth', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    test_dir = 'data/ISIC2016_Task1/test_images'
    mask_dir = 'data/ISIC2016_Task1/test_masks'
    files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')][:5]
    
    transform = T.Compose([T.Resize((256,256)), T.ToTensor(), 
                          T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    mask_transform = T.Compose([T.Resize((256,256)), T.ToTensor()])
    
    ious = []
    dices = []
    
    for f in files:
        img = Image.open(os.path.join(test_dir, f)).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
        
        base = os.path.splitext(f)[0]
        mask_path = os.path.join(mask_dir, f'{base}_segmentation.png')
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask_t = mask_transform(mask).to(device)
            
            with torch.no_grad():
                pred = model(img_t)
                pred_bin = (torch.sigmoid(pred) > 0.5).float()
                
                inter = (pred_bin * mask_t).sum()
                union = pred_bin.sum() + mask_t.sum() - inter
                iou = (inter + 1e-6) / (union + 1e-6)
                
                dice = (2 * inter + 1e-6) / (pred_bin.sum() + mask_t.sum() + 1e-6)
                
                ious.append(iou.item())
                dices.append(dice.item())
    
    avg_iou = np.mean(ious) if ious else 0
    avg_dice = np.mean(dices) if dices else 0
    print(f'Samples tested: {len(files)}')
    print(f'Average IoU: {avg_iou:.4f}')
    print(f'Average Dice: {avg_dice:.4f}')
    results['tests']['skin'] = {'status': 'OK', 'iou': avg_iou, 'dice': avg_dice, 'samples': len(files)}
    
except Exception as e:
    print(f'[ERROR] {e}')
    results['tests']['skin'] = {'status': 'ERROR', 'error': str(e)}

print()

# ============ TEST 2: SOUND MODEL ============
print('TEST 2: Respiratory Sound Classification')
print('-' * 50)
try:
    from models.unified_models import CNNSegmenter
    model = CNNSegmenter(in_channels=1, num_classes=6).to(device)
    ckpt = torch.load('checkpoints/sound_model_cls_seg_trained.pth', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    total = 5
    class_names = ['Healthy', 'Asthma', 'Bronchitis', 'Pneumonia', 'COPD', 'Whooping']
    
    for i in range(total):
        x = torch.randn(1, 1, 64, 128).to(device)
        with torch.no_grad():
            mask, cls = model(x)
            pred = cls.argmax(dim=1).item()
        print(f'  Sample {i+1}: Predicted = {class_names[pred]}')
    
    print(f'Total samples tested: {total}')
    results['tests']['sound'] = {'status': 'OK', 'samples': total}
    
except Exception as e:
    print(f'[ERROR] {e}')
    results['tests']['sound'] = {'status': 'ERROR', 'error': str(e)}

print()

# ============ TEST 3: LAB MODEL ============
print('TEST 3: Lab Data Classification (Diabetes)')
print('-' * 50)
try:
    from models.unified_models import MLPSegmenter
    model = MLPSegmenter(input_dim=8, num_classes=2).to(device)
    ckpt = torch.load('checkpoints/lab_model_cls_seg_trained.pth', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    for i in range(5):
        x = torch.rand(1, 8).to(device) * torch.tensor([10, 200, 120, 50, 300, 40, 2, 80]).to(device)
        with torch.no_grad():
            imp, cls = model(x)
            pred = cls.argmax(dim=1).item()
            prob = torch.softmax(cls, dim=1)[0, pred].item()
        
        status = "Positive" if pred == 1 else "Negative"
        print(f'  Sample {i+1}: Diabetes = {status} (confidence: {prob:.2%})')
    
    print(f'Total samples tested: 5')
    results['tests']['lab'] = {'status': 'OK', 'samples': 5}
    
except Exception as e:
    print(f'[ERROR] {e}')
    results['tests']['lab'] = {'status': 'ERROR', 'error': str(e)}

print()

# ============ TEST 4: CHATBOT MODEL ============
print('TEST 4: Medical Chatbot Classification')
print('-' * 50)
try:
    from models.unified_models import LSTMSegmenter
    ckpt = torch.load('checkpoints/chatbot_model_cls_seg_trained.pth', map_location=device, weights_only=False)
    vocab_size = ckpt.get('vocab_size', 5000)
    num_classes = ckpt.get('num_classes', 100)
    model = LSTMSegmenter(vocab_size=vocab_size, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    for i in range(5):
        x = torch.randint(0, vocab_size, (1, 20)).to(device)
        with torch.no_grad():
            attn, cls = model(x)
            pred = cls.argmax(dim=1).item()
        
        print(f'  Sample {i+1}: Predicted class = {pred}')
    
    print(f'Total samples tested: 5')
    results['tests']['chatbot'] = {'status': 'OK', 'samples': 5}
    
except Exception as e:
    print(f'[ERROR] {e}')
    results['tests']['chatbot'] = {'status': 'ERROR', 'error': str(e)}

print()
print('=' * 70)
print('PIPELINE TEST COMPLETE')
print('=' * 70)

# Save results
with open('models_summary/full_system_test_report.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Results saved to: models_summary/full_system_test_report.json')
