"""
GLM-5 Compatible Training Wrapper
==================================
Provides a CLI-like interface for training classification and segmentation models.
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'models'))

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

# Import existing training infrastructure
from train_unified_complete import (
    SkinDatasetReal, SoundDatasetReal, LabDatasetReal, ChatbotDatasetReal,
    train_skin_model, train_sound_model, train_lab_model, train_chatbot_model,
    CONFIG, train_all
)

def parse_args():
    parser = argparse.ArgumentParser(description='GLM-5 Compatible Training')
    parser.add_argument('--project_dir', type=str, default='.', help='Project directory')
    parser.add_argument('--mode', type=str, default='resume', choices=['train', 'resume', 'finetune'])
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints/')
    parser.add_argument('--models', type=str, default='classification,segmentation', help='Models to train')
    parser.add_argument('--dataset_dir', type=str, default='data/', help='Dataset directory')
    parser.add_argument('--save_every', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--retry_on_timeout', type=bool, default=True, help='Retry on timeout')
    parser.add_argument('--report_file', type=str, default='training_report.json', help='Report file path')
    return parser.parse_args()

def update_config(args):
    """Update configuration with CLI arguments."""
    # Update batch sizes and epochs
    for model in CONFIG:
        CONFIG[model]['batch_size'] = args.batch_size
        CONFIG[model]['epochs'] = args.max_epochs
    
    # Update paths
    CONFIG['skin']['train_images'] = os.path.join(args.dataset_dir, 'ISIC2016_Task1/train_images')
    CONFIG['skin']['train_masks'] = os.path.join(args.dataset_dir, 'ISIC2016_Task1/train_masks')
    CONFIG['skin']['test_images'] = os.path.join(args.dataset_dir, 'ISIC2016_Task1/test_images')
    CONFIG['skin']['test_masks'] = os.path.join(args.dataset_dir, 'ISIC2016_Task1/test_masks')
    CONFIG['skin']['checkpoint'] = os.path.join(args.checkpoints_dir, 'skin_model_cls_seg_trained.pth')
    
    CONFIG['sound']['data_path'] = os.path.join(args.dataset_dir, 'sound')
    CONFIG['sound']['checkpoint'] = os.path.join(args.checkpoints_dir, 'sound_model_cls_seg_trained.pth')
    
    CONFIG['lab']['train_data'] = os.path.join(args.dataset_dir, 'lab/train/diabetes_data_train.csv')
    CONFIG['lab']['val_data'] = os.path.join(args.dataset_dir, 'lab/val/diabetes_data_val.csv')
    CONFIG['lab']['checkpoint'] = os.path.join(args.checkpoints_dir, 'lab_model_cls_seg_trained.pth')
    
    CONFIG['chatbot']['data_path'] = os.path.join(args.dataset_dir, 'chatbot/combined_medical_qa.json')
    CONFIG['chatbot']['checkpoint'] = os.path.join(args.checkpoints_dir, 'chatbot_model_cls_seg_trained.pth')
    
    return CONFIG

def check_data_availability(args):
    """Check which datasets are available."""
    available = {}
    
    # Check skin data
    skin_train = os.path.join(args.project_dir, CONFIG['skin']['train_images'])
    available['skin'] = os.path.exists(skin_train)
    
    # Check sound data
    sound_path = os.path.join(args.project_dir, CONFIG['sound']['data_path'])
    available['sound'] = os.path.exists(sound_path)
    
    # Check lab data
    lab_train = os.path.join(args.project_dir, CONFIG['lab']['train_data'])
    available['lab'] = os.path.exists(lab_train)
    
    # Check chatbot data
    chatbot_path = os.path.join(args.project_dir, CONFIG['chatbot']['data_path'])
    available['chatbot'] = os.path.exists(chatbot_path)
    
    return available

def main():
    print("=" * 70)
    print("GLM-5 COMPATIBLE TRAINING PIPELINE")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    args = parse_args()
    
    # Change to project directory
    os.chdir(args.project_dir)
    
    # Update configuration
    update_config(args)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Check data availability
    print("\n" + "-" * 70)
    print("CHECKING DATA AVAILABILITY")
    print("-" * 70)
    
    available = check_data_availability(args)
    for model, is_available in available.items():
        status = "✓" if is_available else "✗"
        print(f"  {status} {model}: {'Available' if is_available else 'Not found'}")
    
    # Parse models to train
    models_to_train = [m.strip() for m in args.models.split(',')]
    print(f"\nModels to train: {models_to_train}")
    
    # Training results
    results = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'project_dir': args.project_dir,
            'mode': args.mode,
            'checkpoints_dir': args.checkpoints_dir,
            'models': args.models,
            'max_epochs': args.max_epochs,
            'batch_size': args.batch_size,
        },
        'device': str(device),
        'training_results': {},
        'errors': [],
        'status': 'completed'
    }
    
    # Create checkpoints directory
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    
    # Train each available model
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    trained_models = []
    
    if available.get('skin') and 'classification' in models_to_train:
        print("\n[1/4] Training Skin Model (Classification + Segmentation)...")
        try:
            checkpoint = CONFIG['skin']['checkpoint']
            if args.mode == 'resume' and os.path.exists(checkpoint):
                print(f"  Resuming from: {checkpoint}")
            
            result = train_skin_model(CONFIG['skin'], device)
            results['training_results']['skin'] = {'status': 'completed'}
            trained_models.append('skin')
            print("  ✓ Skin model training complete")
        except Exception as e:
            results['errors'].append(f"Skin model: {str(e)}")
            print(f"  ✗ Error: {str(e)[:100]}")
    else:
        print("\n[1/4] Skin model: Skipped (data not available or not selected)")
    
    if available.get('sound') and 'classification' in models_to_train:
        print("\n[2/4] Training Sound Model (Classification + Segmentation)...")
        try:
            result = train_sound_model(CONFIG['sound'], device)
            results['training_results']['sound'] = {'status': 'completed'}
            trained_models.append('sound')
            print("  ✓ Sound model training complete")
        except Exception as e:
            results['errors'].append(f"Sound model: {str(e)}")
            print(f"  ✗ Error: {str(e)[:100]}")
    else:
        print("\n[2/4] Sound model: Skipped (data not available or not selected)")
    
    if available.get('lab') and 'classification' in models_to_train:
        print("\n[3/4] Training Lab Model (Classification + Segmentation)...")
        try:
            result = train_lab_model(CONFIG['lab'], device)
            results['training_results']['lab'] = {'status': 'completed'}
            trained_models.append('lab')
            print("  ✓ Lab model training complete")
        except Exception as e:
            results['errors'].append(f"Lab model: {str(e)}")
            print(f"  ✗ Error: {str(e)[:100]}")
    else:
        print("\n[3/4] Lab model: Skipped (data not available or not selected)")
    
    if available.get('chatbot') and 'classification' in models_to_train:
        print("\n[4/4] Training Chatbot Model (Classification)...")
        try:
            result = train_chatbot_model(CONFIG['chatbot'], device)
            results['training_results']['chatbot'] = {'status': 'completed'}
            trained_models.append('chatbot')
            print("  ✓ Chatbot model training complete")
        except Exception as e:
            results['errors'].append(f"Chatbot model: {str(e)}")
            print(f"  ✗ Error: {str(e)[:100]}")
    else:
        print("\n[4/4] Chatbot model: Skipped (data not available or not selected)")
    
    # Summary
    results['summary'] = {
        'models_trained': trained_models,
        'total_models': len(trained_models),
        'errors_count': len(results['errors'])
    }
    
    # Save report
    print("\n" + "=" * 70)
    print("SAVING REPORT")
    print("=" * 70)
    
    report_dir = os.path.dirname(args.report_file)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    
    with open(args.report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"Report saved to: {args.report_file}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Models trained: {len(trained_models)}")
    for model in trained_models:
        print(f"  ✓ {model}")
    
    if results['errors']:
        print(f"\nErrors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"  ✗ {error}")
    
    return results


if __name__ == '__main__':
    main()
