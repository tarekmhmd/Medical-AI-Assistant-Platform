"""
Fast Arabic Translation Script for Medical Chatbot Dataset (Subset Version)
===========================================================================
Translates a subset of the dataset for faster testing/preview.
Uses batch processing and progress tracking.
"""

import os
import sys
import json
import re
import time
from typing import List, Dict

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "input_path": r"D:\project 2\data\chatbot\combined_medical_qa.json",
    "output_path": r"D:\project 2\data\chatbot\combined_medical_qa_ar.json",
    "model_name": "Helsinki-NLP/opus-mt-en-ar",
    "cache_dir": r"D:\project 2\models_pretrained\translation",
    "batch_size": 16,
    "max_length": 512,
    "max_entries": 500,  # Process first 500 entries for testing
    "device": "cpu",
}


def clean_text(text: str) -> str:
    """Clean text for translation."""
    if not text:
        return ""
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return text.strip()


def main():
    print("=" * 70)
    print("🏥 MEDICAL CHATBOT - FAST ARABIC TRANSLATION")
    print("=" * 70)
    
    # Import libraries
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    # Load model
    print("\n📥 Loading translation model...")
    os.makedirs(CONFIG['cache_dir'], exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG['model_name'],
        cache_dir=CONFIG['cache_dir'],
        legacy=False
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        CONFIG['model_name'],
        cache_dir=CONFIG['cache_dir']
    )
    model.to(CONFIG['device'])
    model.eval()
    print("   ✓ Model loaded")
    
    # Load dataset
    print(f"\n📥 Loading dataset from: {CONFIG['input_path']}")
    with open(CONFIG['input_path'], 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    total = min(len(dataset), CONFIG['max_entries'])
    print(f"   Processing {total:,} of {len(dataset):,} entries")
    
    # Translate
    translated_dataset = []
    start_time = time.time()
    
    for i in range(0, total, CONFIG['batch_size']):
        batch_end = min(i + CONFIG['batch_size'], total)
        batch = dataset[i:batch_end]
        
        # Prepare texts
        questions = [clean_text(e.get('question', '')) for e in batch]
        answers = [clean_text(e.get('answer', '')) for e in batch]
        
        # Translate questions
        q_inputs = tokenizer(
            questions,
            return_tensors="pt",
            max_length=CONFIG['max_length'],
            truncation=True,
            padding=True
        )
        q_inputs = {k: v.to(CONFIG['device']) for k, v in q_inputs.items()}
        
        with torch.no_grad():
            q_outputs = model.generate(
                **q_inputs,
                max_length=CONFIG['max_length'],
                num_beams=2,
                early_stopping=True
            )
        
        q_translated = [tokenizer.decode(o, skip_special_tokens=True) for o in q_outputs]
        
        # Translate answers
        a_inputs = tokenizer(
            answers,
            return_tensors="pt",
            max_length=CONFIG['max_length'],
            truncation=True,
            padding=True
        )
        a_inputs = {k: v.to(CONFIG['device']) for k, v in a_inputs.items()}
        
        with torch.no_grad():
            a_outputs = model.generate(
                **a_inputs,
                max_length=CONFIG['max_length'],
                num_beams=2,
                early_stopping=True
            )
        
        a_translated = [tokenizer.decode(o, skip_special_tokens=True) for o in a_outputs]
        
        # Build bilingual entries
        for j, entry in enumerate(batch):
            translated_dataset.append({
                "question_en": questions[j],
                "answer_en": answers[j],
                "question_ar": q_translated[j],
                "answer_ar": a_translated[j],
            })
        
        # Progress
        elapsed = time.time() - start_time
        rate = (batch_end) / elapsed if elapsed > 0 else 0
        remaining = (total - batch_end) / rate if rate > 0 else 0
        print(f"   Progress: {batch_end:,}/{total:,} ({batch_end/total*100:.1f}%) - "
              f"ETA: {remaining:.0f}s")
    
    # Save
    os.makedirs(os.path.dirname(CONFIG['output_path']), exist_ok=True)
    with open(CONFIG['output_path'], 'w', encoding='utf-8') as f:
        json.dump(translated_dataset, f, ensure_ascii=False, indent=2)
    
    file_size = os.path.getsize(CONFIG['output_path']) / (1024 * 1024)
    print(f"\n✓ Dataset saved: {CONFIG['output_path']}")
    print(f"✓ File size: {file_size:.2f} MB")
    
    # Print samples
    print("\n📝 Sample translations:")
    print("-" * 50)
    import random
    for entry in random.sample(translated_dataset, min(3, len(translated_dataset))):
        print(f"\nEN Q: {entry['question_en'][:80]}...")
        print(f"AR Q: {entry['question_ar'][:80]}...")
        print(f"EN A: {entry['answer_en'][:80]}...")
        print(f"AR A: {entry['answer_ar'][:80]}...")
    
    print("\n" + "=" * 70)
    print("✅ TRANSLATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
