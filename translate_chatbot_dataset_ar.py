"""
Arabic Translation Script for Medical Chatbot Dataset
======================================================
Automatically translates the English medical QA dataset to Arabic
using Helsinki-NLP/opus-mt-en-ar translation model.

Features:
- Automatic library installation
- Model caching for offline use
- Batch translation for efficiency
- Bilingual output format
- Progress tracking
"""

import os
import sys
import json
import re
import time
from typing import List, Dict, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "input_path": r"D:\project 2\data\chatbot\combined_medical_qa.json",
    "output_path": r"D:\project 2\data\chatbot\combined_medical_qa_ar.json",
    "model_name": "Helsinki-NLP/opus-mt-en-ar",
    "cache_dir": r"D:\project 2\models_pretrained\translation",
    "batch_size": 8,
    "max_length": 512,
    "device": "cpu",  # Use "cuda" if GPU available
}

# =============================================================================
# LIBRARY CHECK AND INSTALLATION
# =============================================================================

def check_and_install_libraries():
    """Check for required libraries and install if missing."""
    print("\n" + "=" * 50)
    print("CHECKING REQUIRED LIBRARIES")
    print("=" * 50)
    
    required_libs = {
        "torch": "torch",
        "transformers": "transformers",
        "sentencepiece": "sentencepiece",
    }
    
    missing_libs = []
    
    for lib_name, pip_name in required_libs.items():
        try:
            __import__(lib_name)
            print(f"   ✓ {lib_name} installed")
        except ImportError:
            print(f"   ✗ {lib_name} not found")
            missing_libs.append(pip_name)
    
    if missing_libs:
        print(f"\n📦 Installing missing libraries: {', '.join(missing_libs)}")
        import subprocess
        for lib in missing_libs:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib, "-q"])
                print(f"   ✓ {lib} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"   ✗ Failed to install {lib}: {e}")
                return False
    
    return True


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_translation_model(model_name: str, cache_dir: str, device: str = "cpu"):
    """Load the translation model and tokenizer."""
    print("\n" + "=" * 50)
    print("LOADING TRANSLATION MODEL")
    print("=" * 50)
    
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"\n📥 Downloading/loading model: {model_name}")
    print(f"   Cache directory: {cache_dir}")
    
    try:
        # Load tokenizer
        print("\n   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            legacy=False
        )
        print("   ✓ Tokenizer loaded")
        
        # Load model
        print("   Loading model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        model.to(device)
        model.eval()
        print(f"   ✓ Model loaded on {device}")
        
        return tokenizer, model
        
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        raise


# =============================================================================
# TRANSLATION FUNCTIONS
# =============================================================================

def translate_text(
    text: str, 
    tokenizer, 
    model, 
    device: str = "cpu",
    max_length: int = 512
) -> str:
    """Translate a single text from English to Arabic."""
    if not text or not text.strip():
        return ""
    
    try:
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )
        
        # Decode
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated
        
    except Exception as e:
        print(f"   ⚠️ Translation error: {e}")
        return text  # Return original on error


def translate_batch(
    texts: List[str],
    tokenizer,
    model,
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 512
) -> List[str]:
    """Translate a batch of texts efficiently."""
    translations = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            # Tokenize batch
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate translations
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode batch
            batch_translations = [
                tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            translations.extend(batch_translations)
            
        except Exception as e:
            print(f"   ⚠️ Batch translation error: {e}")
            # Fall back to individual translation
            for text in batch:
                translations.append(translate_text(text, tokenizer, model, device, max_length))
    
    return translations


# =============================================================================
# TEXT CLEANING
# =============================================================================

def clean_text(text: str) -> str:
    """Clean text for translation."""
    if not text:
        return ""
    
    # Fix encoding issues
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove problematic characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    return text.strip()


# =============================================================================
# MAIN TRANSLATION PROCESS
# =============================================================================

def translate_dataset(
    input_path: str,
    output_path: str,
    tokenizer,
    model,
    config: Dict
):
    """Translate the entire dataset."""
    print("\n" + "=" * 50)
    print("TRANSLATING DATASET")
    print("=" * 50)
    
    # Load dataset
    print(f"\n📥 Loading dataset from: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"   ✗ Dataset not found: {input_path}")
        return None
    
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    total_entries = len(dataset)
    print(f"   ✓ Loaded {total_entries:,} entries")
    
    # Process in batches
    translated_dataset = []
    batch_size = config['batch_size']
    device = config['device']
    max_length = config['max_length']
    
    # Extract all questions and answers for batch processing
    questions = [clean_text(entry.get('question', '')) for entry in dataset]
    answers = [clean_text(entry.get('answer', '')) for entry in dataset]
    
    print(f"\n🔄 Translating {total_entries:,} entries...")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}")
    
    start_time = time.time()
    
    # Translate all questions
    print("\n   Translating questions...")
    translated_questions = translate_batch(
        questions, tokenizer, model, device, batch_size, max_length
    )
    
    # Translate all answers
    print("   Translating answers...")
    translated_answers = translate_batch(
        answers, tokenizer, model, device, batch_size, max_length
    )
    
    # Build bilingual dataset
    print("\n   Building bilingual dataset...")
    for i, entry in enumerate(dataset):
        translated_entry = {
            "question_en": questions[i],
            "answer_en": answers[i],
            "question_ar": translated_questions[i],
            "answer_ar": translated_answers[i],
        }
        translated_dataset.append(translated_entry)
        
        # Progress update
        if (i + 1) % 1000 == 0 or (i + 1) == total_entries:
            elapsed = time.time() - start_time
            entries_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (total_entries - i - 1) / entries_per_sec if entries_per_sec > 0 else 0
            print(f"   Progress: {i + 1:,}/{total_entries:,} ({(i+1)/total_entries*100:.1f}%) - "
                  f"ETA: {remaining:.0f}s")
    
    elapsed_time = time.time() - start_time
    print(f"\n   ✓ Translation completed in {elapsed_time:.1f} seconds")
    
    return translated_dataset


# =============================================================================
# SAVE AND VERIFY
# =============================================================================

def save_dataset(dataset: List[Dict], output_path: str):
    """Save the translated dataset."""
    print("\n" + "=" * 50)
    print("SAVING TRANSLATED DATASET")
    print("=" * 50)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n   ✓ Dataset saved to: {output_path}")
    print(f"   ✓ File size: {file_size:.2f} MB")


def print_summary_and_samples(dataset: List[Dict]):
    """Print summary and sample translations."""
    print("\n" + "=" * 70)
    print("📊 DATASET SUMMARY")
    print("=" * 70)
    
    print(f"\n📈 Total entries: {len(dataset):,}")
    
    # Print sample entries
    print("\n📝 Sample translations (5 random entries):")
    print("-" * 70)
    
    import random
    samples = random.sample(dataset, min(5, len(dataset)))
    
    for i, entry in enumerate(samples, 1):
        print(f"\n{i}.")
        print(f"   EN Q: {entry['question_en'][:100]}...")
        print(f"   AR Q: {entry['question_ar'][:100]}...")
        print(f"   EN A: {entry['answer_en'][:100]}...")
        print(f"   AR A: {entry['answer_ar'][:100]}...")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function to run the translation process."""
    print("=" * 70)
    print("🏥 MEDICAL CHATBOT DATASET - ARABIC TRANSLATION")
    print("=" * 70)
    
    # Step 1: Check and install libraries
    if not check_and_install_libraries():
        print("\n✗ Failed to install required libraries")
        return
    
    # Import torch after checking installation
    global torch
    import torch
    
    # Check for GPU
    if torch.cuda.is_available():
        CONFIG['device'] = 'cuda'
        print(f"\n🎮 GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("\n💻 Using CPU for translation (slower but works)")
    
    # Step 2: Load model
    try:
        tokenizer, model = load_translation_model(
            CONFIG['model_name'],
            CONFIG['cache_dir'],
            CONFIG['device']
        )
    except Exception as e:
        print(f"\n✗ Failed to load translation model: {e}")
        return
    
    # Step 3: Translate dataset
    translated_dataset = translate_dataset(
        CONFIG['input_path'],
        CONFIG['output_path'],
        tokenizer,
        model,
        CONFIG
    )
    
    if translated_dataset is None:
        print("\n✗ Translation failed")
        return
    
    # Step 4: Save dataset
    save_dataset(translated_dataset, CONFIG['output_path'])
    
    # Step 5: Print summary
    print_summary_and_samples(translated_dataset)
    
    print("\n" + "=" * 70)
    print("✅ TRANSLATION COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Output file: {CONFIG['output_path']}")


if __name__ == "__main__":
    main()
