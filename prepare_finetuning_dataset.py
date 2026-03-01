"""
Prepare Fine-Tuning Dataset for Medical Chatbot
Converts the processed chatbot dataset into formats ready for various training frameworks
"""

import json
import random
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(r"D:\project 2")
PROCESSED_DIR = BASE_DIR / "data" / "drugs-datasets-processed"
CHATBOT_DATASET = PROCESSED_DIR / "chatbot_training_dataset.json"
FINETUNING_DIR = PROCESSED_DIR / "finetuning_data"

# Create output directory
FINETUNING_DIR.mkdir(parents=True, exist_ok=True)


def load_chatbot_dataset():
    """Load the processed chatbot dataset"""
    with open(CHATBOT_DATASET, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_huggingface_format(qa_pairs):
    """
    Create dataset in Hugging Face format
    Format: {"text": "### Instruction: ... ### Input: ... ### Response: ..."}
    """
    hf_data = []
    
    for qa in qa_pairs:
        instruction = qa.get("instruction", "")
        input_text = qa.get("input", "")
        output = qa.get("output", "")
        
        # Alpaca/LLaMA format
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        
        hf_data.append({"text": text})
    
    return hf_data


def create_openai_format(qa_pairs):
    """
    Create dataset in OpenAI fine-tuning format
    Format: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    """
    openai_data = []
    
    system_message = "You are a medical AI assistant specialized in drug information. Provide accurate, helpful information about medications while emphasizing that users should consult healthcare professionals for medical advice. Always include appropriate disclaimers."
    
    for qa in qa_pairs:
        input_text = qa.get("input", "")
        output = qa.get("output", "")
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output}
        ]
        
        openai_data.append({"messages": messages})
    
    return openai_data


def create_conversational_format(qa_pairs):
    """
    Create conversational format for multi-turn training
    Format: {"conversations": [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}]}
    """
    conv_data = []
    
    for qa in qa_pairs:
        input_text = qa.get("input", "")
        output = qa.get("output", "")
        
        conversations = [
            {"from": "human", "value": input_text},
            {"from": "gpt", "value": output}
        ]
        
        conv_data.append({"conversations": conversations})
    
    return conv_data


def create_instruction_format(qa_pairs):
    """
    Create instruction-tuning format with separate fields
    Format: {"instruction": ..., "input": ..., "output": ...}
    """
    inst_data = []
    
    for qa in qa_pairs:
        inst_data.append({
            "instruction": qa.get("instruction", ""),
            "input": qa.get("input", ""),
            "output": qa.get("output", "")
        })
    
    return inst_data


def split_dataset(data, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05):
    """Split dataset into train, validation, and test sets"""
    random.seed(42)
    random.shuffle(data)
    
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    return {
        "train": data[:train_end],
        "validation": data[train_end:val_end],
        "test": data[val_end:]
    }


def save_jsonl(data, filepath):
    """Save data in JSONL format (one JSON per line)"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def save_json(data, filepath):
    """Save data in JSON format"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    print("=" * 60)
    print("Preparing Fine-Tuning Dataset")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/6] Loading chatbot dataset...")
    dataset = load_chatbot_dataset()
    qa_pairs = dataset.get("qa_pairs", [])
    print(f"Loaded {len(qa_pairs)} QA pairs")
    
    # Create different formats
    print("\n[2/6] Creating HuggingFace format...")
    hf_data = create_huggingface_format(qa_pairs)
    
    print("[3/6] Creating OpenAI format...")
    openai_data = create_openai_format(qa_pairs)
    
    print("[4/6] Creating conversational format...")
    conv_data = create_conversational_format(qa_pairs)
    
    print("[5/6] Creating instruction format...")
    inst_data = create_instruction_format(qa_pairs)
    
    # Split and save datasets
    print("\n[6/6] Splitting and saving datasets...")
    
    formats = {
        "huggingface": hf_data,
        "openai": openai_data,
        "conversational": conv_data,
        "instruction": inst_data
    }
    
    statistics = {
        "created": datetime.now().isoformat(),
        "total_qa_pairs": len(qa_pairs),
        "formats": {}
    }
    
    for format_name, data in formats.items():
        splits = split_dataset(data)
        format_dir = FINETUNING_DIR / format_name
        format_dir.mkdir(parents=True, exist_ok=True)
        
        # Save splits as JSONL
        save_jsonl(splits["train"], format_dir / "train.jsonl")
        save_jsonl(splits["validation"], format_dir / "validation.jsonl")
        save_jsonl(splits["test"], format_dir / "test.jsonl")
        
        # Also save as JSON for convenience
        save_json(splits, format_dir / "all_splits.json")
        
        statistics["formats"][format_name] = {
            "train": len(splits["train"]),
            "validation": len(splits["validation"]),
            "test": len(splits["test"])
        }
        
        print(f"  - {format_name}: train={len(splits['train'])}, val={len(splits['validation'])}, test={len(splits['test'])}")
    
    # Save statistics
    save_json(statistics, FINETUNING_DIR / "statistics.json")
    
    # Create README
    readme_content = f"""# Fine-Tuning Dataset for Medical Chatbot

Generated: {datetime.now().isoformat()}

## Overview
This dataset contains QA pairs for training a medical AI assistant specialized in drug information.

## Statistics
- Total QA Pairs: {len(qa_pairs)}
- Train/Val/Test Split: 90%/5%/5%

## Available Formats

### 1. HuggingFace Format (`huggingface/`)
```json
{{"text": "### Instruction:\\n...\\n### Input:\\n...\\n### Response:\\n..."}}
```
Compatible with: LLaMA, Alpaca, and similar models

### 2. OpenAI Format (`openai/`)
```json
{{"messages": [{{"role": "system", "content": "..."}}, {{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}
```
Compatible with: OpenAI fine-tuning API

### 3. Conversational Format (`conversational/`)
```json
{{"conversations": [{{"from": "human", "value": "..."}}, {{"from": "gpt", "value": "..."}}]}}
```
Compatible with: Vicuna, FastChat

### 4. Instruction Format (`instruction/`)
```json
{{"instruction": "...", "input": "...", "output": "..."}}
```
Compatible with: Alpaca, instruction-tuned models

## QA Categories
- `active_ingredient`: Questions about drug composition
- `indication`: Questions about drug uses
- `general`: General drug information
- `classification`: ATC classification questions
- `safety`: Safety and disclaimer QAs

## Usage Example

```python
from datasets import load_dataset

# Load HuggingFace format
dataset = load_dataset('json', data_files='huggingface/train.jsonl')
```

## Important Notes
- This dataset is for educational and research purposes
- Always include medical disclaimers in production use
- Data is in Spanish (from Latin American drug database)
- Consult healthcare professionals for medical advice
"""
    
    with open(FINETUNING_DIR / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("\n" + "=" * 60)
    print("Fine-Tuning Dataset Preparation Complete!")
    print(f"Output directory: {FINETUNING_DIR}")
    print("=" * 60)
    
    return statistics


if __name__ == "__main__":
    main()
