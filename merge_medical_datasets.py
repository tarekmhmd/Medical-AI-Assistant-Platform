"""
Merge Medical QA Datasets for Chatbot Fine-Tuning
Combines old MedQuad data with new drugs dataset, removes duplicates, and prepares training data
"""

import json
import random
import hashlib
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Paths
BASE_DIR = Path(r"D:\project 2")
DATA_DIR = BASE_DIR / "data"

# Input datasets
OLD_QA_EN = DATA_DIR / "chatbot" / "combined_medical_qa.json"
OLD_QA_AR = DATA_DIR / "chatbot" / "combined_medical_qa_ar.json"
NEW_DRUGS_QA = DATA_DIR / "drugs-datasets-processed" / "chatbot_training_dataset.json"

# Output directory
OUTPUT_DIR = DATA_DIR / "chatbot_training_combined"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Processing log
processing_log = {
    "start_time": datetime.now().isoformat(),
    "sources": {},
    "statistics": {},
    "warnings": [],
    "errors": []
}


def log_info(message):
    print(f"[INFO] {message}")


def log_warning(message):
    print(f"[WARNING] {message}")
    processing_log["warnings"].append(message)


def log_error(message):
    print(f"[ERROR] {message}")
    processing_log["errors"].append(message)


def normalize_question(question):
    """Normalize question for duplicate detection"""
    if not question:
        return ""
    # Lowercase, remove extra whitespace, remove punctuation
    q = question.lower().strip()
    q = re.sub(r'\s+', ' ', q)
    q = re.sub(r'[^\w\s]', '', q)
    return q


def create_hash(question, answer):
    """Create a hash for duplicate detection"""
    combined = f"{normalize_question(question)}:{answer[:100] if answer else ''}"
    return hashlib.md5(combined.encode('utf-8', errors='ignore')).hexdigest()


def load_old_qa_dataset(filepath, language="en"):
    """Load and standardize old QA dataset"""
    log_info(f"Loading {filepath}...")
    
    qa_pairs = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                if language == "en":
                    question = item.get("question", item.get("question_en", ""))
                    answer = item.get("answer", item.get("answer_en", ""))
                else:  # Arabic
                    question = item.get("question_ar", item.get("question", ""))
                    answer = item.get("answer_ar", item.get("answer", ""))
                
                if question and answer:
                    qa_pairs.append({
                        "question": question.strip(),
                        "answer": answer.strip(),
                        "source": item.get("source", "MedQuad"),
                        "focus_area": item.get("focus_area", "General"),
                        "language": language,
                        "category": categorize_question(question)
                    })
        
        log_info(f"Loaded {len(qa_pairs)} QA pairs from {filepath}")
        return qa_pairs
        
    except Exception as e:
        log_error(f"Error loading {filepath}: {str(e)}")
        return []


def load_drugs_qa_dataset(filepath):
    """Load and standardize drugs QA dataset"""
    log_info(f"Loading {filepath}...")
    
    qa_pairs = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        qa_list = data.get("qa_pairs", [])
        
        for item in qa_list:
            question = item.get("input", item.get("question", ""))
            answer = item.get("output", item.get("answer", ""))
            
            if question and answer:
                qa_pairs.append({
                    "question": question.strip(),
                    "answer": answer.strip(),
                    "source": "DrugsDataset",
                    "focus_area": "Medication",
                    "language": "en",
                    "category": item.get("category", "drug_info")
                })
        
        log_info(f"Loaded {len(qa_pairs)} QA pairs from {filepath}")
        return qa_pairs
        
    except Exception as e:
        log_error(f"Error loading {filepath}: {str(e)}")
        return []


def categorize_question(question):
    """Categorize question by type"""
    q_lower = question.lower()
    
    if any(word in q_lower for word in ["what is", "what are", "define", "definition"]):
        return "definition"
    elif any(word in q_lower for word in ["cause", "causes", "why"]):
        return "causes"
    elif any(word in q_lower for word in ["symptom", "signs", "feel"]):
        return "symptoms"
    elif any(word in q_lower for word in ["treat", "treatment", "cure", "medicine", "medication"]):
        return "treatment"
    elif any(word in q_lower for word in ["prevent", "prevention", "avoid"]):
        return "prevention"
    elif any(word in q_lower for word in ["diagnos", "test", "exam"]):
        return "diagnosis"
    elif any(word in q_lower for word in ["risk", "who get", "who is at"]):
        return "risk_factors"
    elif any(word in q_lower for word in ["ingredient", "active", "composition"]):
        return "active_ingredient"
    elif any(word in q_lower for word in ["side effect", "adverse"]):
        return "side_effects"
    elif any(word in q_lower for word in ["dosage", "dose", "how much"]):
        return "dosage"
    else:
        return "general"


def remove_duplicates(qa_pairs):
    """Remove duplicate QA pairs based on normalized question"""
    log_info("Removing duplicates...")
    
    seen_questions = {}
    unique_pairs = []
    duplicates = 0
    
    for qa in qa_pairs:
        norm_q = normalize_question(qa["question"])
        
        if norm_q not in seen_questions:
            seen_questions[norm_q] = qa
            unique_pairs.append(qa)
        else:
            duplicates += 1
    
    log_info(f"Removed {duplicates} duplicates, {len(unique_pairs)} unique pairs remaining")
    
    return unique_pairs


def validate_qa_pair(qa):
    """Validate QA pair quality"""
    # Check minimum lengths
    if len(qa["question"]) < 10:
        return False, "Question too short"
    if len(qa["answer"]) < 20:
        return False, "Answer too short"
    
    # Check for placeholder text
    placeholders = ["N/A", "TODO", "PLACEHOLDER", "[INSERT"]
    for placeholder in placeholders:
        if qa["answer"].startswith(placeholder):
            return False, f"Contains placeholder: {placeholder}"
    
    return True, "Valid"


def clean_answer(answer):
    """Clean answer text"""
    # Remove excessive whitespace
    answer = re.sub(r'\s+', ' ', answer)
    
    # Remove video references
    answer = re.sub(r'\(Watch the[^)]*\)', '', answer)
    answer = re.sub(r'To enlarge the video[^.]*\.', '', answer)
    answer = re.sub(r'To reduce the video[^.]*\.', '', answer)
    
    # Remove multiple consecutive periods
    answer = re.sub(r'\.{2,}', '.', answer)
    
    return answer.strip()


def merge_datasets():
    """Merge all datasets into one unified dataset"""
    log_info("=" * 60)
    log_info("Starting Dataset Merge Process")
    log_info("=" * 60)
    
    all_qa_pairs = []
    
    # Load old English QA dataset
    if OLD_QA_EN.exists():
        old_en_qa = load_old_qa_dataset(OLD_QA_EN, "en")
        all_qa_pairs.extend(old_en_qa)
        processing_log["sources"]["old_qa_en"] = {
            "path": str(OLD_QA_EN),
            "count": len(old_en_qa)
        }
    else:
        log_warning(f"File not found: {OLD_QA_EN}")
    
    # Load old Arabic QA dataset (only use English portions to avoid duplicates)
    # We'll skip the Arabic file to avoid duplicates with English file
    
    # Load new drugs QA dataset
    if NEW_DRUGS_QA.exists():
        new_drugs_qa = load_drugs_qa_dataset(NEW_DRUGS_QA)
        all_qa_pairs.extend(new_drugs_qa)
        processing_log["sources"]["new_drugs_qa"] = {
            "path": str(NEW_DRUGS_QA),
            "count": len(new_drugs_qa)
        }
    else:
        log_warning(f"File not found: {NEW_DRUGS_QA}")
    
    log_info(f"Total QA pairs before deduplication: {len(all_qa_pairs)}")
    
    # Clean answers
    log_info("Cleaning answer text...")
    for qa in all_qa_pairs:
        qa["answer"] = clean_answer(qa["answer"])
    
    # Remove duplicates
    unique_qa_pairs = remove_duplicates(all_qa_pairs)
    
    # Validate and filter
    log_info("Validating QA pairs...")
    valid_qa_pairs = []
    invalid_count = 0
    validation_issues = defaultdict(int)
    
    for qa in unique_qa_pairs:
        is_valid, reason = validate_qa_pair(qa)
        if is_valid:
            valid_qa_pairs.append(qa)
        else:
            invalid_count += 1
            validation_issues[reason] += 1
    
    log_info(f"Validated: {len(valid_qa_pairs)} valid, {invalid_count} invalid")
    for reason, count in validation_issues.items():
        log_info(f"  - {reason}: {count}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(valid_qa_pairs)
    
    return valid_qa_pairs


def split_dataset(qa_pairs, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05):
    """Split dataset into train, validation, and test sets"""
    log_info("Splitting dataset...")
    
    total = len(qa_pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        "train": qa_pairs[:train_end],
        "validation": qa_pairs[train_end:val_end],
        "test": qa_pairs[val_end:]
    }
    
    for split_name, split_data in splits.items():
        log_info(f"  - {split_name}: {len(split_data)} pairs")
    
    return splits


def create_huggingface_format(qa_pairs):
    """Create HuggingFace/Alpaca format"""
    hf_data = []
    
    for qa in qa_pairs:
        text = f"### Instruction:\nAnswer the following medical question accurately and safely.\n\n### Input:\n{qa['question']}\n\n### Response:\n{qa['answer']}"
        hf_data.append({"text": text})
    
    return hf_data


def create_openai_format(qa_pairs):
    """Create OpenAI fine-tuning format"""
    openai_data = []
    
    system_message = "You are a medical AI assistant. Provide accurate, helpful information while emphasizing that users should consult healthcare professionals for medical advice. Include appropriate disclaimers when discussing treatments or medications."
    
    for qa in qa_pairs:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": qa["question"]},
            {"role": "assistant", "content": qa["answer"]}
        ]
        openai_data.append({"messages": messages})
    
    return openai_data


def create_conversational_format(qa_pairs):
    """Create conversational format"""
    conv_data = []
    
    for qa in qa_pairs:
        conversations = [
            {"from": "human", "value": qa["question"]},
            {"from": "gpt", "value": qa["answer"]}
        ]
        conv_data.append({"conversations": conversations})
    
    return conv_data


def create_instruction_format(qa_pairs):
    """Create instruction-tuning format"""
    inst_data = []
    
    for qa in qa_pairs:
        inst_data.append({
            "instruction": "Answer the following medical question accurately and safely.",
            "input": qa["question"],
            "output": qa["answer"]
        })
    
    return inst_data


def save_jsonl(data, filepath):
    """Save data as JSONL"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def save_json(data, filepath):
    """Save data as JSON"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    # Merge datasets
    merged_qa = merge_datasets()
    
    # Save full merged dataset
    log_info("Saving merged dataset...")
    save_json({
        "metadata": {
            "created": datetime.now().isoformat(),
            "total_qa_pairs": len(merged_qa),
            "description": "Merged medical QA dataset for chatbot training"
        },
        "qa_pairs": merged_qa
    }, OUTPUT_DIR / "chatbot_training_combined.json")
    
    # Split dataset
    splits = split_dataset(merged_qa)
    
    # Create and save different formats
    formats = {
        "huggingface": create_huggingface_format,
        "openai": create_openai_format,
        "conversational": create_conversational_format,
        "instruction": create_instruction_format
    }
    
    log_info("Creating fine-tuning formats...")
    
    format_stats = {}
    
    for format_name, create_func in formats.items():
        log_info(f"  Creating {format_name} format...")
        format_dir = OUTPUT_DIR / "finetuning_data" / format_name
        format_dir.mkdir(parents=True, exist_ok=True)
        
        format_stats[format_name] = {}
        
        for split_name, split_data in splits.items():
            formatted_data = create_func(split_data)
            
            # Save as JSONL
            save_jsonl(formatted_data, format_dir / f"{split_name}.jsonl")
            
            format_stats[format_name][split_name] = len(formatted_data)
    
    # Calculate statistics
    categories = defaultdict(int)
    sources = defaultdict(int)
    languages = defaultdict(int)
    
    for qa in merged_qa:
        categories[qa.get("category", "unknown")] += 1
        sources[qa.get("source", "unknown")] += 1
        languages[qa.get("language", "unknown")] += 1
    
    # Update processing log
    processing_log["end_time"] = datetime.now().isoformat()
    processing_log["status"] = "completed"
    processing_log["statistics"] = {
        "total_qa_pairs": len(merged_qa),
        "train": len(splits["train"]),
        "validation": len(splits["validation"]),
        "test": len(splits["test"]),
        "categories": dict(categories),
        "sources": dict(sources),
        "languages": dict(languages),
        "formats": format_stats
    }
    
    # Save processing report
    save_json(processing_log, OUTPUT_DIR / "processing_report.json")
    
    # Print summary
    log_info("=" * 60)
    log_info("Dataset Merge Complete!")
    log_info("=" * 60)
    log_info(f"Total QA pairs: {len(merged_qa)}")
    log_info(f"  - Train: {len(splits['train'])}")
    log_info(f"  - Validation: {len(splits['validation'])}")
    log_info(f"  - Test: {len(splits['test'])}")
    log_info(f"Output directory: {OUTPUT_DIR}")
    log_info("=" * 60)
    
    return merged_qa


if __name__ == "__main__":
    main()
