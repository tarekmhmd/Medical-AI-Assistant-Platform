"""
Drug Dataset Preparation Script for Medical Chatbot Training
Automates: Discovery, Extraction, Preprocessing, and Chatbot Dataset Creation
"""

import os
import sys
import json
import zipfile
import shutil
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Setup paths
BASE_DIR = Path(r"D:\project 2")
DATA_DIR = BASE_DIR / "data"
ZIP_FILE = DATA_DIR / "drugs-datasets-master.zip"
EXTRACTED_DIR = DATA_DIR / "drugs-datasets-master"
PROCESSED_DIR = DATA_DIR / "drugs-datasets-processed"
LOG_FILE = BASE_DIR / "logs" / "drugs_dataset_processing.log"

# Create directories
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
(PROCESSED_DIR / "drugs").mkdir(exist_ok=True)
(PROCESSED_DIR / "active_ingredients").mkdir(exist_ok=True)
(PROCESSED_DIR / "diseases").mkdir(exist_ok=True)
(PROCESSED_DIR / "side_effects").mkdir(exist_ok=True)
(BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Processing log for tracking
processing_log = {
    "start_time": datetime.now().isoformat(),
    "files_processed": [],
    "transformations": [],
    "warnings": [],
    "errors": [],
    "statistics": {}
}


def log_event(event_type, message):
    """Log events to both logger and processing log"""
    if event_type == "info":
        logger.info(message)
    elif event_type == "warning":
        logger.warning(message)
        processing_log["warnings"].append(message)
    elif event_type == "error":
        logger.error(message)
        processing_log["errors"].append(message)
    processing_log["transformations"].append({
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "message": message
    })


def extract_zip():
    """Extract the drugs-datasets-master.zip file"""
    log_event("info", f"Starting extraction of {ZIP_FILE}")
    
    if not ZIP_FILE.exists():
        log_event("error", f"Zip file not found: {ZIP_FILE}")
        return False
    
    try:
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        log_event("info", f"Successfully extracted to {EXTRACTED_DIR}")
        return True
    except Exception as e:
        log_event("error", f"Failed to extract zip: {str(e)}")
        return False


def discover_files(directory):
    """Discover all relevant data files in the directory"""
    relevant_extensions = {'.csv', '.json', '.txt', '.xlsx', '.xls', '.tsv'}
    files = {
        'csv': [],
        'json': [],
        'txt': [],
        'excel': [],
        'other': []
    }
    
    if not directory.exists():
        log_event("warning", f"Directory not found: {directory}")
        return files
    
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            filepath = Path(root) / filename
            ext = filepath.suffix.lower()
            
            if ext in {'.csv', '.tsv'}:
                files['csv'].append(str(filepath))
            elif ext == '.json':
                files['json'].append(str(filepath))
            elif ext in {'.xlsx', '.xls'}:
                files['excel'].append(str(filepath))
            elif ext == '.txt':
                files['txt'].append(str(filepath))
    
    log_event("info", f"Discovered: {len(files['csv'])} CSV, {len(files['json'])} JSON, {len(files['excel'])} Excel, {len(files['txt'])} TXT files")
    return files


def categorize_file(filepath, content_sample=None):
    """Categorize a file based on its name and content"""
    filepath_lower = filepath.lower()
    
    categories = {
        'drugs': ['drug', 'medication', 'medicine', 'product', 'brand'],
        'active_ingredients': ['ingredient', 'active', 'substance', 'compound'],
        'diseases': ['disease', 'condition', 'indication', 'diagnosis', 'illness'],
        'side_effects': ['side', 'effect', 'adverse', 'reaction', 'toxicity', 'warning']
    }
    
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in filepath_lower:
                return category
    
    return 'drugs'  # Default category


def read_csv_safe(filepath):
    """Safely read CSV file with different encodings"""
    import csv
    
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding, errors='replace') as f:
                reader = csv.DictReader(f)
                data = list(reader)
                return data, reader.fieldnames
        except Exception as e:
            continue
    
    log_event("warning", f"Could not read CSV file: {filepath}")
    return None, None


def read_json_safe(filepath):
    """Safely read JSON file"""
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                data = json.load(f)
                return data
        except Exception as e:
            continue
    
    log_event("warning", f"Could not read JSON file: {filepath}")
    return None


def standardize_columns(columns):
    """Standardize column names to a consistent format"""
    if not columns:
        return []
    
    column_mapping = {
        'drugname': 'drug_name',
        'drug_name': 'drug_name',
        'drug': 'drug_name',
        'medicinalproduct': 'drug_name',
        'productname': 'drug_name',
        'brandname': 'drug_name',
        'activeingredient': 'active_ingredient',
        'active_ingredient': 'active_ingredient',
        'ingredient': 'active_ingredient',
        'substancename': 'active_ingredient',
        'drugindication': 'indication',
        'indication': 'indication',
        'disease': 'disease',
        'condition': 'condition',
        'reaction': 'side_effect',
        'sideeffect': 'side_effect',
        'side_effect': 'side_effect',
        'adversereaction': 'side_effect',
        'reactionmeddrapt': 'side_effect',
        'serious': 'is_serious',
        'seriousness': 'is_serious',
        'patientage': 'patient_age',
        'patientsex': 'patient_sex',
        'patientweight': 'patient_weight',
        'dosage': 'dosage',
        'dose': 'dosage',
        'route': 'administration_route',
        'administrationroute': 'administration_route'
    }
    
    standardized = []
    for col in columns:
        col_lower = col.lower().replace(' ', '_').replace('-', '_')
        standardized.append(column_mapping.get(col_lower, col_lower))
    
    return standardized


def handle_missing_values(data, columns):
    """Handle missing values in the dataset"""
    if not data or not columns:
        return data
    
    filled_count = 0
    for row in data:
        for col in columns:
            if col not in row or row[col] is None or str(row[col]).strip() == '':
                row[col] = 'N/A'
                filled_count += 1
    
    if filled_count > 0:
        log_event("info", f"Filled {filled_count} missing values")
    
    return data


def process_csv_file(filepath, target_dir):
    """Process a single CSV file"""
    log_event("info", f"Processing CSV: {filepath}")
    
    data, columns = read_csv_safe(filepath)
    if not data:
        return None
    
    # Standardize columns
    std_columns = standardize_columns(columns)
    
    # Update column names in data
    processed_data = []
    for row in data:
        new_row = {}
        for old_col, new_col in zip(columns, std_columns):
            if old_col in row:
                new_row[new_col] = row[old_col]
        processed_data.append(new_row)
    
    # Handle missing values
    processed_data = handle_missing_values(processed_data, std_columns)
    
    # Determine category and save
    category = categorize_file(filepath)
    filename = Path(filepath).stem + "_processed.json"
    output_path = target_dir / category / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "source_file": str(filepath),
            "columns": std_columns,
            "row_count": len(processed_data),
            "data": processed_data
        }, f, indent=2, ensure_ascii=False)
    
    log_event("info", f"Saved processed data to {output_path}")
    processing_log["files_processed"].append({
        "original": str(filepath),
        "processed": str(output_path),
        "rows": len(processed_data),
        "category": category
    })
    
    return {
        "category": category,
        "columns": std_columns,
        "data": processed_data
    }


def process_json_file(filepath, target_dir):
    """Process a single JSON file"""
    log_event("info", f"Processing JSON: {filepath}")
    
    data = read_json_safe(filepath)
    if not data:
        return None
    
    # Handle both list and dict formats
    if isinstance(data, dict):
        if 'results' in data:
            data = data['results']
        elif 'data' in data:
            data = data['data']
        else:
            data = [data]
    
    if not isinstance(data, list):
        data = [data]
    
    # Standardize keys
    if data and isinstance(data[0], dict):
        columns = list(data[0].keys())
        std_columns = standardize_columns(columns)
        
        processed_data = []
        for row in data:
            new_row = {}
            for old_col, new_col in zip(columns, std_columns):
                if old_col in row:
                    new_row[new_col] = row[old_col]
            processed_data.append(new_row)
        
        processed_data = handle_missing_values(processed_data, std_columns)
    else:
        processed_data = data
        std_columns = []
    
    category = categorize_file(filepath)
    filename = Path(filepath).stem + "_processed.json"
    output_path = target_dir / category / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "source_file": str(filepath),
            "columns": std_columns,
            "row_count": len(processed_data),
            "data": processed_data
        }, f, indent=2, ensure_ascii=False)
    
    log_event("info", f"Saved processed JSON to {output_path}")
    processing_log["files_processed"].append({
        "original": str(filepath),
        "processed": str(output_path),
        "rows": len(processed_data),
        "category": category
    })
    
    return {
        "category": category,
        "columns": std_columns,
        "data": processed_data
    }


def load_processed_data(processed_dir):
    """Load all processed JSON files into structured dictionaries"""
    data = {
        'commonnamegroup': {},  # Drug ID -> Drug info
        'indication': {},       # Indication ID -> Indication name
        'sideeffect': {},       # Side effect ID -> Side effect name
        'atcclass': {},         # ATC ID -> ATC info
        'drug_indications': defaultdict(list),  # Drug ID -> [Indication IDs]
        'drug_atc': defaultdict(list),          # Drug ID -> [ATC IDs]
        'cim10': {},            # ICD-10 code -> info
        'cim10_indicationgroup': defaultdict(list),
        'indicationgroup': {},
        'indicationgroup_indication': defaultdict(list)
    }
    
    drugs_dir = processed_dir / 'drugs'
    
    # Load commonnamegroup (drugs)
    drug_file = drugs_dir / 'commonnamegroup_processed.json'
    if drug_file.exists():
        with open(drug_file, 'r', encoding='utf-8') as f:
            content = json.load(f)
            for item in content.get('data', []):
                drug_id = item.get('commonnamegroupid', '')
                if drug_id:
                    data['commonnamegroup'][drug_id] = {
                        'name': item.get('name', ''),
                        'publicname': item.get('publicname', ''),
                        'name_noaccent': item.get('name_noaccent', '')
                    }
    
    # Load indications
    indication_file = drugs_dir / 'indication_processed.json'
    if indication_file.exists():
        with open(indication_file, 'r', encoding='utf-8') as f:
            content = json.load(f)
            for item in content.get('data', []):
                ind_id = item.get('indicationid', '')
                if ind_id:
                    data['indication'][ind_id] = item.get('name', '')
    
    # Load side effects
    sideeffect_file = drugs_dir / 'sideeffect_processed.json'
    if sideeffect_file.exists():
        with open(sideeffect_file, 'r', encoding='utf-8') as f:
            content = json.load(f)
            for item in content.get('data', []):
                se_id = item.get('sideeffectid', '')
                if se_id:
                    data['sideeffect'][se_id] = item.get('name', '')
    
    # Load ATC classification
    atc_file = drugs_dir / 'atcclass_processed.json'
    if atc_file.exists():
        with open(atc_file, 'r', encoding='utf-8') as f:
            content = json.load(f)
            for item in content.get('data', []):
                atc_id = item.get('atcclassid', '')
                if atc_id:
                    data['atcclass'][atc_id] = {
                        'name': item.get('name', ''),
                        'code': item.get('code', ''),
                        'parentid': item.get('parentid', '')
                    }
    
    # Load drug-indication relationships
    drug_ind_file = drugs_dir / 'commonnamegroup_indication_processed.json'
    if drug_ind_file.exists():
        with open(drug_ind_file, 'r', encoding='utf-8') as f:
            content = json.load(f)
            for item in content.get('data', []):
                drug_id = item.get('commonnamegroupid', '')
                ind_id = item.get('indicationid', '')
                if drug_id and ind_id:
                    data['drug_indications'][drug_id].append(ind_id)
    
    # Load drug-ATC relationships
    drug_atc_file = drugs_dir / 'commonnamegroup_atc_processed.json'
    if drug_atc_file.exists():
        with open(drug_atc_file, 'r', encoding='utf-8') as f:
            content = json.load(f)
            for item in content.get('data', []):
                drug_id = item.get('commonnamegroupid', '')
                atc_id = item.get('atcclassid', '')
                if drug_id and atc_id:
                    data['drug_atc'][drug_id].append(atc_id)
    
    # Load CIM10 (ICD-10)
    cim10_file = drugs_dir / 'cim10_processed.json'
    if cim10_file.exists():
        with open(cim10_file, 'r', encoding='utf-8') as f:
            content = json.load(f)
            for item in content.get('data', []):
                cim10_id = item.get('cim10id', '')
                if cim10_id:
                    data['cim10'][cim10_id] = {
                        'code': item.get('code', ''),
                        'name': item.get('name', '')
                    }
    
    return data


def extract_active_ingredient(drug_name):
    """Extract active ingredient from drug name (Spanish format)"""
    if not drug_name:
        return []
    
    ingredients = []
    
    # Pattern: "ingredient * dosage" or "ingredient1 + ingredient2"
    # Example: "ácido salicílico * 0,1 % + triamcinolona acetónido * 0,02 %"
    import re
    
    # Remove dosage and form information
    name_clean = re.sub(r'\*\s*[\d,.\s%µ]+', '', drug_name)
    name_clean = re.sub(r'\s*;\s*[^;]+$', '', name_clean)  # Remove route/form
    
    # Split by + for combinations
    parts = re.split(r'\s*\+\s*', name_clean)
    
    for part in parts:
        part = part.strip()
        # Remove common suffixes
        part = re.sub(r'\s*\([^)]+\)\s*$', '', part)
        if part and len(part) > 2:
            ingredients.append(part)
    
    return ingredients


def create_chatbot_dataset(processed_dir):
    """Create chatbot-ready JSON dataset by joining relational tables"""
    log_event("info", "Creating chatbot training dataset with relational joins...")
    
    chatbot_dataset = {
        "metadata": {
            "created": datetime.now().isoformat(),
            "version": "2.0",
            "description": "Medical Drug Information Dataset for Chatbot Training - Relational Data Joined"
        },
        "drug_database": [],
        "qa_pairs": []
    }
    
    # Load all processed data
    data = load_processed_data(processed_dir)
    
    log_event("info", f"Loaded {len(data['commonnamegroup'])} drugs, {len(data['indication'])} indications, {len(data['sideeffect'])} side effects")
    
    # Build comprehensive drug database
    for drug_id, drug_info in data['commonnamegroup'].items():
        drug_name = drug_info.get('publicname', drug_info.get('name', ''))
        
        if not drug_name or drug_name == 'N/A':
            continue
        
        # Get indications for this drug
        indication_ids = data['drug_indications'].get(drug_id, [])
        indications = []
        for ind_id in indication_ids:
            ind_name = data['indication'].get(ind_id, '')
            if ind_name and ind_name != 'N/A':
                indications.append(ind_name)
        
        # Get ATC classification
        atc_ids = data['drug_atc'].get(drug_id, [])
        atc_names = []
        atc_codes = []
        for atc_id in atc_ids:
            atc_info = data['atcclass'].get(atc_id, {})
            if atc_info.get('name'):
                atc_names.append(atc_info['name'])
            if atc_info.get('code') and atc_info['code'] != 'N/A':
                atc_codes.append(atc_info['code'])
        
        # Extract active ingredients from drug name
        active_ingredients = extract_active_ingredient(drug_info.get('name', ''))
        
        drug_entry = {
            "drug_id": drug_id,
            "name": drug_name,
            "full_name": drug_info.get('name', ''),
            "active_ingredients": active_ingredients,
            "indications": list(set(indications)),
            "atc_classification": atc_names[:3],  # Top 3 ATC classes
            "atc_codes": atc_codes[:3],
            "side_effects": []  # Will be populated if we have drug-sideeffect relationships
        }
        
        chatbot_dataset["drug_database"].append(drug_entry)
    
    # Generate QA pairs for training
    for drug in chatbot_dataset["drug_database"]:
        drug_name = drug["name"]
        indications = drug["indications"]
        ingredients = drug["active_ingredients"]
        
        # Skip if no useful information
        if not indications and not ingredients:
            continue
        
        # QA: Active ingredients
        if ingredients:
            chatbot_dataset["qa_pairs"].append({
                "instruction": "Provide information about the active ingredient in this medication.",
                "input": f"What is the active ingredient in {drug_name}?",
                "output": f"The active ingredient(s) in {drug_name} is/are: {', '.join(ingredients)}.",
                "category": "active_ingredient"
            })
        
        # QA: Indications/Uses
        if indications:
            # Limit to first 5 indications for readability
            ind_list = indications[:5]
            chatbot_dataset["qa_pairs"].append({
                "instruction": "Provide information about what this medication is used for.",
                "input": f"What is {drug_name} used for?",
                "output": f"{drug_name} is indicated for: {', '.join(ind_list)}. Always consult your healthcare provider for proper diagnosis and treatment.",
                "category": "indication"
            })
        
        # QA: General drug information
        general_answer = f"{drug_name} is a medication"
        if ingredients:
            general_answer += f" containing {', '.join(ingredients[:2])}"
        if indications:
            general_answer += f". It is used for {', '.join(indications[:3])}"
        general_answer += ". Always consult your healthcare provider before use."
        
        chatbot_dataset["qa_pairs"].append({
            "instruction": "Provide general information about this medication.",
            "input": f"Tell me about {drug_name}.",
            "output": general_answer,
            "category": "general"
        })
        
        # QA: ATC classification
        if drug["atc_codes"]:
            chatbot_dataset["qa_pairs"].append({
                "instruction": "Provide ATC classification information.",
                "input": f"What is the ATC code for {drug_name}?",
                "output": f"{drug_name} has ATC code(s): {', '.join(drug['atc_codes'])}. This classification helps identify the drug's therapeutic use.",
                "category": "classification"
            })
    
    # Add medical disclaimer QA pairs
    chatbot_dataset["qa_pairs"].extend([
        {
            "instruction": "Provide medical disclaimer.",
            "input": "Can you prescribe medication?",
            "output": "I cannot prescribe medication. I am an AI assistant designed to provide general information about medications. Please consult a licensed healthcare professional for medical advice, diagnosis, or treatment.",
            "category": "safety"
        },
        {
            "instruction": "Provide safety guidance.",
            "input": "What should I do if I experience side effects?",
            "output": "If you experience any side effects from medication, stop taking it and contact your healthcare provider immediately. For severe reactions such as difficulty breathing, chest pain, or swelling, seek emergency medical attention.",
            "category": "safety"
        },
        {
            "instruction": "Provide dosage disclaimer.",
            "input": "What dosage should I take?",
            "output": "I cannot provide specific dosage recommendations. Dosage depends on many factors including your age, weight, medical condition, and other medications. Always follow your doctor's prescription and the medication label instructions.",
            "category": "safety"
        }
    ])
    
    # Save chatbot dataset
    output_path = processed_dir / "chatbot_training_dataset.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chatbot_dataset, f, indent=2, ensure_ascii=False)
    
    log_event("info", f"Created chatbot dataset with {len(chatbot_dataset['drug_database'])} drugs and {len(chatbot_dataset['qa_pairs'])} QA pairs")
    
    # Update statistics
    drugs_with_indications = sum(1 for d in chatbot_dataset["drug_database"] if d["indications"])
    drugs_with_ingredients = sum(1 for d in chatbot_dataset["drug_database"] if d["active_ingredients"])
    
    processing_log["statistics"] = {
        "total_drugs": len(chatbot_dataset["drug_database"]),
        "total_qa_pairs": len(chatbot_dataset["qa_pairs"]),
        "drugs_with_ingredients": drugs_with_ingredients,
        "drugs_with_indications": drugs_with_indications,
        "drugs_with_atc": sum(1 for d in chatbot_dataset["drug_database"] if d["atc_codes"]),
        "qa_categories": {
            "active_ingredient": sum(1 for qa in chatbot_dataset["qa_pairs"] if qa.get("category") == "active_ingredient"),
            "indication": sum(1 for qa in chatbot_dataset["qa_pairs"] if qa.get("category") == "indication"),
            "general": sum(1 for qa in chatbot_dataset["qa_pairs"] if qa.get("category") == "general"),
            "classification": sum(1 for qa in chatbot_dataset["qa_pairs"] if qa.get("category") == "classification"),
            "safety": sum(1 for qa in chatbot_dataset["qa_pairs"] if qa.get("category") == "safety")
        }
    }
    
    return chatbot_dataset


def main():
    """Main execution function"""
    log_event("info", "=" * 60)
    log_event("info", "Starting Drug Dataset Preparation Pipeline")
    log_event("info", "=" * 60)
    
    # Step 1: Extract zip if needed
    if ZIP_FILE.exists() and not EXTRACTED_DIR.exists():
        if not extract_zip():
            log_event("error", "Extraction failed, aborting")
            return
    elif EXTRACTED_DIR.exists():
        log_event("info", f"Directory already exists: {EXTRACTED_DIR}")
    else:
        log_event("error", f"No data source found. Please ensure {ZIP_FILE} exists.")
        return
    
    # Step 2: Discover files
    log_event("info", "Discovering data files...")
    files = discover_files(EXTRACTED_DIR)
    
    # Step 3: Process files
    log_event("info", "Processing discovered files...")
    
    for filepath in files['csv']:
        try:
            process_csv_file(filepath, PROCESSED_DIR)
        except Exception as e:
            log_event("error", f"Error processing {filepath}: {str(e)}")
    
    for filepath in files['json']:
        try:
            process_json_file(filepath, PROCESSED_DIR)
        except Exception as e:
            log_event("error", f"Error processing {filepath}: {str(e)}")
    
    # Step 4: Create chatbot dataset
    log_event("info", "Creating chatbot training dataset...")
    chatbot_dataset = create_chatbot_dataset(PROCESSED_DIR)
    
    # Step 5: Save processing log
    processing_log["end_time"] = datetime.now().isoformat()
    processing_log["status"] = "completed"
    
    log_path = PROCESSED_DIR / "processing_report.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(processing_log, f, indent=2, ensure_ascii=False)
    
    log_event("info", "=" * 60)
    log_event("info", "Processing Complete!")
    log_event("info", f"Total drugs processed: {processing_log['statistics'].get('total_drugs', 0)}")
    log_event("info", f"Total QA pairs generated: {processing_log['statistics'].get('total_qa_pairs', 0)}")
    log_event("info", f"Output directory: {PROCESSED_DIR}")
    log_event("info", "=" * 60)
    
    return chatbot_dataset


if __name__ == "__main__":
    main()
