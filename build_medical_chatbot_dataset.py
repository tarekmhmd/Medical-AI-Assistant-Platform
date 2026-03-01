"""
Medical Chatbot Dataset Builder
================================
Automatically builds a large, high-quality medical QA dataset from multiple sources.

Sources:
1. MedQuad (local CSV - 22MB)
2. PubMed QA (downloaded)
3. BioASQ (downloaded)
4. Disease database (local JSON)
5. Additional medical QA pairs (generated)

Features:
- Text normalization
- Deduplication
- Data augmentation (synonym replacement, paraphrasing)
- Quality filtering
"""

import os
import json
import csv
import re
import random
import hashlib
import urllib.request
import time
from typing import List, Dict, Set, Tuple
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "output_path": r"D:\project 2\data\chatbot\combined_medical_qa.json",
    "backup_path": r"D:\project 2\data\chatbot\backup",
    "medquad_path": r"D:\project 2\data\archive (1)\medquad.csv",
    "disease_db_path": r"D:\project 2\processed_data\databases\disease_database.json",
    "skin_db_path": r"D:\project 2\processed_data\databases\skin_disease_database.json",
    "respiratory_db_path": r"D:\project 2\processed_data\databases\respiratory_database.json",
    "lab_db_path": r"D:\project 2\processed_data\databases\lab_test_database.json",
    
    # Augmentation settings
    "augment_synonyms": True,
    "augment_paraphrase": True,
    "max_augmented_per_entry": 2,
    
    # Quality filters
    "min_question_length": 10,
    "min_answer_length": 20,
    "max_question_length": 500,
    "max_answer_length": 5000,
    
    # Random seed for reproducibility
    "random_seed": 42,
}

# Medical synonym dictionary for augmentation
MEDICAL_SYNONYMS = {
    "pain": ["discomfort", "ache", "soreness", "tenderness", "hurt"],
    "headache": ["head pain", "migraine", "cephalgia", "head ache"],
    "fever": ["high temperature", "pyrexia", "elevated temperature", "feverish"],
    "cough": ["coughing", "hacking", "dry cough", "productive cough"],
    "cold": ["common cold", "viral infection", "upper respiratory infection"],
    "doctor": ["physician", "medical professional", "healthcare provider", "clinician"],
    "medicine": ["medication", "drug", "pharmaceutical", "remedy", "treatment"],
    "hospital": ["medical center", "healthcare facility", "clinic", "medical facility"],
    "symptoms": ["signs", "indications", "manifestations", "clinical features"],
    "disease": ["illness", "condition", "disorder", "ailment", "affliction"],
    "treatment": ["therapy", "intervention", "management", "care", "remedy"],
    "diagnosis": ["detection", "identification", "prognosis", "clinical assessment"],
    "patient": ["sufferer", "case", "individual", "affected person"],
    "chronic": ["long-term", "persistent", "ongoing", "recurring", "prolonged"],
    "acute": ["sudden", "severe", "rapid-onset", "intense"],
    "infection": ["infectious disease", "contagion", "bacterial/viral illness"],
    "surgery": ["operation", "surgical procedure", "surgical intervention"],
    "blood pressure": ["bp", "hypertension", "hypotension"],
    "diabetes": ["diabetes mellitus", "high blood sugar", "sugar disease"],
    "cancer": ["malignancy", "tumor", "neoplasm", "carcinoma"],
    "heart attack": ["myocardial infarction", "cardiac arrest", "heart failure"],
    "stroke": ["cerebrovascular accident", "brain attack", "cva"],
    "allergy": ["allergic reaction", "hypersensitivity", "immune response"],
    "nausea": ["queasiness", "stomach upset", "feeling sick", "sickness"],
    "vomiting": ["throwing up", "emesis", "being sick", "regurgitation"],
    "diarrhea": ["loose stools", "watery bowel movements", "the runs"],
    "constipation": ["difficulty passing stool", "hard stools", "bowel obstruction"],
    "fatigue": ["tiredness", "exhaustion", "lethargy", "weariness"],
    "insomnia": ["sleeplessness", "difficulty sleeping", "sleep disorder"],
    "anxiety": ["worry", "nervousness", "stress", "unease", "apprehension"],
    "depression": ["low mood", "sadness", "melancholy", "depressive disorder"],
    "injury": ["wound", "trauma", "damage", "harm", "lesion"],
    "rash": ["skin eruption", "dermatitis", "skin reaction", "hives"],
    "swelling": ["edema", "inflammation", "enlargement", "puffiness"],
    "bleeding": ["hemorrhage", "blood loss", "hemorrhaging"],
}

# Common medical question patterns for generating additional QA pairs
QUESTION_PATTERNS = {
    "what is": [
        "What is {condition}?",
        "What are the causes of {condition}?",
        "What are the symptoms of {condition}?",
        "What are the risk factors for {condition}?",
        "What are the complications of {condition}?",
    ],
    "how to": [
        "How is {condition} diagnosed?",
        "How is {condition} treated?",
        "How can {condition} be prevented?",
        "How do I know if I have {condition}?",
    ],
    "when to": [
        "When should I see a doctor for {condition}?",
        "When is {condition} considered serious?",
    ],
    "can": [
        "Can {condition} be cured?",
        "Can {condition} be prevented?",
        "Can {condition} spread to others?",
    ],
}


# =============================================================================
# TEXT PROCESSING UTILITIES
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text by removing special characters and fixing encoding."""
    if not text:
        return ""
    
    # Fix encoding issues
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\?\!\'\"\-\:\;\(\)]', '', text)
    
    # Fix multiple punctuation
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'\!{2,}', '!', text)
    
    # Standardize medical terminology
    medical_terms = {
        "dr.": "Dr.",
        "dr ": "Dr. ",
        "mg": "mg",
        "ml": "ml",
        "kg": "kg",
        "bp": "blood pressure",
        "hr": "heart rate",
        "temp": "temperature",
        "meds": "medications",
        "rx": "prescription",
        "otc": "over-the-counter",
    }
    
    for old, new in medical_terms.items():
        text = re.sub(rf'\b{old}\b', new, text, flags=re.IGNORECASE)
    
    return text.strip()


def clean_question(question: str) -> str:
    """Clean and format a question."""
    question = normalize_text(question)
    
    # Ensure question ends with ?
    if question and not question.endswith('?'):
        question = question.rstrip('.!') + '?'
    
    # Capitalize first letter
    if question:
        question = question[0].upper() + question[1:]
    
    return question


def clean_answer(answer: str) -> str:
    """Clean and format an answer."""
    answer = normalize_text(answer)
    
    # Ensure answer ends with proper punctuation
    if answer and not answer.endswith(('.', '!', '?')):
        answer = answer + '.'
    
    return answer


def get_text_hash(text: str) -> str:
    """Get a hash of text for deduplication."""
    return hashlib.md5(text.lower().encode()).hexdigest()


# =============================================================================
# DATA SOURCE LOADERS
# =============================================================================

def load_medquad(filepath: str) -> List[Dict]:
    """Load MedQuad dataset from CSV."""
    print("\n📥 Loading MedQuad dataset...")
    qa_pairs = []
    
    if not os.path.exists(filepath):
        print(f"   ⚠️ File not found: {filepath}")
        return qa_pairs
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                question = clean_question(row.get('question', ''))
                answer = clean_answer(row.get('answer', ''))
                
                if question and answer:
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'source': 'MedQuad',
                        'focus_area': row.get('focus_area', 'general')
                    })
                    count += 1
        
        print(f"   ✓ Loaded {count} QA pairs from MedQuad")
    except Exception as e:
        print(f"   ✗ Error loading MedQuad: {e}")
    
    return qa_pairs


def load_disease_database(filepath: str) -> List[Dict]:
    """Generate QA pairs from disease database."""
    print("\n📥 Loading disease database...")
    qa_pairs = []
    
    if not os.path.exists(filepath):
        print(f"   ⚠️ File not found: {filepath}")
        return qa_pairs
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for disease in data.get('diseases', []):
            name = disease.get('name', '')
            symptoms = disease.get('symptoms', [])
            treatments = disease.get('treatments', [])
            description = disease.get('description', '')
            causes = disease.get('causes', [])
            prevention = disease.get('prevention', [])
            
            if name:
                # What is this disease?
                if description:
                    qa_pairs.append({
                        'question': f"What is {name}?",
                        'answer': description,
                        'source': 'DiseaseDatabase'
                    })
                
                # What are the symptoms?
                if symptoms:
                    symptom_text = ", ".join(symptoms[:10])
                    qa_pairs.append({
                        'question': f"What are the symptoms of {name}?",
                        'answer': f"The common symptoms of {name} include: {symptom_text}. If you experience these symptoms, consult a healthcare professional.",
                        'source': 'DiseaseDatabase'
                    })
                
                # How is it treated?
                if treatments:
                    treatment_text = ", ".join(treatments[:5])
                    qa_pairs.append({
                        'question': f"How is {name} treated?",
                        'answer': f"Treatment options for {name} include: {treatment_text}. Always follow your doctor's recommendations.",
                        'source': 'DiseaseDatabase'
                    })
                
                # What causes it?
                if causes:
                    causes_text = ", ".join(causes[:5])
                    qa_pairs.append({
                        'question': f"What causes {name}?",
                        'answer': f"The causes of {name} include: {causes_text}. Understanding the cause can help in prevention and treatment.",
                        'source': 'DiseaseDatabase'
                    })
                
                # How to prevent?
                if prevention:
                    prevention_text = ", ".join(prevention[:5])
                    qa_pairs.append({
                        'question': f"How can {name} be prevented?",
                        'answer': f"Prevention measures for {name} include: {prevention_text}. Following these guidelines can reduce your risk.",
                        'source': 'DiseaseDatabase'
                    })
        
        print(f"   ✓ Generated {len(qa_pairs)} QA pairs from disease database")
    except Exception as e:
        print(f"   ✗ Error loading disease database: {e}")
    
    return qa_pairs


def load_skin_database(filepath: str) -> List[Dict]:
    """Generate QA pairs from skin disease database."""
    print("\n📥 Loading skin disease database...")
    qa_pairs = []
    
    if not os.path.exists(filepath):
        print(f"   ⚠️ File not found: {filepath}")
        return qa_pairs
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for condition in data.get('conditions', []):
            name = condition.get('name', '')
            description = condition.get('description', '')
            symptoms = condition.get('symptoms', [])
            medications = condition.get('medications', [])
            severity = condition.get('severity', '')
            
            if name:
                if description:
                    qa_pairs.append({
                        'question': f"What is {name}?",
                        'answer': description,
                        'source': 'SkinDatabase'
                    })
                
                if symptoms:
                    symptom_text = ", ".join(symptoms[:8])
                    qa_pairs.append({
                        'question': f"What are the symptoms of {name}?",
                        'answer': f"Symptoms of {name} include: {symptom_text}. Severity: {severity}.",
                        'source': 'SkinDatabase'
                    })
                
                if medications:
                    med_text = ", ".join(medications[:5])
                    qa_pairs.append({
                        'question': f"What medications treat {name}?",
                        'answer': f"Medications for {name} include: {med_text}. Consult a dermatologist for proper diagnosis and treatment.",
                        'source': 'SkinDatabase'
                    })
        
        print(f"   ✓ Generated {len(qa_pairs)} QA pairs from skin database")
    except Exception as e:
        print(f"   ✗ Error loading skin database: {e}")
    
    return qa_pairs


def load_respiratory_database(filepath: str) -> List[Dict]:
    """Generate QA pairs from respiratory database."""
    print("\n📥 Loading respiratory database...")
    qa_pairs = []
    
    if not os.path.exists(filepath):
        print(f"   ⚠️ File not found: {filepath}")
        return qa_pairs
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for condition in data.get('conditions', []):
            name = condition.get('name', '')
            description = condition.get('description', '')
            symptoms = condition.get('symptoms', [])
            medications = condition.get('medications', [])
            emergency_signs = condition.get('emergency_signs', [])
            
            if name:
                if description:
                    qa_pairs.append({
                        'question': f"What is {name}?",
                        'answer': description,
                        'source': 'RespiratoryDatabase'
                    })
                
                if symptoms:
                    symptom_text = ", ".join(symptoms[:8])
                    qa_pairs.append({
                        'question': f"What are the symptoms of {name}?",
                        'answer': f"Symptoms of {name} include: {symptom_text}.",
                        'source': 'RespiratoryDatabase'
                    })
                
                if emergency_signs:
                    emergency_text = ", ".join(emergency_signs)
                    qa_pairs.append({
                        'question': f"What are the emergency signs of {name}?",
                        'answer': f"Seek immediate medical attention if you experience: {emergency_text}.",
                        'source': 'RespiratoryDatabase'
                    })
        
        print(f"   ✓ Generated {len(qa_pairs)} QA pairs from respiratory database")
    except Exception as e:
        print(f"   ✗ Error loading respiratory database: {e}")
    
    return qa_pairs


def load_lab_database(filepath: str) -> List[Dict]:
    """Generate QA pairs from lab test database."""
    print("\n📥 Loading lab test database...")
    qa_pairs = []
    
    if not os.path.exists(filepath):
        print(f"   ⚠️ File not found: {filepath}")
        return qa_pairs
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for test in data.get('tests', []):
            name = test.get('name', '')
            description = test.get('description', '')
            normal_range = test.get('normal_range', {})
            purpose = test.get('purpose', '')
            preparation = test.get('preparation', '')
            
            if name:
                if description:
                    qa_pairs.append({
                        'question': f"What is a {name} test?",
                        'answer': description,
                        'source': 'LabDatabase'
                    })
                
                if normal_range:
                    min_val = normal_range.get('min', 'N/A')
                    max_val = normal_range.get('max', 'N/A')
                    unit = normal_range.get('unit', '')
                    qa_pairs.append({
                        'question': f"What is the normal range for {name}?",
                        'answer': f"The normal range for {name} is typically {min_val} to {max_val} {unit}. Values outside this range may indicate a health issue.",
                        'source': 'LabDatabase'
                    })
                
                if purpose:
                    qa_pairs.append({
                        'question': f"Why is the {name} test performed?",
                        'answer': purpose,
                        'source': 'LabDatabase'
                    })
                
                if preparation:
                    qa_pairs.append({
                        'question': f"How should I prepare for a {name} test?",
                        'answer': preparation,
                        'source': 'LabDatabase'
                    })
        
        print(f"   ✓ Generated {len(qa_pairs)} QA pairs from lab database")
    except Exception as e:
        print(f"   ✗ Error loading lab database: {e}")
    
    return qa_pairs


def download_pubmedqa() -> List[Dict]:
    """Download PubMedQA dataset."""
    print("\n📥 Attempting to download PubMedQA...")
    qa_pairs = []
    
    # PubMedQA simplified dataset URL (JSONL format)
    pubmed_url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
    
    try:
        print("   Downloading from GitHub...")
        with urllib.request.urlopen(pubmed_url, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
            
            count = 0
            for pmid, item in data.items():
                question = clean_question(item.get('QUESTION', ''))
                # Get the long answer
                answer = clean_answer(item.get('LONG_ANSWER', ''))
                
                if question and answer:
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'source': 'PubMedQA'
                    })
                    count += 1
                    
                    if count >= 5000:  # Limit to 5000 entries
                        break
            
            print(f"   ✓ Downloaded {count} QA pairs from PubMedQA")
    except Exception as e:
        print(f"   ⚠️ Could not download PubMedQA: {e}")
        print("   Using offline mode - skipping PubMedQA")
    
    return qa_pairs


def generate_general_medical_qa() -> List[Dict]:
    """Generate general medical QA pairs."""
    print("\n📝 Generating general medical QA pairs...")
    qa_pairs = []
    
    general_qa = [
        # General health questions
        {
            "question": "What should I do if I have a fever?",
            "answer": "If you have a fever: 1) Rest and stay hydrated, 2) Take over-the-counter fever reducers like acetaminophen or ibuprofen, 3) Use a cool compress, 4) Seek medical attention if fever exceeds 103°F (39.4°C) or lasts more than 3 days."
        },
        {
            "question": "How often should I have a check-up?",
            "answer": "Adults should have a general health check-up at least once a year. More frequent visits may be needed if you have chronic conditions, are over 50, or have specific health concerns."
        },
        {
            "question": "What are the warning signs of a heart attack?",
            "answer": "Warning signs of a heart attack include: chest pain or pressure, pain radiating to the arm, jaw, or back, shortness of breath, nausea, cold sweats, and unusual fatigue. Call emergency services immediately if these occur."
        },
        {
            "question": "How can I improve my sleep quality?",
            "answer": "To improve sleep: maintain a regular sleep schedule, avoid caffeine and screens before bed, keep your bedroom cool and dark, exercise regularly but not close to bedtime, and limit naps to 20-30 minutes."
        },
        {
            "question": "What are the symptoms of dehydration?",
            "answer": "Symptoms of dehydration include: extreme thirst, dry mouth, dark urine, fatigue, dizziness, confusion, and infrequent urination. Severe dehydration requires immediate medical attention."
        },
        {
            "question": "How can I boost my immune system?",
            "answer": "To boost immunity: eat a balanced diet rich in fruits and vegetables, exercise regularly, get adequate sleep, manage stress, maintain a healthy weight, avoid smoking, and stay up to date with vaccinations."
        },
        {
            "question": "When should I go to the emergency room?",
            "answer": "Go to the ER for: difficulty breathing, chest pain, severe bleeding, head trauma, signs of stroke (FAST: Face drooping, Arm weakness, Speech difficulty, Time to call 911), severe allergic reactions, or loss of consciousness."
        },
        {
            "question": "What is the difference between a cold and the flu?",
            "answer": "Colds develop gradually with mild symptoms like runny nose and sore throat. Flu comes on suddenly with fever, body aches, fatigue, and more severe symptoms. Flu can lead to serious complications."
        },
        {
            "question": "How much water should I drink daily?",
            "answer": "The general recommendation is about 8 glasses (64 ounces) of water per day. Individual needs vary based on activity level, climate, and health conditions. Urine color should be light yellow."
        },
        {
            "question": "What are healthy blood pressure numbers?",
            "answer": "Normal blood pressure is below 120/80 mmHg. Elevated is 120-129/less than 80. High blood pressure (hypertension) is 130/80 or higher. Consult your doctor for personalized guidance."
        },
        # Medication questions
        {
            "question": "Should I take antibiotics for a viral infection?",
            "answer": "No, antibiotics only work against bacterial infections, not viruses like colds or flu. Taking antibiotics unnecessarily can lead to antibiotic resistance. Your doctor will prescribe antibiotics only when needed."
        },
        {
            "question": "Can I take expired medication?",
            "answer": "It's not recommended to take expired medication. While some medications may still be effective, others can degrade or become harmful. Always check expiration dates and dispose of expired medications properly."
        },
        {
            "question": "What should I tell my doctor before starting a new medication?",
            "answer": "Tell your doctor about: all current medications and supplements, allergies, previous adverse reactions, chronic conditions, pregnancy or breastfeeding status, and any lifestyle factors (alcohol, smoking)."
        },
        # Mental health
        {
            "question": "What are signs of anxiety?",
            "answer": "Signs of anxiety include: excessive worry, restlessness, fatigue, difficulty concentrating, irritability, muscle tension, sleep problems, and physical symptoms like rapid heartbeat or sweating. Seek help if symptoms interfere with daily life."
        },
        {
            "question": "How can I manage stress?",
            "answer": "Stress management techniques include: regular exercise, deep breathing exercises, meditation, adequate sleep, time management, maintaining social connections, limiting caffeine and alcohol, and seeking professional help when needed."
        },
        {
            "question": "When should I seek help for depression?",
            "answer": "Seek help if you experience persistent sadness, loss of interest in activities, changes in sleep or appetite, difficulty functioning, or thoughts of self-harm. Depression is treatable with therapy, medication, or both."
        },
        # Prevention
        {
            "question": "What vaccines do adults need?",
            "answer": "Adult vaccines include: annual flu shot, Tdap (tetanus, diphtheria, pertussis) every 10 years, shingles vaccine (50+), pneumonia vaccine (65+ or with conditions), and COVID-19 vaccines. Consult your doctor for personalized recommendations."
        },
        {
            "question": "How can I prevent heart disease?",
            "answer": "Prevent heart disease by: not smoking, exercising regularly (150 minutes/week), eating a heart-healthy diet, maintaining healthy weight, managing stress, controlling blood pressure and cholesterol, and limiting alcohol."
        },
        {
            "question": "What are the screening tests I should get?",
            "answer": "Important screenings include: blood pressure (annually), cholesterol (every 4-6 years), diabetes (if at risk), colorectal cancer (45+), breast cancer (women 40+), cervical cancer (women 21-65), and prostate cancer (men 50+)."
        },
        # Common conditions
        {
            "question": "What causes headaches?",
            "answer": "Headaches can be caused by: stress, dehydration, lack of sleep, eye strain, poor posture, certain foods, caffeine, hormonal changes, or underlying conditions. See a doctor for severe, persistent, or unusual headaches."
        },
        {
            "question": "How can I treat a minor burn at home?",
            "answer": "For minor burns: cool the burn under running water for 10-20 minutes, cover with a clean bandage, take pain relievers if needed, and don't apply ice, butter, or ointments. Seek medical care for severe burns."
        },
        {
            "question": "What are the symptoms of food poisoning?",
            "answer": "Food poisoning symptoms include: nausea, vomiting, diarrhea, abdominal cramps, fever, and dehydration. Most cases resolve within 48 hours. Seek medical care for severe symptoms, dehydration, or if symptoms persist."
        },
        {
            "question": "How can I prevent back pain?",
            "answer": "Prevent back pain by: maintaining good posture, exercising regularly (especially core-strengthening), using proper lifting techniques, maintaining healthy weight, sleeping on a supportive mattress, and taking breaks from sitting."
        },
        {
            "question": "What is the recommended amount of exercise?",
            "answer": "Adults should aim for at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity weekly, plus muscle-strengthening activities twice a week. Even small amounts of activity are beneficial."
        },
        {
            "question": "How can I maintain a healthy weight?",
            "answer": "Maintain healthy weight through: balanced nutrition, portion control, regular physical activity, adequate sleep, stress management, and avoiding fad diets. Consult healthcare providers for personalized guidance."
        },
    ]
    
    for qa in general_qa:
        qa_pairs.append({
            'question': clean_question(qa['question']),
            'answer': clean_answer(qa['answer']),
            'source': 'GeneralMedical'
        })
    
    print(f"   ✓ Generated {len(qa_pairs)} general medical QA pairs")
    return qa_pairs


# =============================================================================
# DATA AUGMENTATION
# =============================================================================

def synonym_augment(text: str, synonyms: Dict, num_replacements: int = 2) -> str:
    """Augment text by replacing words with synonyms."""
    words = text.split()
    augmented = words.copy()
    
    replacements_made = 0
    for i, word in enumerate(words):
        word_lower = word.lower()
        if word_lower in synonyms and replacements_made < num_replacements:
            replacement = random.choice(synonyms[word_lower])
            # Preserve capitalization
            if word[0].isupper():
                replacement = replacement.capitalize()
            augmented[i] = replacement
            replacements_made += 1
    
    return ' '.join(augmented)


def augment_dataset(qa_pairs: List[Dict], synonyms: Dict, max_augmented: int = 2) -> List[Dict]:
    """Augment the dataset with paraphrased questions."""
    print("\n🔄 Augmenting dataset...")
    augmented_pairs = []
    
    for qa in qa_pairs:
        # Add original
        augmented_pairs.append(qa)
        
        # Generate augmented versions
        for _ in range(min(max_augmented, 1)):
            if random.random() < 0.3:  # 30% chance to augment
                aug_question = synonym_augment(qa['question'], synonyms, num_replacements=2)
                if aug_question != qa['question']:
                    augmented_pairs.append({
                        'question': aug_question,
                        'answer': qa['answer'],
                        'source': qa['source'] + '_Augmented'
                    })
    
    print(f"   ✓ Augmented dataset from {len(qa_pairs)} to {len(augmented_pairs)} entries")
    return augmented_pairs


# =============================================================================
# DEDUPLICATION AND FILTERING
# =============================================================================

def deduplicate_qa(qa_pairs: List[Dict]) -> List[Dict]:
    """Remove duplicate questions and answers."""
    print("\n🔍 Deduplicating dataset...")
    
    seen_questions: Set[str] = set()
    seen_answer_hashes: Set[str] = set()
    unique_pairs: List[Dict] = []
    
    duplicates = 0
    
    for qa in qa_pairs:
        q_hash = get_text_hash(qa['question'])
        a_hash = get_text_hash(qa['answer'])
        
        # Check for duplicate questions
        if q_hash in seen_questions:
            duplicates += 1
            continue
        
        # Check for very similar answers (optional)
        if a_hash in seen_answer_hashes:
            duplicates += 1
            continue
        
        seen_questions.add(q_hash)
        seen_answer_hashes.add(a_hash)
        unique_pairs.append(qa)
    
    print(f"   ✓ Removed {duplicates} duplicates")
    print(f"   ✓ Unique entries: {len(unique_pairs)}")
    
    return unique_pairs


def filter_by_quality(qa_pairs: List[Dict], config: Dict) -> List[Dict]:
    """Filter QA pairs by quality criteria."""
    print("\n✅ Filtering by quality...")
    
    filtered = []
    removed = 0
    
    for qa in qa_pairs:
        q_len = len(qa['question'])
        a_len = len(qa['answer'])
        
        # Check length constraints
        if q_len < config['min_question_length']:
            removed += 1
            continue
        if a_len < config['min_answer_length']:
            removed += 1
            continue
        if q_len > config['max_question_length']:
            removed += 1
            continue
        if a_len > config['max_answer_length']:
            removed += 1
            continue
        
        # Check for meaningful content
        if not qa['answer'].strip() or not qa['question'].strip():
            removed += 1
            continue
        
        filtered.append(qa)
    
    print(f"   ✓ Removed {removed} low-quality entries")
    print(f"   ✓ High-quality entries: {len(filtered)}")
    
    return filtered


# =============================================================================
# MAIN BUILD FUNCTION
# =============================================================================

def build_medical_chatbot_dataset():
    """Main function to build the medical chatbot dataset."""
    print("=" * 70)
    print("🏥 MEDICAL CHATBOT DATASET BUILDER")
    print("=" * 70)
    
    random.seed(CONFIG['random_seed'])
    
    # Create output directory
    os.makedirs(os.path.dirname(CONFIG['output_path']), exist_ok=True)
    os.makedirs(CONFIG['backup_path'], exist_ok=True)
    
    # Step 1: Load all data sources
    print("\n" + "=" * 50)
    print("STEP 1: LOADING DATA SOURCES")
    print("=" * 50)
    
    all_qa_pairs = []
    
    # Load MedQuad (main source)
    all_qa_pairs.extend(load_medquad(CONFIG['medquad_path']))
    
    # Load local databases
    all_qa_pairs.extend(load_disease_database(CONFIG['disease_db_path']))
    all_qa_pairs.extend(load_skin_database(CONFIG['skin_db_path']))
    all_qa_pairs.extend(load_respiratory_database(CONFIG['respiratory_db_path']))
    all_qa_pairs.extend(load_lab_database(CONFIG['lab_db_path']))
    
    # Download additional datasets
    all_qa_pairs.extend(download_pubmedqa())
    
    # Generate general medical QA
    all_qa_pairs.extend(generate_general_medical_qa())
    
    print(f"\n📊 Total entries loaded: {len(all_qa_pairs)}")
    
    # Step 2: Normalize text
    print("\n" + "=" * 50)
    print("STEP 2: NORMALIZING TEXT")
    print("=" * 50)
    
    for qa in all_qa_pairs:
        qa['question'] = clean_question(qa['question'])
        qa['answer'] = clean_answer(qa['answer'])
    
    print("   ✓ Text normalization complete")
    
    # Step 3: Filter by quality
    print("\n" + "=" * 50)
    print("STEP 3: QUALITY FILTERING")
    print("=" * 50)
    
    filtered_qa = filter_by_quality(all_qa_pairs, CONFIG)
    
    # Step 4: Deduplicate
    print("\n" + "=" * 50)
    print("STEP 4: DEDUPLICATION")
    print("=" * 50)
    
    unique_qa = deduplicate_qa(filtered_qa)
    
    # Step 5: Augmentation
    print("\n" + "=" * 50)
    print("STEP 5: DATA AUGMENTATION")
    print("=" * 50)
    
    if CONFIG['augment_synonyms']:
        augmented_qa = augment_dataset(
            unique_qa, 
            MEDICAL_SYNONYMS, 
            CONFIG['max_augmented_per_entry']
        )
    else:
        augmented_qa = unique_qa
    
    # Step 6: Final deduplication after augmentation
    print("\n" + "=" * 50)
    print("STEP 6: FINAL DEDUPLICATION")
    print("=" * 50)
    
    final_qa = deduplicate_qa(augmented_qa)
    
    # Step 7: Save to JSON
    print("\n" + "=" * 50)
    print("STEP 7: SAVING DATASET")
    print("=" * 50)
    
    # Check for existing file and create backup
    if os.path.exists(CONFIG['output_path']):
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(
            CONFIG['backup_path'], 
            f"combined_medical_qa_backup_{timestamp}.json"
        )
        
        try:
            import shutil
            shutil.copy(CONFIG['output_path'], backup_file)
            print(f"   ✓ Backup created: {backup_file}")
        except Exception as e:
            print(f"   ⚠️ Could not create backup: {e}")
    
    # Save the final dataset
    try:
        with open(CONFIG['output_path'], 'w', encoding='utf-8') as f:
            json.dump(final_qa, f, ensure_ascii=False, indent=2)
        
        file_size = os.path.getsize(CONFIG['output_path']) / (1024 * 1024)
        print(f"   ✓ Dataset saved to: {CONFIG['output_path']}")
        print(f"   ✓ File size: {file_size:.2f} MB")
    except Exception as e:
        print(f"   ✗ Error saving dataset: {e}")
        return
    
    # Step 8: Verification and summary
    print("\n" + "=" * 70)
    print("📊 DATASET VERIFICATION & SUMMARY")
    print("=" * 70)
    
    # Count by source
    source_counts = defaultdict(int)
    for qa in final_qa:
        source_counts[qa.get('source', 'Unknown')] += 1
    
    print(f"\n📈 Total entries: {len(final_qa)}")
    print("\n📋 Entries by source:")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {source}: {count:,}")
    
    # Print sample entries
    print("\n📝 Sample entries (5 random):")
    print("-" * 50)
    
    samples = random.sample(final_qa, min(5, len(final_qa)))
    for i, qa in enumerate(samples, 1):
        print(f"\n{i}. Q: {qa['question']}")
        answer_preview = qa['answer'][:200] + "..." if len(qa['answer']) > 200 else qa['answer']
        print(f"   A: {answer_preview}")
        print(f"   Source: {qa.get('source', 'Unknown')}")
    
    print("\n" + "=" * 70)
    print("✅ DATASET BUILD COMPLETE!")
    print("=" * 70)
    
    return final_qa


if __name__ == "__main__":
    build_medical_chatbot_dataset()
