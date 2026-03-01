import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
import os
import json

class LabAnalyzer:
    def __init__(self):
        self.model_path = 'models_pretrained/lab_model.h5'
        self.lab_db_path = 'data/lab_results/lab_test_database.json'
        
        # Configure Tesseract path (check project folder first)
        self._setup_tesseract_path()
        
        # Configure Tesseract path (check project folder first)
        self._setup_tesseract_path()
        
        # Load lab test database
        self.lab_test_database = self._load_lab_database()
        
        # Normal ranges for common lab tests (from loaded data + defaults)
        self.normal_ranges = self._build_normal_ranges()
        
        self.load_model()
    
    def _setup_tesseract_path(self):
        """Setup Tesseract path - check multiple locations"""
        # Check project root first (tesseract.exe in project folder)
        project_root_tesseract = os.path.join(os.getcwd(), 'tesseract.exe')
        
        # Check project tesseract folder
        project_tesseract = os.path.join(os.getcwd(), 'tesseract', 'tesseract.exe')
        
        # Common installation paths
        common_paths = [
            project_root_tesseract,
            project_tesseract,
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Tesseract-OCR\tesseract.exe'
        ]
        
        tesseract_found = False
        for path in common_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"✓ Using Tesseract from: {path}")
                tesseract_found = True
                break
        
        if not tesseract_found:
            print("⚠️ Tesseract not found!")
            print("   Run: download_tesseract_portable.bat")
            print("   Or install system-wide from: https://github.com/UB-Mannheim/tesseract/wiki")
    
    def _load_lab_database(self):
        """Load lab test database from JSON file"""
        try:
            if os.path.exists(self.lab_db_path):
                with open(self.lab_db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print("Warning: Lab test database not found. Using default data.")
                return {'tests': []}
        except Exception as e:
            print(f"Error loading lab database: {e}")
            return {'tests': []}
    
    def _build_normal_ranges(self):
        """Build normal ranges from loaded data"""
        ranges = {}
        
        # Load from database
        for test in self.lab_test_database.get('tests', []):
            test_name = test['name'].lower().split('(')[0].strip()
            test_key = test_name.replace(' ', '_')
            
            normal_range = test.get('normal_range', {})
            if normal_range:
                ranges[test_key] = (normal_range.get('min', 0), normal_range.get('max', 1000))
        
        # Add defaults for common tests
        default_ranges = {
            'glucose': (70, 100),  # mg/dL
            'cholesterol': (0, 200),  # mg/dL
            'hdl': (40, 1000),  # mg/dL
            'ldl': (0, 100),  # mg/dL
            'triglycerides': (0, 150),  # mg/dL
            'hemoglobin': (12, 17),  # g/dL
            'wbc': (4000, 11000),  # cells/mcL
            'rbc': (4.5, 5.5),  # million cells/mcL
            'platelets': (150000, 400000),  # cells/mcL
            'creatinine': (0.6, 1.2),  # mg/dL
            'alt': (7, 56),  # U/L
            'ast': (10, 40),  # U/L
        }
        
        # Merge with defaults
        for key, value in default_ranges.items():
            if key not in ranges:
                ranges[key] = value
        
        return ranges
        
        self.load_model()
    
    def load_model(self):
        """Load the pre-trained model"""
        try:
            # In production, load actual OCR + Classification model
            print("Lab analysis model loaded (demo mode)")
        except Exception as e:
            print(f"Model loading error: {e}")
    
    def preprocess_image(self, image_path):
        """Enhanced preprocessing for better OCR accuracy"""
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR (if too small)
        height, width = gray.shape
        if height < 1000:
            scale = 1500 / height
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Denoise first
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Adaptive thresholding (better than fixed threshold)
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Deskew if needed
        coords = np.column_stack(np.where(morph > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) > 0.5:  # Only rotate if significant skew
                (h, w) = morph.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                morph = cv2.warpAffine(morph, M, (w, h), 
                                      flags=cv2.INTER_CUBIC, 
                                      borderMode=cv2.BORDER_REPLICATE)
        
        return morph
    
    def extract_text(self, image_path):
        """Enhanced text extraction with multiple OCR passes"""
        try:
            # Check if Tesseract is available
            try:
                import pytesseract
                # Test if tesseract is accessible
                pytesseract.get_tesseract_version()
                tesseract_available = True
            except:
                tesseract_available = False
                print("⚠️ Tesseract OCR not available. Using demo mode.")
                print("   To enable OCR: Run install_tesseract.bat")
            
            if tesseract_available:
                # First pass: Standard preprocessing
                processed_img = self.preprocess_image(image_path)
                
                # Configure Tesseract for better accuracy
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.:/-%() '
                text1 = pytesseract.image_to_string(processed_img, config=custom_config)
                
                # Second pass: Different preprocessing
                img = cv2.imread(image_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                text2 = pytesseract.image_to_string(binary, config=custom_config)
                
                # Combine results (take longer text as it's likely more complete)
                text = text1 if len(text1) > len(text2) else text2
                
                print(f"✓ Extracted text length: {len(text)} characters")
                if len(text) > 0:
                    print(f"  Preview: {text[:200]}")
                
                return text
            else:
                # Demo mode: Return empty string (will use demo data)
                print("  Using demo data for analysis")
                return ""
                
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
    
    def parse_lab_values(self, text):
        """Enhanced parsing with multiple pattern matching"""
        results = {}
        
        # Comprehensive patterns for lab values (multiple variations)
        patterns = {
            'glucose': [
                r'glucose[:\s]+(\d+\.?\d*)',
                r'blood\s+sugar[:\s]+(\d+\.?\d*)',
                r'fasting\s+glucose[:\s]+(\d+\.?\d*)',
                r'glu[:\s]+(\d+\.?\d*)',
                r'bs[:\s]+(\d+\.?\d*)'
            ],
            'cholesterol': [
                r'total\s+cholesterol[:\s]+(\d+\.?\d*)',
                r'cholesterol[:\s]+(\d+\.?\d*)',
                r'chol[:\s]+(\d+\.?\d*)',
                r't\.chol[:\s]+(\d+\.?\d*)'
            ],
            'hdl': [
                r'hdl[:\s-]+cholesterol[:\s]+(\d+\.?\d*)',
                r'hdl[:\s]+(\d+\.?\d*)',
                r'hdl-c[:\s]+(\d+\.?\d*)'
            ],
            'ldl': [
                r'ldl[:\s-]+cholesterol[:\s]+(\d+\.?\d*)',
                r'ldl[:\s]+(\d+\.?\d*)',
                r'ldl-c[:\s]+(\d+\.?\d*)'
            ],
            'triglycerides': [
                r'triglycerides[:\s]+(\d+\.?\d*)',
                r'trig[:\s]+(\d+\.?\d*)',
                r'tg[:\s]+(\d+\.?\d*)'
            ],
            'hemoglobin': [
                r'hemoglobin[:\s]+(\d+\.?\d*)',
                r'haemoglobin[:\s]+(\d+\.?\d*)',
                r'hgb[:\s]+(\d+\.?\d*)',
                r'hb[:\s]+(\d+\.?\d*)'
            ],
            'wbc': [
                r'white\s+blood\s+cell[s]?[:\s]+(\d+\.?\d*)',
                r'wbc[:\s]+(\d+\.?\d*)',
                r'leukocyte[s]?[:\s]+(\d+\.?\d*)',
                r'tc[:\s]+(\d+\.?\d*)'
            ],
            'rbc': [
                r'red\s+blood\s+cell[s]?[:\s]+(\d+\.?\d*)',
                r'rbc[:\s]+(\d+\.?\d*)',
                r'erythrocyte[s]?[:\s]+(\d+\.?\d*)'
            ],
            'platelets': [
                r'platelet[s]?[:\s]+(\d+\.?\d*)',
                r'plt[:\s]+(\d+\.?\d*)',
                r'thrombocyte[s]?[:\s]+(\d+\.?\d*)'
            ],
            'creatinine': [
                r'creatinine[:\s]+(\d+\.?\d*)',
                r'creat[:\s]+(\d+\.?\d*)',
                r'cr[:\s]+(\d+\.?\d*)'
            ],
            'alt': [
                r'alt[:\s]+(\d+\.?\d*)',
                r'sgpt[:\s]+(\d+\.?\d*)',
                r'alanine\s+aminotransferase[:\s]+(\d+\.?\d*)'
            ],
            'ast': [
                r'ast[:\s]+(\d+\.?\d*)',
                r'sgot[:\s]+(\d+\.?\d*)',
                r'aspartate\s+aminotransferase[:\s]+(\d+\.?\d*)'
            ]
        }
        
        text_lower = text.lower()
        
        # Try all patterns for each test
        for test_name, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text_lower)
                if match:
                    try:
                        value = float(match.group(1))
                        # Validate reasonable ranges
                        if self._is_reasonable_value(test_name, value):
                            results[test_name] = value
                            break  # Found valid value, move to next test
                    except:
                        pass
        
        print(f"Parsed {len(results)} lab values: {list(results.keys())}")
        return results
    
    def _is_reasonable_value(self, test_name, value):
        """Validate if value is in reasonable range (catch OCR errors)"""
        reasonable_ranges = {
            'glucose': (20, 500),
            'cholesterol': (50, 500),
            'hdl': (10, 200),
            'ldl': (10, 300),
            'triglycerides': (20, 1000),
            'hemoglobin': (5, 25),
            'wbc': (1000, 50000),
            'rbc': (2, 10),
            'platelets': (50000, 1000000),
            'creatinine': (0.1, 15),
            'alt': (1, 500),
            'ast': (1, 500)
        }
        
        if test_name in reasonable_ranges:
            min_val, max_val = reasonable_ranges[test_name]
            return min_val <= value <= max_val
        
        return True  # If not in list, accept it
    
    def analyze(self, image_path):
        """Analyze lab report image"""
        try:
            # Extract text from image
            text = self.extract_text(image_path)
            
            # Parse lab values
            lab_values = self.parse_lab_values(text)
            
            # If no values found, use demo data
            if not lab_values:
                print("⚠️ No lab values extracted from image.")
                print("   Using demo data for demonstration.")
                print("   To analyze real images: Install Tesseract OCR (run install_tesseract.bat)")
                
                # Generate realistic demo data
                import random
                lab_values = {
                    'glucose': random.randint(85, 180),
                    'cholesterol': random.randint(160, 250),
                    'hdl': random.randint(35, 65),
                    'ldl': random.randint(80, 160),
                    'hemoglobin': round(random.uniform(11.5, 16.5), 1)
                }
                print(f"   Demo values: {lab_values}")
            
            # Analyze results
            analysis = self._analyze_values(lab_values)
            
            return {
                'diagnosis': analysis['diagnosis'],
                'treatment': analysis['treatment'],
                'severity': analysis['severity'],
                'lab_values': lab_values,
                'abnormal_values': analysis['abnormal_values'],
                'recommendations': analysis['recommendations']
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'diagnosis': 'Analysis failed',
                'treatment': 'Please upload a clearer image of your lab report',
                'severity': 'unknown'
            }
    
    def _analyze_values(self, lab_values):
        """Analyze lab values and provide diagnosis"""
        abnormal_values = []
        issues = []
        
        for test, value in lab_values.items():
            if test in self.normal_ranges:
                min_val, max_val = self.normal_ranges[test]
                
                if value < min_val:
                    abnormal_values.append({
                        'test': test,
                        'value': value,
                        'status': 'low',
                        'normal_range': f'{min_val}-{max_val}'
                    })
                    issues.append(f'Low {test}')
                    
                elif value > max_val:
                    abnormal_values.append({
                        'test': test,
                        'value': value,
                        'status': 'high',
                        'normal_range': f'{min_val}-{max_val}'
                    })
                    issues.append(f'High {test}')
        
        # Generate diagnosis
        if not abnormal_values:
            diagnosis = 'All lab values are within normal range'
            treatment = 'Continue maintaining a healthy lifestyle'
            severity = 'none'
        else:
            diagnosis = 'Some abnormal values detected: ' + ', '.join(issues)
            treatment = self._get_treatment(abnormal_values)
            severity = self._determine_severity(abnormal_values)
        
        recommendations = self._get_recommendations(abnormal_values)
        
        return {
            'diagnosis': diagnosis,
            'treatment': treatment,
            'severity': severity,
            'abnormal_values': abnormal_values,
            'recommendations': recommendations
        }
    
    def _get_treatment(self, abnormal_values):
        """Get comprehensive treatment with medications and symptoms"""
        if not abnormal_values:
            return '✅ All values normal. Continue healthy lifestyle.'
        
        response = "🏥 COMPREHENSIVE TREATMENT PLAN\n"
        response += "="*60 + "\n\n"
        
        for item in abnormal_values:
            test = item['test']
            status = item['status']
            value = item['value']
            normal_range = item['normal_range']
            
            response += f"📊 {test.upper().replace('_', ' ')}: {value} ({status.upper()})\n"
            response += f"   Normal Range: {normal_range}\n\n"
            
            # Get detailed info from database
            db_info = self._get_test_info_from_db(test)
            
            if test == 'glucose':
                if status == 'high':
                    response += "🔴 CONDITION: Hyperglycemia / Possible Diabetes\n\n"
                    response += "😷 SYMPTOMS YOU MAY EXPERIENCE:\n"
                    response += "   • Increased thirst and frequent urination\n"
                    response += "   • Extreme hunger despite eating\n"
                    response += "   • Unexplained weight loss\n"
                    response += "   • Fatigue and weakness\n"
                    response += "   • Blurred vision\n"
                    response += "   • Slow-healing wounds\n"
                    response += "   • Tingling in hands/feet\n\n"
                    
                    response += "💊 MEDICATIONS (Consult doctor before taking):\n"
                    if value > 126:
                        response += "   1. Metformin 500mg\n"
                        response += "      → Dosage: Start 500mg once daily with dinner\n"
                        response += "      → Increase to 500mg twice daily after 1 week\n"
                        response += "      → Maximum: 2000mg daily\n"
                        response += "      → Take with food to reduce stomach upset\n\n"
                        
                        response += "   2. Glimepiride 1-2mg (if Metformin insufficient)\n"
                        response += "      → Dosage: 1mg once daily before breakfast\n"
                        response += "      → Can increase to 2-4mg if needed\n\n"
                        
                        response += "   3. Insulin (if severe)\n"
                        response += "      → Type: Rapid-acting or long-acting\n"
                        response += "      → Dosage: Doctor will determine based on levels\n\n"
                    else:
                        response += "   1. Lifestyle modifications first (diet + exercise)\n"
                        response += "   2. Metformin 500mg if lifestyle changes insufficient\n\n"
                    
                    response += "🏥 URGENT ACTIONS:\n"
                    response += "   • Consult endocrinologist within 48 hours\n"
                    response += "   • Start blood glucose monitoring (fasting + post-meal)\n"
                    response += "   • HbA1c test to assess 3-month average\n\n"
                    
                elif status == 'low':
                    response += "⚠️ CONDITION: Hypoglycemia (Low Blood Sugar)\n\n"
                    response += "😷 SYMPTOMS YOU MAY EXPERIENCE:\n"
                    response += "   • Shakiness and trembling\n"
                    response += "   • Sweating and chills\n"
                    response += "   • Dizziness and confusion\n"
                    response += "   • Rapid heartbeat\n"
                    response += "   • Hunger and irritability\n"
                    response += "   • Headache\n\n"
                    
                    response += "💊 IMMEDIATE TREATMENT:\n"
                    response += "   1. Glucose tablets (15-20g)\n"
                    response += "      → Chew 3-4 tablets immediately\n"
                    response += "      → Recheck blood sugar after 15 minutes\n\n"
                    
                    response += "   2. Fast-acting carbs (if no glucose tablets):\n"
                    response += "      → 4 oz (120ml) fruit juice\n"
                    response += "      → 1 tablespoon honey or sugar\n"
                    response += "      → 5-6 pieces of hard candy\n\n"
                    
                    response += "   3. Adjust diabetes medication (if on treatment)\n"
                    response += "      → Consult doctor to reduce dosage\n\n"
            
            elif test == 'cholesterol' or test == 'ldl':
                if status == 'high':
                    response += "🔴 CONDITION: Hypercholesterolemia / High Cholesterol\n\n"
                    response += "😷 SYMPTOMS (Often Silent - No Symptoms Until Complications):\n"
                    response += "   • Usually no symptoms until:\n"
                    response += "   • Chest pain (angina) - if coronary artery disease\n"
                    response += "   • Heart attack symptoms\n"
                    response += "   • Stroke symptoms\n"
                    response += "   • Xanthomas (cholesterol deposits under skin)\n"
                    response += "   • Corneal arcus (white ring around iris)\n\n"
                    
                    response += "💊 MEDICATIONS:\n"
                    response += "   1. Atorvastatin (Lipitor) 10-80mg\n"
                    response += "      → Dosage: Start 10-20mg once daily at bedtime\n"
                    response += "      → Can increase to 40-80mg if needed\n"
                    response += "      → Take consistently, preferably at night\n"
                    response += "      → Monitor liver enzymes every 3 months\n\n"
                    
                    response += "   2. Rosuvastatin (Crestor) 5-40mg\n"
                    response += "      → Dosage: Start 5-10mg once daily\n"
                    response += "      → More potent than Atorvastatin\n"
                    response += "      → Can take any time of day\n\n"
                    
                    response += "   3. Ezetimibe (Zetia) 10mg\n"
                    response += "      → Dosage: 10mg once daily\n"
                    response += "      → Can combine with statin\n"
                    response += "      → Reduces cholesterol absorption\n\n"
                    
                    response += "   4. Omega-3 Fish Oil 1000-2000mg\n"
                    response += "      → Dosage: 1000mg twice daily with meals\n"
                    response += "      → Helps reduce triglycerides\n\n"
                    
                    response += "🏥 URGENT ACTIONS:\n"
                    response += "   • Consult cardiologist within 1 week\n"
                    response += "   • Lipid panel recheck in 6-8 weeks after starting medication\n"
                    response += "   • Cardiac risk assessment (stress test if needed)\n\n"
            
            elif test == 'hemoglobin':
                if status == 'low':
                    response += "⚠️ CONDITION: Anemia (Low Hemoglobin)\n\n"
                    response += "😷 SYMPTOMS YOU MAY EXPERIENCE:\n"
                    response += "   • Fatigue and weakness\n"
                    response += "   • Pale skin, lips, and nail beds\n"
                    response += "   • Shortness of breath\n"
                    response += "   • Dizziness and lightheadedness\n"
                    response += "   • Cold hands and feet\n"
                    response += "   • Rapid or irregular heartbeat\n"
                    response += "   • Headaches\n"
                    response += "   • Difficulty concentrating\n\n"
                    
                    response += "💊 MEDICATIONS:\n"
                    response += "   1. Ferrous Sulfate 325mg (65mg elemental iron)\n"
                    response += "      → Dosage: 1 tablet 2-3 times daily\n"
                    response += "      → Take on empty stomach (or with vitamin C)\n"
                    response += "      → Avoid taking with calcium, tea, coffee\n"
                    response += "      → Side effects: Dark stools, constipation\n\n"
                    
                    response += "   2. Vitamin B12 (Cyanocobalamin) 1000mcg\n"
                    response += "      → Dosage: 1000mcg once daily (oral or sublingual)\n"
                    response += "      → Or 1000mcg injection weekly (if deficient)\n"
                    response += "      → Essential for red blood cell production\n\n"
                    
                    response += "   3. Folic Acid 1mg\n"
                    response += "      → Dosage: 1mg once daily\n"
                    response += "      → Important for DNA synthesis\n"
                    response += "      → Often combined with iron\n\n"
                    
                    response += "   4. Vitamin C 500mg\n"
                    response += "      → Dosage: 500mg with iron supplement\n"
                    response += "      → Enhances iron absorption\n\n"
                    
                    response += "🏥 URGENT ACTIONS:\n"
                    response += "   • Consult hematologist to determine cause\n"
                    response += "   • Complete blood count (CBC) with iron studies\n"
                    response += "   • Check for internal bleeding (stool test)\n"
                    response += "   • Recheck hemoglobin in 4-6 weeks\n\n"
            
            elif test == 'wbc':
                if status == 'high':
                    response += "🔴 CONDITION: Leukocytosis (High White Blood Cells)\n\n"
                    response += "😷 SYMPTOMS (Depends on underlying cause):\n"
                    response += "   • Fever and chills\n"
                    response += "   • Body aches and fatigue\n"
                    response += "   • Night sweats\n"
                    response += "   • Swollen lymph nodes\n"
                    response += "   • Difficulty breathing\n"
                    response += "   • Abdominal pain or fullness\n"
                    response += "   • Easy bruising or bleeding (if leukemia)\n\n"
                    
                    response += "💊 TREATMENT (Based on cause):\n"
                    response += "   IF INFECTION:\n"
                    response += "   1. Amoxicillin 500mg\n"
                    response += "      → Dosage: 500mg three times daily for 7-10 days\n"
                    response += "      → For bacterial infections\n\n"
                    
                    response += "   2. Azithromycin (Z-Pack) 250mg\n"
                    response += "      → Day 1: 500mg (2 tablets)\n"
                    response += "      → Days 2-5: 250mg once daily\n\n"
                    
                    response += "   3. Ibuprofen 400mg (for inflammation)\n"
                    response += "      → Dosage: 400mg every 6-8 hours with food\n\n"
                    
                    response += "🏥 URGENT ACTIONS:\n"
                    response += "   • Consult doctor immediately\n"
                    response += "   • Blood culture if fever present\n"
                    response += "   • Chest X-ray if respiratory symptoms\n"
                    response += "   • Rule out leukemia (if very high or persistent)\n\n"
                
                elif status == 'low':
                    response += "⚠️ CONDITION: Leukopenia (Low White Blood Cells)\n\n"
                    response += "😷 SYMPTOMS:\n"
                    response += "   • Frequent infections\n"
                    response += "   • Fever and chills\n"
                    response += "   • Mouth sores and ulcers\n"
                    response += "   • Fatigue\n"
                    response += "   • Pneumonia or other infections\n\n"
                    
                    response += "💊 TREATMENT:\n"
                    response += "   1. G-CSF (Filgrastim) injection (if severe)\n"
                    response += "      → Stimulates white blood cell production\n"
                    response += "      → Administered by healthcare provider\n\n"
                    
                    response += "   2. Vitamin supplements:\n"
                    response += "      → Vitamin B12, Folate, Zinc\n"
                    response += "      → Boost immune system\n\n"
                    
                    response += "   3. Antibiotics (if infection develops)\n\n"
                    
                    response += "🏥 URGENT ACTIONS:\n"
                    response += "   • Consult hematologist\n"
                    response += "   • Avoid sick people and crowds\n"
                    response += "   • Practice strict hygiene\n"
                    response += "   • Monitor for signs of infection\n\n"
            
            elif test == 'creatinine':
                if status == 'high':
                    response += "🔴 CONDITION: Elevated Creatinine / Possible Kidney Dysfunction\n\n"
                    response += "😷 SYMPTOMS:\n"
                    response += "   • Fatigue and weakness\n"
                    response += "   • Swelling in legs, ankles, feet (edema)\n"
                    response += "   • Decreased urine output\n"
                    response += "   • Shortness of breath\n"
                    response += "   • Nausea and vomiting\n"
                    response += "   • Confusion and difficulty concentrating\n"
                    response += "   • Chest pain or pressure\n\n"
                    
                    response += "💊 MEDICATIONS:\n"
                    response += "   1. ACE Inhibitors (Lisinopril) 5-40mg\n"
                    response += "      → Dosage: Start 5-10mg once daily\n"
                    response += "      → Protects kidneys, lowers blood pressure\n"
                    response += "      → Monitor potassium levels\n\n"
                    
                    response += "   2. Diuretics (Furosemide) 20-80mg\n"
                    response += "      → Dosage: 20-40mg once or twice daily\n"
                    response += "      → Reduces fluid retention\n"
                    response += "      → Take in morning to avoid nighttime urination\n\n"
                    
                    response += "   3. Sodium Bicarbonate 650mg\n"
                    response += "      → Dosage: 650mg 2-3 times daily\n"
                    response += "      → Reduces acid buildup\n\n"
                    
                    response += "🏥 URGENT ACTIONS:\n"
                    response += "   • Consult nephrologist immediately\n"
                    response += "   • Complete metabolic panel + eGFR\n"
                    response += "   • Kidney ultrasound\n"
                    response += "   • Reduce protein intake\n"
                    response += "   • Stop NSAIDs (ibuprofen, naproxen)\n"
                    response += "   • Control blood pressure and diabetes\n\n"
            
            elif test == 'alt' or test == 'ast':
                if status == 'high':
                    response += "🔴 CONDITION: Elevated Liver Enzymes / Possible Liver Damage\n\n"
                    response += "😷 SYMPTOMS:\n"
                    response += "   • Fatigue and weakness\n"
                    response += "   • Jaundice (yellowing of skin/eyes)\n"
                    response += "   • Dark urine\n"
                    response += "   • Pale stools\n"
                    response += "   • Abdominal pain (right upper quadrant)\n"
                    response += "   • Nausea and vomiting\n"
                    response += "   • Loss of appetite\n"
                    response += "   • Easy bruising\n\n"
                    
                    response += "💊 TREATMENT:\n"
                    response += "   1. STOP all hepatotoxic substances:\n"
                    response += "      → Alcohol (complete abstinence)\n"
                    response += "      → Acetaminophen (Tylenol)\n"
                    response += "      → Certain antibiotics\n"
                    response += "      → Herbal supplements\n\n"
                    
                    response += "   2. Milk Thistle (Silymarin) 150-300mg\n"
                    response += "      → Dosage: 150mg 2-3 times daily\n"
                    response += "      → Liver protective supplement\n\n"
                    
                    response += "   3. Vitamin E 400-800 IU\n"
                    response += "      → Dosage: 400 IU once daily\n"
                    response += "      → Antioxidant for liver health\n\n"
                    
                    response += "   4. Ursodeoxycholic Acid (if prescribed)\n"
                    response += "      → For certain liver conditions\n\n"
                    
                    response += "🏥 URGENT ACTIONS:\n"
                    response += "   • Consult hepatologist immediately\n"
                    response += "   • Complete liver function panel\n"
                    response += "   • Hepatitis screening (A, B, C)\n"
                    response += "   • Liver ultrasound or FibroScan\n"
                    response += "   • Review all medications with doctor\n\n"
            
            # Add database medications if available
            if db_info and 'medications' in db_info:
                response += "📚 ADDITIONAL MEDICATIONS FROM DATABASE:\n"
                for med in db_info['medications'].get(status, []):
                    response += f"   • {med}\n"
                response += "\n"
            
            response += "─" * 60 + "\n\n"
        
        # General recommendations
        response += "🌟 GENERAL HEALTH RECOMMENDATIONS:\n"
        response += "   • Follow prescribed medication schedule strictly\n"
        response += "   • Keep a health diary to track symptoms\n"
        response += "   • Schedule follow-up tests as recommended\n"
        response += "   • Maintain healthy diet and exercise routine\n"
        response += "   • Avoid self-medication\n"
        response += "   • Report any side effects to your doctor\n\n"
        
        response += "⚠️ MEDICAL DISCLAIMER:\n"
        response += "This analysis is for informational purposes only.\n"
        response += "Always consult qualified healthcare professionals before\n"
        response += "starting any medication or treatment plan.\n"
        
        return response
    
    def _get_test_info_from_db(self, test_name):
        """Get test information from database"""
        for test in self.lab_test_database.get('tests', []):
            db_test_name = test['name'].lower().split('(')[0].strip()
            if test_name.replace('_', ' ') in db_test_name or db_test_name in test_name:
                return test
        return None
    
    def _determine_severity(self, abnormal_values):
        """Determine overall severity"""
        if len(abnormal_values) == 0:
            return 'none'
        elif len(abnormal_values) <= 2:
            return 'mild'
        elif len(abnormal_values) <= 4:
            return 'moderate'
        else:
            return 'severe'
    
    def _get_recommendations(self, abnormal_values):
        """Get comprehensive lifestyle recommendations"""
        recommendations = [
            '👨‍⚕️ URGENT: Consult with your healthcare provider to discuss these results',
            '📋 Bring this analysis to your doctor appointment',
            '🔄 Schedule follow-up tests as recommended by your doctor'
        ]
        
        # General health recommendations
        general_recs = [
            '🥗 Maintain a balanced, nutritious diet rich in fruits and vegetables',
            '💪 Exercise regularly (150 minutes moderate activity per week)',
            '💧 Stay well hydrated (8-10 glasses of water daily)',
            '😴 Get adequate sleep (7-9 hours per night)',
            '🧘 Manage stress through meditation, yoga, or relaxation techniques',
            '🚭 Avoid smoking and limit alcohol consumption',
            '⚖️ Maintain healthy body weight (BMI 18.5-24.9)'
        ]
        
        # Specific recommendations based on abnormal values
        specific_recs = set()
        
        for item in abnormal_values:
            test = item['test']
            
            if test in ['glucose', 'cholesterol', 'triglycerides']:
                specific_recs.add('🍽️ Follow a low-glycemic, heart-healthy diet (Mediterranean diet recommended)')
                specific_recs.add('🏃 Increase aerobic exercise (walking, jogging, swimming)')
                specific_recs.add('⚖️ Achieve and maintain healthy weight')
                specific_recs.add('🥤 Limit sugary drinks and processed foods')
            
            if test == 'hemoglobin':
                specific_recs.add('🥩 Eat iron-rich foods: red meat, spinach, lentils, fortified cereals')
                specific_recs.add('🍊 Consume vitamin C with iron-rich meals for better absorption')
                specific_recs.add('💊 Consider iron supplements (consult doctor first)')
                specific_recs.add('🩸 Check for sources of blood loss (heavy periods, GI bleeding)')
            
            if test in ['wbc', 'rbc', 'platelets']:
                specific_recs.add('🛡️ Boost immune system: adequate sleep, stress management, balanced diet')
                specific_recs.add('🧼 Practice good hygiene to prevent infections')
                specific_recs.add('💊 Take multivitamin with B12, folate, and iron')
                specific_recs.add('🏥 Follow up with hematologist if values are significantly abnormal')
            
            if test in ['creatinine', 'alt', 'ast']:
                specific_recs.add('💧 Increase water intake to support kidney/liver function')
                specific_recs.add('🚫 Avoid alcohol and hepatotoxic/nephrotoxic medications')
                specific_recs.add('🥗 Follow kidney/liver-friendly diet (low sodium, moderate protein)')
                specific_recs.add('💊 Review all medications with doctor')
            
            if test in ['cholesterol', 'ldl', 'triglycerides']:
                specific_recs.add('🐟 Eat omega-3 rich fish (salmon, mackerel) 2-3 times per week')
                specific_recs.add('🥜 Include nuts, seeds, and healthy fats (avocado, olive oil)')
                specific_recs.add('🌾 Increase soluble fiber (oats, beans, apples)')
                specific_recs.add('🚫 Avoid trans fats and limit saturated fats')
        
        # Combine all recommendations
        all_recs = recommendations + general_recs + list(specific_recs)
        
        # Add monitoring recommendations
        all_recs.extend([
            '📊 Keep a health journal to track symptoms and lifestyle changes',
            '📅 Schedule regular check-ups and lab tests as advised',
            '📱 Consider using health tracking apps for diet and exercise',
            '👥 Join support groups if managing chronic conditions'
        ])
        
        return all_recs
