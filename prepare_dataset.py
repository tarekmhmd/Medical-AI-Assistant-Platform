"""
Dataset Preparation Script for Medical Assistant Diagnostic Platform
Loads, validates, organizes, and splits datasets for training.

Usage:
    python prepare_dataset.py

This script will:
1. Load dataset from "D:\\project 2\\data"
2. Verify all files exist and are readable
3. Organize dataset into training, validation, and test folders
4. Prepare dataset according to file type (audio, images, CSV)
5. Save all processed files into "D:\\project 2\\processed_data"
6. Provide a summary report of number of files in each split
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import csv

# Try to import optional dependencies
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not installed. Image processing will be limited.")

try:
    import librosa
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    print("Warning: librosa/soundfile not installed. Audio processing will be limited.")

# Configuration
PROJECT_ROOT = Path("D:/project 2")
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "processed_data"

# Split ratios (70% train, 15% validation, 15% test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Image processing settings
IMAGE_SIZE = (224, 224)  # Standard size for CNN models

# Audio processing settings
AUDIO_SAMPLE_RATE = 22050  # Standard sample rate
AUDIO_FORMAT = "wav"


class DatasetProcessor:
    """Main class for processing and organizing datasets."""
    
    def __init__(self):
        self.stats = {
            "audio": {"train": 0, "val": 0, "test": 0, "errors": 0},
            "images": {"train": 0, "val": 0, "test": 0, "errors": 0},
            "csv": {"train": 0, "val": 0, "test": 0, "errors": 0},
            "json": {"total": 0, "errors": 0}
        }
        self.errors = []
        self.start_time = datetime.now()
    
    def create_directory_structure(self):
        """Create the processed_data directory structure."""
        print("\n" + "="*60)
        print("CREATING DIRECTORY STRUCTURE")
        print("="*60)
        
        # Main directories
        splits = ["train", "val", "test"]
        data_types = ["audio", "images", "csv"]
        
        for split in splits:
            for data_type in data_types:
                dir_path = PROCESSED_DIR / split / data_type
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"  ✓ Created: {dir_path.relative_to(PROJECT_ROOT)}")
        
        # JSON databases directory
        json_dir = PROCESSED_DIR / "databases"
        json_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {json_dir.relative_to(PROJECT_ROOT)}")
        
        print("\n✓ Directory structure created successfully!")
    
    def verify_file_readable(self, file_path: Path) -> bool:
        """Check if a file exists and is readable."""
        try:
            if not file_path.exists():
                self.errors.append(f"File not found: {file_path}")
                return False
            if not file_path.is_file():
                self.errors.append(f"Not a file: {file_path}")
                return False
            # Try to open the file
            with open(file_path, 'rb') as f:
                f.read(1)
            return True
        except PermissionError:
            self.errors.append(f"Permission denied: {file_path}")
            return False
        except Exception as e:
            self.errors.append(f"Error reading {file_path}: {e}")
            return False
    
    def split_data(self, items: List, seed: int = 42) -> Tuple[List, List, List]:
        """Split data into train, validation, and test sets."""
        random.seed(seed)
        items = items.copy()
        random.shuffle(items)
        
        n = len(items)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)
        
        train = items[:train_end]
        val = items[train_end:val_end]
        test = items[val_end:]
        
        return train, val, test
    
    def process_csv_files(self):
        """Process CSV files - validate columns and types, then split."""
        print("\n" + "="*60)
        print("PROCESSING CSV FILES")
        print("="*60)
        
        csv_files = list(DATA_DIR.rglob("*.csv"))
        
        # Filter out problematic files
        valid_csvs = []
        for csv_file in csv_files:
            if self.verify_file_readable(csv_file):
                valid_csvs.append(csv_file)
                print(f"  ✓ Found: {csv_file.relative_to(DATA_DIR)}")
        
        if not valid_csvs:
            print("  No CSV files found.")
            return
        
        # Process each CSV file
        for csv_file in valid_csvs:
            try:
                print(f"\n  Processing: {csv_file.name}")
                
                # Read CSV
                if HAS_PANDAS:
                    df = pd.read_csv(csv_file)
                    row_count = len(df)
                    columns = list(df.columns)
                    print(f"    Rows: {row_count}, Columns: {len(columns)}")
                    print(f"    Columns: {columns[:5]}{'...' if len(columns) > 5 else ''}")
                else:
                    with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
                        reader = csv.reader(f)
                        columns = next(reader)
                        row_count = sum(1 for _ in reader)
                    print(f"    Rows: {row_count}, Columns: {len(columns)}")
                
                # Split data
                if HAS_PANDAS:
                    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
                    train_end = int(len(df_shuffled) * TRAIN_RATIO)
                    val_end = train_end + int(len(df_shuffled) * VAL_RATIO)
                    
                    df_train = df_shuffled[:train_end]
                    df_val = df_shuffled[train_end:val_end]
                    df_test = df_shuffled[val_end:]
                    
                    # Save splits
                    base_name = csv_file.stem
                    csv_subdir = PROCESSED_DIR / "train" / "csv"
                    
                    df_train.to_csv(csv_subdir / f"{base_name}_train.csv", index=False)
                    df_val.to_csv(PROCESSED_DIR / "val" / "csv" / f"{base_name}_val.csv", index=False)
                    df_test.to_csv(PROCESSED_DIR / "test" / "csv" / f"{base_name}_test.csv", index=False)
                    
                    self.stats["csv"]["train"] += len(df_train)
                    self.stats["csv"]["val"] += len(df_val)
                    self.stats["csv"]["test"] += len(df_test)
                    
                    print(f"    ✓ Split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
                else:
                    # Fallback: copy file to train only
                    shutil.copy(csv_file, PROCESSED_DIR / "train" / "csv" / csv_file.name)
                    self.stats["csv"]["train"] += row_count
                    print(f"    ✓ Copied to train (pandas not available for splitting)")
                
            except Exception as e:
                self.stats["csv"]["errors"] += 1
                self.errors.append(f"Error processing {csv_file}: {e}")
                print(f"    ✗ Error: {e}")
    
    def process_json_files(self):
        """Process JSON database files - validate and copy."""
        print("\n" + "="*60)
        print("PROCESSING JSON DATABASE FILES")
        print("="*60)
        
        json_files = list(DATA_DIR.rglob("*.json"))
        
        for json_file in json_files:
            if not self.verify_file_readable(json_file):
                continue
            
            print(f"  Found: {json_file.relative_to(DATA_DIR)}")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Validate JSON structure
                if isinstance(data, dict):
                    keys = list(data.keys())
                    print(f"    Keys: {keys}")
                elif isinstance(data, list):
                    print(f"    Items: {len(data)}")
                
                # Save to processed directory
                dest = PROCESSED_DIR / "databases" / json_file.name
                shutil.copy(json_file, dest)
                
                self.stats["json"]["total"] += 1
                print(f"    ✓ Copied to: {dest.relative_to(PROJECT_ROOT)}")
                
            except json.JSONDecodeError as e:
                self.stats["json"]["errors"] += 1
                self.errors.append(f"Invalid JSON in {json_file}: {e}")
                print(f"    ✗ Invalid JSON: {e}")
            except Exception as e:
                self.stats["json"]["errors"] += 1
                self.errors.append(f"Error processing {json_file}: {e}")
                print(f"    ✗ Error: {e}")
    
    def process_audio_files(self):
        """Process audio files - organize and split."""
        print("\n" + "="*60)
        print("PROCESSING AUDIO FILES")
        print("="*60)
        
        # Find all audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(list(DATA_DIR.rglob(f"*{ext}")))
        
        if not audio_files:
            print("  No audio files found.")
            return
        
        print(f"  Found {len(audio_files)} audio files")
        
        # Verify files
        valid_audio = []
        for audio_file in audio_files:
            if self.verify_file_readable(audio_file):
                valid_audio.append(audio_file)
        
        print(f"  Valid audio files: {len(valid_audio)}")
        
        # Split data
        train_files, val_files, test_files = self.split_data(valid_audio)
        
        splits = {"train": train_files, "val": val_files, "test": test_files}
        
        for split_name, files in splits.items():
            print(f"\n  Processing {split_name} set ({len(files)} files)...")
            
            for audio_file in files:
                try:
                    dest_dir = PROCESSED_DIR / split_name / "audio"
                    dest_file = dest_dir / audio_file.name
                    
                    # If audio processing is available, normalize
                    if HAS_AUDIO and audio_file.suffix.lower() != '.wav':
                        # Convert to WAV format
                        y, sr = librosa.load(audio_file, sr=AUDIO_SAMPLE_RATE)
                        dest_file = dest_dir / f"{audio_file.stem}.wav"
                        sf.write(dest_file, y, sr)
                        print(f"    ✓ Converted: {audio_file.name} -> {dest_file.name}")
                    else:
                        # Just copy the file
                        shutil.copy(audio_file, dest_file)
                        print(f"    ✓ Copied: {audio_file.name}")
                    
                    self.stats["audio"][split_name] += 1
                    
                except Exception as e:
                    self.stats["audio"]["errors"] += 1
                    self.errors.append(f"Error processing audio {audio_file}: {e}")
                    print(f"    ✗ Error: {e}")
    
    def process_image_files(self):
        """Process image files - resize, normalize, and split."""
        print("\n" + "="*60)
        print("PROCESSING IMAGE FILES")
        print("="*60)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(DATA_DIR.rglob(f"*{ext}")))
            image_files.extend(list(DATA_DIR.rglob(f"*{ext.upper()}")))
        
        if not image_files:
            print("  No image files found.")
            return
        
        print(f"  Found {len(image_files)} image files")
        
        # Verify files
        valid_images = []
        for image_file in image_files:
            if self.verify_file_readable(image_file):
                valid_images.append(image_file)
        
        print(f"  Valid image files: {len(valid_images)}")
        
        # Split data
        train_files, val_files, test_files = self.split_data(valid_images)
        
        splits = {"train": train_files, "val": val_files, "test": test_files}
        
        for split_name, files in splits.items():
            print(f"\n  Processing {split_name} set ({len(files)} files)...")
            
            for image_file in files:
                try:
                    dest_dir = PROCESSED_DIR / split_name / "images"
                    dest_file = dest_dir / image_file.name
                    
                    if HAS_PIL:
                        # Open and process image
                        with Image.open(image_file) as img:
                            # Convert to RGB if necessary
                            if img.mode not in ('RGB', 'L'):
                                img = img.convert('RGB')
                            
                            # Resize to standard size
                            img_resized = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                            
                            # Save as JPEG for consistency
                            if img.mode == 'L':
                                dest_file = dest_dir / f"{image_file.stem}.png"
                                img_resized.save(dest_file, 'PNG')
                            else:
                                dest_file = dest_dir / f"{image_file.stem}.jpg"
                                img_resized.save(dest_file, 'JPEG', quality=95)
                            
                            print(f"    ✓ Processed: {image_file.name} ({IMAGE_SIZE[0]}x{IMAGE_SIZE[1]})")
                    else:
                        # Just copy the file
                        shutil.copy(image_file, dest_file)
                        print(f"    ✓ Copied: {image_file.name}")
                    
                    self.stats["images"][split_name] += 1
                    
                except Exception as e:
                    self.stats["images"]["errors"] += 1
                    self.errors.append(f"Error processing image {image_file}: {e}")
                    print(f"    ✗ Error: {e}")
    
    def generate_report(self):
        """Generate summary report."""
        print("\n" + "="*60)
        print("SUMMARY REPORT")
        print("="*60)
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        report = {
            "timestamp": end_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "data_source": str(DATA_DIR),
            "output_directory": str(PROCESSED_DIR),
            "split_ratios": {
                "train": TRAIN_RATIO,
                "validation": VAL_RATIO,
                "test": TEST_RATIO
            },
            "statistics": self.stats,
            "total_files_processed": sum([
                sum(self.stats["audio"].values()),
                sum(self.stats["images"].values()),
                sum(self.stats["csv"].values()),
                sum(self.stats["json"].values())
            ]),
            "errors": self.errors if self.errors else "None"
        }
        
        # Print summary
        print(f"\nProcessing completed in: {duration}")
        print(f"\nFile Counts by Type and Split:")
        print("-" * 40)
        
        print(f"\n  AUDIO FILES:")
        print(f"    Training:   {self.stats['audio']['train']}")
        print(f"    Validation: {self.stats['audio']['val']}")
        print(f"    Test:       {self.stats['audio']['test']}")
        print(f"    Errors:     {self.stats['audio']['errors']}")
        
        print(f"\n  IMAGE FILES:")
        print(f"    Training:   {self.stats['images']['train']}")
        print(f"    Validation: {self.stats['images']['val']}")
        print(f"    Test:       {self.stats['images']['test']}")
        print(f"    Errors:     {self.stats['images']['errors']}")
        
        print(f"\n  CSV FILES (rows):")
        print(f"    Training:   {self.stats['csv']['train']}")
        print(f"    Validation: {self.stats['csv']['val']}")
        print(f"    Test:       {self.stats['csv']['test']}")
        print(f"    Errors:     {self.stats['csv']['errors']}")
        
        print(f"\n  JSON DATABASES:")
        print(f"    Total:      {self.stats['json']['total']}")
        print(f"    Errors:     {self.stats['json']['errors']}")
        
        total = (self.stats['audio']['train'] + self.stats['audio']['val'] + self.stats['audio']['test'] +
                 self.stats['images']['train'] + self.stats['images']['val'] + self.stats['images']['test'])
        
        print(f"\n  TOTAL FILES: {total}")
        
        # Save report
        report_file = PROCESSED_DIR / "processing_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✓ Report saved to: {report_file.relative_to(PROJECT_ROOT)}")
        
        # Print errors if any
        if self.errors:
            print(f"\n⚠ WARNINGS/ERRORS ({len(self.errors)}):")
            for error in self.errors[:10]:  # Show first 10 errors
                print(f"    - {error}")
            if len(self.errors) > 10:
                print(f"    ... and {len(self.errors) - 10} more")
        
        return report
    
    def run(self):
        """Run the complete data processing pipeline."""
        print("\n" + "="*60)
        print("  MEDICAL ASSISTANT - DATASET PREPARATION")
        print("="*60)
        print(f"  Source: {DATA_DIR}")
        print(f"  Output: {PROCESSED_DIR}")
        print(f"  Split Ratios: Train={TRAIN_RATIO*100}%, Val={VAL_RATIO*100}%, Test={TEST_RATIO*100}%")
        print("="*60)
        
        # Step 1: Create directory structure
        self.create_directory_structure()
        
        # Step 2: Process JSON databases first (these are reference data)
        self.process_json_files()
        
        # Step 3: Process CSV files
        self.process_csv_files()
        
        # Step 4: Process audio files
        self.process_audio_files()
        
        # Step 5: Process image files
        self.process_image_files()
        
        # Step 6: Generate report
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("✓ DATASET PREPARATION COMPLETE!")
        print("="*60)
        
        return report


def main():
    """Main entry point."""
    processor = DatasetProcessor()
    report = processor.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
