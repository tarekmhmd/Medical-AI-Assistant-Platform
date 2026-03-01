"""
ISIC Dataset Extraction and Organization Script
================================================
Extracts ZIP files and organizes them into proper ISIC folder structure.
"""

import os
import zipfile
import shutil
from pathlib import Path

# Base paths
BASE_DIR = Path(r"D:\project 2\data")
ISIC2018_DIR = BASE_DIR / "ISIC2018"
ISIC2019_DIR = BASE_DIR / "ISIC2019"

# ZIP files to extract
ZIP_FILES = {
    # ISIC2018 Task 3 - Training
    "ISIC2018_Task3_Training_Input": {
        "source": BASE_DIR / "ISIC2018_Task3_Training_Input.zip",
        "dest": ISIC2018_DIR / "Training" / "Input"
    },
    "ISIC2018_Task3_Training_GroundTruth": {
        "source": BASE_DIR / "ISIC2018_Task3_Training_GroundTruth.zip",
        "dest": ISIC2018_DIR / "Training" / "GroundTruth"
    },
    # ISIC2018 Task 3 - Validation
    "ISIC2018_Task3_Validation_Input": {
        "source": BASE_DIR / "ISIC2018_Task3_Validation_Input.zip",
        "dest": ISIC2018_DIR / "Validation" / "Input"
    },
    "ISIC2018_Task3_Validation_GroundTruth": {
        "source": BASE_DIR / "ISIC2018_Task3_Validation_GroundTruth.zip",
        "dest": ISIC2018_DIR / "Validation" / "GroundTruth"
    },
    # ISIC2018 Task 3 - Test
    "ISIC2018_Task3_Test_Input": {
        "source": BASE_DIR / "ISIC2018_Task3_Test_Input.zip",
        "dest": ISIC2018_DIR / "Test" / "Input"
    },
    "ISIC2018_Task3_Test_GroundTruth": {
        "source": BASE_DIR / "ISIC2018_Task3_Test_GroundTruth.zip",
        "dest": ISIC2018_DIR / "Test" / "GroundTruth"
    },
}

# Also check task 3 data folder
TASK3_DIR = BASE_DIR / "task 3 data"
TASK3_ZIP_FILES = {
    "ISIC2018_Task3_Training_Input": {
        "source": TASK3_DIR / "ISIC2018_Task3_Training_Input.zip",
        "dest": ISIC2018_DIR / "Training" / "Input"
    },
    "ISIC2018_Task3_Validation_Input": {
        "source": TASK3_DIR / "ISIC2018_Task3_Validation_Input.zip",
        "dest": ISIC2018_DIR / "Validation" / "Input"
    },
    "ISIC2018_Task3_Validation_GroundTruth": {
        "source": TASK3_DIR / "ISIC2018_Task3_Validation_GroundTruth.zip",
        "dest": ISIC2018_DIR / "Validation" / "GroundTruth"
    },
    "ISIC2018_Task3_Test_Input": {
        "source": TASK3_DIR / "ISIC2018_Task3_Test_Input.zip",
        "dest": ISIC2018_DIR / "Test" / "Input"
    },
    "ISIC2018_Task3_Test_GroundTruth": {
        "source": TASK3_DIR / "ISIC2018_Task3_Test_GroundTruth.zip",
        "dest": ISIC2018_DIR / "Test" / "GroundTruth"
    },
}


def extract_zip(zip_path: Path, dest_path: Path) -> int:
    """Extract ZIP file to destination. Returns number of files extracted."""
    if not zip_path.exists():
        print(f"   ⚠️ ZIP not found: {zip_path}")
        return 0
    
    # Check if already extracted
    if dest_path.exists() and any(dest_path.iterdir()):
        print(f"   ✓ Already extracted: {dest_path.name}")
        return -1  # Already exists
    
    # Create destination
    dest_path.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Count files
            file_count = len([f for f in zf.namelist() if not f.endswith('/')])
            
            print(f"   📦 Extracting {zip_path.name} ({file_count} files)...")
            zf.extractall(dest_path)
            
            return file_count
    except Exception as e:
        print(f"   ❌ Error extracting {zip_path.name}: {e}")
        return 0


def count_images_in_folder(folder: Path) -> int:
    """Count image files in a folder recursively."""
    if not folder.exists():
        return 0
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    count = 0
    
    for f in folder.rglob('*'):
        if f.is_file() and f.suffix.lower() in image_extensions:
            count += 1
    
    return count


def main():
    print("=" * 70)
    print("ISIC DATASET EXTRACTION & ORGANIZATION")
    print("=" * 70)
    
    # Step 1: Extract ZIP files from root data folder
    print("\n📦 STEP 1: Extracting ZIP files from data root...")
    extracted_count = 0
    
    for name, info in ZIP_FILES.items():
        source = info["source"]
        dest = info["dest"]
        
        if source.exists():
            result = extract_zip(source, dest)
            if result > 0:
                extracted_count += 1
            print(f"   → {name}: {dest}")
    
    # Step 2: Extract ZIP files from task 3 data folder
    print("\n📦 STEP 2: Extracting ZIP files from 'task 3 data' folder...")
    
    for name, info in TASK3_ZIP_FILES.items():
        source = info["source"]
        dest = info["dest"]
        
        if source.exists():
            result = extract_zip(source, dest)
            if result > 0:
                extracted_count += 1
            print(f"   → {name}: {dest}")
    
    # Step 3: Organize Excel files
    print("\n📊 STEP 3: Organizing Excel files...")
    
    excel_source = BASE_DIR / "archive (3)" / "AHD.xlsx"
    if excel_source.exists():
        # For now, just note it exists
        print(f"   ✓ Found: {excel_source}")
        print(f"   Size: {excel_source.stat().st_size / (1024*1024):.2f} MB")
    
    # Step 4: Verify and summarize
    print("\n" + "=" * 70)
    print("📁 FOLDER STRUCTURE SUMMARY")
    print("=" * 70)
    
    # Check ISIC2018 structure
    if ISIC2018_DIR.exists():
        print(f"\n📂 ISIC2018:")
        
        for split in ["Training", "Validation", "Test"]:
            split_dir = ISIC2018_DIR / split
            if split_dir.exists():
                print(f"\n   {split}:")
                
                input_dir = split_dir / "Input"
                gt_dir = split_dir / "GroundTruth"
                
                if input_dir.exists():
                    img_count = count_images_in_folder(input_dir)
                    print(f"      Input: {img_count} images")
                else:
                    print(f"      Input: Not found")
                
                if gt_dir.exists():
                    gt_count = count_images_in_folder(gt_dir)
                    csv_count = len(list(gt_dir.glob("*.csv")))
                    print(f"      GroundTruth: {gt_count} images, {csv_count} CSV files")
                else:
                    print(f"      GroundTruth: Not found")
    
    # Check existing ISIC2016
    isic2016_dir = BASE_DIR / "ISIC2016_Task1"
    if isic2016_dir.exists():
        print(f"\n📂 ISIC2016_Task1 (existing):")
        for subdir in ["train_images", "train_masks", "test_images", "test_masks"]:
            sd = isic2016_dir / subdir
            if sd.exists():
                count = count_images_in_folder(sd)
                print(f"      {subdir}: {count} images")
    
    # Check HAM10000
    ham_dir = BASE_DIR / "datasets_raw" / "HAM10000"
    if ham_dir.exists():
        count = count_images_in_folder(ham_dir)
        print(f"\n📂 HAM10000: {count} images")
    
    print("\n" + "=" * 70)
    print("✅ ORGANIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()