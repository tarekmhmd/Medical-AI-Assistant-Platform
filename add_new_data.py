import os
import shutil

# -------------------------------
# Helper function to add new files
# -------------------------------
def add_new_files(target_dir, new_files, extensions=None):
    """
    target_dir: target folder (train/val/test)
    new_files: list of new file paths
    extensions: if specified, only add files with these extensions
    """
    os.makedirs(target_dir, exist_ok=True)
    added = 0
    for f in new_files:
        if extensions is None or f.lower().endswith(extensions):
            filename = os.path.basename(f)
            dst = os.path.join(target_dir, filename)
            if not os.path.exists(dst):
                shutil.copy2(f, dst)
                added += 1
    total = len([f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))])
    return added, total

# -------------------------------
# Check for new data sources
# -------------------------------
data_root = r"D:\project 2\data"

# Find all potential new data folders
new_data_sources = []
for item in os.listdir(data_root):
    item_path = os.path.join(data_root, item)
    if os.path.isdir(item_path) and item not in ["skin", "sound", "lab", "ISIC2016_Task1", "processed", "train", "val", "test"]:
        new_data_sources.append(item_path)

print("Found potential new data sources:")
for src in new_data_sources:
    print(f"  - {src}")

# -------------------------------
# 1. Skin Images - check for new
# -------------------------------
print("\n=== SKIN ===")
skin_target = {
    "train": os.path.join(data_root, "skin", "train"),
    "val": os.path.join(data_root, "skin", "val"),
    "test": os.path.join(data_root, "skin", "test"),
}

# Find new skin files
new_skin_files = []
for src in new_data_sources:
    for root, dirs, files in os.walk(src):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                new_skin_files.append(os.path.join(root, f))

if new_skin_files:
    print(f"Found {len(new_skin_files)} new image files")
else:
    print("No new skin files found")

# -------------------------------
# 2. Sound Files - check for new
# -------------------------------
print("\n=== SOUND ===")
new_sound_files = []
for src in new_data_sources:
    for root, dirs, files in os.walk(src):
        for f in files:
            if f.lower().endswith((".wav", ".mp3", ".flac")):
                new_sound_files.append(os.path.join(root, f))

if new_sound_files:
    print(f"Found {len(new_sound_files)} new audio files")
else:
    print("No new sound files found")

# -------------------------------
# 3. Lab CSVs - check for new
# -------------------------------
print("\n=== LAB ===")
new_lab_files = []
for src in new_data_sources:
    for root, dirs, files in os.walk(src):
        for f in files:
            if f.lower().endswith(".csv"):
                new_lab_files.append(os.path.join(root, f))

if new_lab_files:
    print(f"Found {len(new_lab_files)} new CSV files")
else:
    print("No new lab files found")

# -------------------------------
# Add new files to train folders
# -------------------------------
print("\n=== ADDING NEW FILES ===")

# Add sound files to train
if new_sound_files:
    import random
    random.shuffle(new_sound_files)
    n = len(new_sound_files)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    
    splits = {
        "train": new_sound_files[:n_train],
        "val": new_sound_files[n_train:n_train+n_val],
        "test": new_sound_files[n_train+n_val:]
    }
    
    for split, files in splits.items():
        split_dir = os.path.join(data_root, "sound", split)
        os.makedirs(split_dir, exist_ok=True)
        for f in files:
            dst = os.path.join(split_dir, os.path.basename(f))
            if not os.path.exists(dst):
                shutil.copy2(f, dst)
        print(f"Sound {split}: added {len(files)} files")

# Add lab files to train
if new_lab_files:
    import random
    random.shuffle(new_lab_files)
    n = len(new_lab_files)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    
    splits = {
        "train": new_lab_files[:n_train],
        "val": new_lab_files[n_train:n_train+n_val],
        "test": new_lab_files[n_train+n_val:]
    }
    
    for split, files in splits.items():
        split_dir = os.path.join(data_root, "lab", split)
        os.makedirs(split_dir, exist_ok=True)
        for f in files:
            dst = os.path.join(split_dir, os.path.basename(f))
            if not os.path.exists(dst):
                shutil.copy2(f, dst)
        print(f"Lab {split}: added {len(files)} files")

# -------------------------------
# Summary
# -------------------------------
print("\n=== FINAL SUMMARY ===")
print(f"Sound files added: {len(new_sound_files)}")
print(f"Lab files added: {len(new_lab_files)}")
