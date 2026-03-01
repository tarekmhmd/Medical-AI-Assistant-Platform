import os

# Current data folders
data_root = r"D:\project 2\data"

modalities = {
    "Skin": "skin",
    "Sound": "sound", 
    "Lab": "lab"
}

print("=== FINAL DATA INVENTORY ===")
print()
print(f"{'Modality':<10} {'Train':>12} {'Val':>12} {'Test':>12} {'Total':>12}")
print("-" * 60)

total_all = 0
for mod_name, mod_folder in modalities.items():
    train_dir = os.path.join(data_root, mod_folder, "train")
    val_dir = os.path.join(data_root, mod_folder, "val")
    test_dir = os.path.join(data_root, mod_folder, "test")
    
    train_count = len([f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]) if os.path.exists(train_dir) else 0
    val_count = len([f for f in os.listdir(val_dir) if os.path.isfile(os.path.join(val_dir, f))]) if os.path.exists(val_dir) else 0
    test_count = len([f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]) if os.path.exists(test_dir) else 0
    total = train_count + val_count + test_count
    total_all += total
    
    print(f"{mod_name:<10} {train_count:>12} {val_count:>12} {test_count:>12} {total:>12}")

print("-" * 60)
print(f"{'TOTAL':<10} {'':<12} {'':<12} {'':<12} {total_all:>12}")
