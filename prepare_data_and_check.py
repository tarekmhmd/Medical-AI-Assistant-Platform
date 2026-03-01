# prepare_data_and_check.py
import os
import shutil
import json
import random

random.seed(42)  # for reproducibility

data_root = r"D:\project 2\data"
isic_images = r"D:\project 2\data\ISIC2016_Task1\train_images"
isic_masks = r"D:\project 2\data\ISIC2016_Task1\train_masks"

modalities = ["skin", "sound", "lab"]
report = {}

# -----------------------------
# 1️⃣ Prepare skin dataset (split train/val/test)
# -----------------------------
skin_target = {
    "train": os.path.join(data_root, "skin", "train"),
    "val": os.path.join(data_root, "skin", "val"),
    "test": os.path.join(data_root, "skin", "test"),
}

# ensure directories exist
for folder in skin_target.values():
    os.makedirs(folder, exist_ok=True)

# get all ISIC images
if os.path.exists(isic_images):
    all_images = [f for f in os.listdir(isic_images) if f.lower().endswith(".jpg")]
    random.shuffle(all_images)
    n = len(all_images)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train:n_train+n_val],
        "test": all_images[n_train+n_val:]
    }

    for split, files in splits.items():
        for f in files:
            # Copy image
            src_img = os.path.join(isic_images, f)
            dst_img = os.path.join(skin_target[split], f)
            if not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)
            
            # Copy corresponding mask
            mask_name = f.replace(".jpg", "_segmentation.png")
            src_mask = os.path.join(isic_masks, mask_name)
            dst_mask = os.path.join(skin_target[split], mask_name)
            if os.path.exists(src_mask) and not os.path.exists(dst_mask):
                shutil.copy2(src_mask, dst_mask)

    print(f"✅ Skin data split: train={n_train}, val={n_val}, test={n_test}")
else:
    print("⚠️ ISIC2016_Task1 folder not found!")

# -----------------------------
# 2️⃣ Check all data folders recursively
# -----------------------------
def get_all_files(folder):
    all_files = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            all_files.append(os.path.join(root, f))
    return all_files

for mod in modalities:
    mod_path = os.path.join(data_root, mod)
    if os.path.exists(mod_path):
        files = get_all_files(mod_path)
        report[mod] = {
            "num_files": len(files),
            "sample_files": [os.path.basename(f) for f in files[:5]]
        }
    else:
        report[mod] = {
            "num_files": 0,
            "sample_files": []
        }

# -----------------------------
# 3️⃣ Save JSON report
# -----------------------------
report_path = "data_preparation_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=4)

print(f"\n✅ Data preparation & check complete! Report saved: {report_path}")

# -----------------------------
# 4️⃣ Print summary
# -----------------------------
for mod, info in report.items():
    print(f"{mod.upper()}: {info['num_files']} files found")