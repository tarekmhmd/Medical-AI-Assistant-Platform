# check_data_folders.py
import os
import json

data_root = r"D:\project 2\data"

modalities = ["skin", "sound", "lab"]
report = {}

for mod in modalities:
    report[mod] = {}
    for split in ["train", "val", "test"]:
        folder_path = os.path.join(data_root, split, mod)
        if os.path.exists(folder_path):
            files = os.listdir(folder_path)
            report[mod][split] = {
                "path": folder_path,
                "num_files": len(files),
                "files": files[:5] if len(files) > 5 else files
            }
        else:
            report[mod][split] = {
                "path": folder_path,
                "num_files": 0,
                "files": []
            }

# Save report
report_path = "data_check_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=4)

print(f"✅ Data check complete! Report saved: {report_path}")

# Print to screen
for mod, splits in report.items():
    print(f"\n=== {mod.upper()} ===")
    for split, info in splits.items():
        print(f"{split}: {info['num_files']} files")
