# check_data_folders_recursive.py
import os
import json

data_root = r"D:\project 2\data"

modalities = ["skin", "sound", "lab"]
report = {}

def get_all_files(folder):
    all_files = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            all_files.append(os.path.join(root, f))
    return all_files

for mod in modalities:
    report[mod] = {}
    mod_path = os.path.join(data_root, mod)
    if os.path.exists(mod_path):
        files = get_all_files(mod_path)
        report[mod]["num_files"] = len(files)
        report[mod]["sample_files"] = [os.path.basename(f) for f in files[:10]]
    else:
        report[mod]["num_files"] = 0
        report[mod]["sample_files"] = []

report_path = "data_check_recursive_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=4)

print(f"Recursive data check complete! Report saved: {report_path}")
print()
for mod, info in report.items():
    print(f"{mod.upper()}: {info['num_files']} files found")
