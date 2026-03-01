import os
import json

# -----------------------------
# 1️⃣ Setup folders
# -----------------------------
folders = [
    "data/train/skin", "data/val/skin", "data/test/skin",
    "data/train/sound", "data/val/sound", "data/test/sound",
    "data/train/lab", "data/val/lab", "data/test/lab",
    "models", "logs", "checkpoints", "notebooks"
]

created_folders = []
existing_folders = []

for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        created_folders.append(folder)
    else:
        existing_folders.append(folder)

# -----------------------------
# 2️⃣ Generate Starter Code for each Analyzer (DL Skeleton)
# -----------------------------
starter_code = {
    "skin_analyzer.py": '''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Simple U-Net Skeleton
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Define layers here

    def forward(self, x):
        return x

class SkinDataset(Dataset):
    def __init__(self, data_dir):
        # Load images & masks
        pass

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None, None

def train():
    dataset = SkinDataset("data/train/skin")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = UNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(5):
        for imgs, masks in dataloader:
            # forward + backward placeholder
            pass
    torch.save(model.state_dict(), "models/skin_model.pth")

if __name__ == "__main__":
    train()
''',
    "sound_analyzer.py": "# Starter code for SoundAnalyzer (CNN or spectrogram-based)\n",
    "lab_analyzer.py": "# Starter code for LabAnalyzer (tabular / OCR ML)\n",
    "chatbot.py": "# Starter code for Chatbot (rule-based / NLP skeleton)\n"
}

created_files = []
skipped_files = []

for filename, code in starter_code.items():
    path = os.path.join("models", filename)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(code)
        created_files.append(path)
    else:
        skipped_files.append(path)

# -----------------------------
# 3️⃣ Generate Master Training Script
# -----------------------------
train_script_path = "train_all.py"
if not os.path.exists(train_script_path):
    with open(train_script_path, "w") as f:
        f.write("""# Master training script
from models.skin_analyzer import train as train_skin

if __name__ == "__main__":
    train_skin()
    print("Training completed!")
""")
    master_script_created = True
else:
    master_script_created = False

# -----------------------------
# 4️⃣ JSON Report
# -----------------------------
report = {
    "created_folders": created_folders,
    "existing_folders": existing_folders,
    "created_files": created_files,
    "skipped_files": skipped_files,
    "master_training_script_created": master_script_created
}

with open("project_full_setup_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("✅ Project full setup completed.")
print("✅ Report generated: project_full_setup_report.json")
print()
print("Folders created:", created_folders)
print("Folders existing:", existing_folders)
print("Files created:", created_files)
print("Files skipped:", skipped_files)
print("Master script created:", master_script_created)
