import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np

# -----------------------------
# 1️⃣ Paths
# -----------------------------
paths = {
    "skin_train": "data/train/skin",
    "skin_val": "data/val/skin",
    "sound_train": "data/train/sound",
    "sound_val": "data/val/sound",
    "lab_train": "data/train/lab",
    "lab_val": "data/val/lab",
    "chatbot_train": "data/train/chatbot",
    "checkpoints": "checkpoints"
}
os.makedirs(paths["checkpoints"], exist_ok=True)

# -----------------------------
# 2️⃣ SkinAnalyzer - U-Net
# -----------------------------
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 1, 3, padding=1)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class SkinDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]
        self.data_dir = data_dir
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = torch.randn(3, 128, 128)  # placeholder for real image
        mask = torch.randint(0, 2, (1, 128, 128)).float()  # placeholder
        return img, mask

def train_skin(epochs=2):
    train_dataset = SkinDataset(paths["skin_train"])
    val_dataset = SkinDataset(paths["skin_val"])
    
    if len(train_dataset) == 0:
        print("⚠️ [Skin] No training data found. Skipping...")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    model = UNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        for imgs, masks in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        print(f"[Skin] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), os.path.join(paths["checkpoints"], "skin_model.pth"))
    print("✅ Skin model trained and saved!")

# -----------------------------
# 3️⃣ SoundAnalyzer - CNN on spectrogram (placeholder)
# -----------------------------
class SoundCNN(nn.Module):
    def __init__(self):
        super(SoundCNN, self).__init__()
        self.conv = nn.Conv2d(1, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8*32*32, 2)
    def forward(self, x):
        x = self.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class SoundDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]
        self.data_dir = data_dir
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        spec = torch.randn(1, 32, 32)  # placeholder for spectrogram
        label = torch.randint(0, 2, (1,)).long()
        return spec, label

def train_sound(epochs=2):
    train_dataset = SoundDataset(paths["sound_train"])
    
    if len(train_dataset) == 0:
        print("⚠️ [Sound] No training data found. Skipping...")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataset = SoundDataset(paths["sound_val"])
    val_loader = DataLoader(val_dataset, batch_size=4)

    model = SoundCNN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for specs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
        print(f"[Sound] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), os.path.join(paths["checkpoints"], "sound_model.pth"))
    print("✅ Sound model trained and saved!")

# -----------------------------
# 4️⃣ LabAnalyzer - FFN
# -----------------------------
class LabFFN(nn.Module):
    def __init__(self, input_dim=10):
        super(LabFFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LabDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        self.data_dir = data_dir
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        x = torch.randn(10)  # placeholder features
        y = torch.randint(0, 2, (1,)).long()
        return x, y

def train_lab(epochs=2):
    train_dataset = LabDataset(paths["lab_train"])
    
    if len(train_dataset) == 0:
        print("⚠️ [Lab] No training data found. Skipping...")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataset = LabDataset(paths["lab_val"])
    val_loader = DataLoader(val_dataset, batch_size=4)

    model = LabFFN()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
        print(f"[Lab] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), os.path.join(paths["checkpoints"], "lab_model.pth"))
    print("✅ Lab model trained and saved!")

# -----------------------------
# 5️⃣ Chatbot - NLP skeleton
# -----------------------------
def train_chatbot():
    print("✅ Chatbot NLP skeleton: implement training here")
    # Placeholder, يمكن تضيف transformer / classifier هنا
    with open(os.path.join(paths["checkpoints"], "chatbot_model.pth"), "w") as f:
        f.write("chatbot placeholder")

# -----------------------------
# 6️⃣ Master training
# -----------------------------
def train_all():
    train_skin()
    train_sound()
    train_lab()
    train_chatbot()
    print("✅ All training finished!")

# -----------------------------
# 7️⃣ JSON report
# -----------------------------
report = {
    "skin_model_path": os.path.join(paths["checkpoints"], "skin_model.pth"),
    "sound_model_path": os.path.join(paths["checkpoints"], "sound_model.pth"),
    "lab_model_path": os.path.join(paths["checkpoints"], "lab_model.pth"),
    "chatbot_model_path": os.path.join(paths["checkpoints"], "chatbot_model.pth")
}

with open("training_report.json", "w") as f:
    json.dump(report, f, indent=4)

# -----------------------------
# Run master training
# -----------------------------
if __name__ == "__main__":
    train_all()
    print("✅ Training report saved: training_report.json")
