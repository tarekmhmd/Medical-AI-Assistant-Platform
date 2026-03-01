# train_all_iflow.py
import os
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# -----------------------------
# Paths
# -----------------------------
paths = {
    "skin_train": r"D:\project 2\data\ISIC2016_Task1\train_images",
    "skin_train_masks": r"D:\project 2\data\ISIC2016_Task1\train_masks",
    "skin_val": r"D:\project 2\data\ISIC2016_Task1\test_images",
    "skin_val_masks": r"D:\project 2\data\ISIC2016_Task1\test_masks",
    "checkpoints": r"D:\project 2\checkpoints"
}
os.makedirs(paths["checkpoints"], exist_ok=True)

# -----------------------------
# 1️⃣ SkinAnalyzer - U-Net
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
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
        self.img_size = (256, 256)
        self.transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.files[idx])
        mask_path = os.path.join(self.masks_dir, self.files[idx].replace(".jpg", "_segmentation.png"))

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale mask

        img = self.transform(img)
        mask = T.Compose([T.Resize(self.img_size), T.ToTensor()])(mask)
        return img, mask

def train_skin(epochs=2):
    train_dataset = SkinDataset(paths["skin_train"], paths["skin_train_masks"])
    val_dataset = SkinDataset(paths["skin_val"], paths["skin_val_masks"])

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
# 2️⃣ SoundAnalyzer - placeholder
# -----------------------------
def train_sound():
    print("⚠️ [Sound] Placeholder training - implement with your audio dataset")
    with open(os.path.join(paths["checkpoints"], "sound_model.pth"), "w") as f:
        f.write("sound placeholder")

# -----------------------------
# 3️⃣ LabAnalyzer - placeholder
# -----------------------------
def train_lab():
    print("⚠️ [Lab] Placeholder training - implement with your lab CSV data")
    with open(os.path.join(paths["checkpoints"], "lab_model.pth"), "w") as f:
        f.write("lab placeholder")

# -----------------------------
# 4️⃣ Chatbot - placeholder
# -----------------------------
def train_chatbot():
    print("⚠️ [Chatbot] Placeholder - implement NLP training here")
    with open(os.path.join(paths["checkpoints"], "chatbot_model.pth"), "w") as f:
        f.write("chatbot placeholder")

# -----------------------------
# 5️⃣ Master training
# -----------------------------
def train_all():
    train_skin()
    train_sound()
    train_lab()
    train_chatbot()
    print("✅ All training finished!")

# -----------------------------
# 6️⃣ Save JSON report
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
# 7️⃣ Run training
# -----------------------------
if __name__ == "__main__":
    train_all()
    print("✅ Training report saved: training_report.json")
