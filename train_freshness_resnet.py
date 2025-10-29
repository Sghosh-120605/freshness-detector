import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
from torchvision.models import ResNet18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 1. Dataset structure & transforms
# -----------------------------
data_dir = "dataset"  # root folder
batch_size = 32
num_workers = 0  # ⚠️ Important for Windows

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)


# -----------------------------
# 2. Label Parsing (Freshness + Fruit)
# -----------------------------
def parse_labels(folder_name):
    if folder_name.startswith("F_"):
        freshness = "fresh"
        fruit = folder_name.split("_", 1)[1]
    elif folder_name.startswith("S_"):
        freshness = "spoiled"
        fruit = folder_name.split("_", 1)[1]
    else:
        freshness = "unknown"
        fruit = folder_name
    return freshness, fruit


folders = [os.path.basename(x[0]) for x in os.walk(data_dir) if x[0] != data_dir]
freshness_labels, fruit_labels = zip(*[parse_labels(f) for f in folders])

fruit_encoder = LabelEncoder()
fruit_encoder.fit(fruit_labels)

os.makedirs("models", exist_ok=True)
np.save("models/fruit_classes.npy", fruit_encoder.classes_)


# -----------------------------
# 3. Model Definition (Dual-Head)
# -----------------------------
class DualHeadResNet(nn.Module):
    def __init__(self, num_fruits):
        super().__init__()
        # Use new weights argument
        self.base = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = self.base.fc.in_features
        self.base.fc = nn.Identity()

        # Two heads
        self.freshness_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Fresh vs Spoiled
        )
        self.fruit_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_fruits)
        )

    def forward(self, x):
        features = self.base(x)
        fresh_out = self.freshness_head(features)
        fruit_out = self.fruit_head(features)
        return fresh_out, fruit_out


# -----------------------------
# 4. Training (wrapped for Windows)
# -----------------------------
if __name__ == "__main__":
    num_fruits = len(fruit_encoder.classes_)
    model = DualHeadResNet(num_fruits).to(device)

    criterion_fresh = nn.CrossEntropyLoss()
    criterion_fruit = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    EPOCHS = 50

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs = imgs.to(device)
            labels = [dataset.classes[i] for i in labels.tolist()]

            # Extract dual labels
            freshness = [parse_labels(l)[0] for l in labels]
            fruit = [parse_labels(l)[1] for l in labels]

            freshness_targets = torch.tensor([0 if f == "fresh" else 1 for f in freshness]).to(device)
            fruit_targets = torch.tensor(fruit_encoder.transform(fruit)).to(device)

            optimizer.zero_grad()
            out_fresh, out_fruit = model(imgs)

            loss_fresh = criterion_fresh(out_fresh, freshness_targets)
            loss_fruit = criterion_fruit(out_fruit, fruit_targets)
            loss = loss_fresh + 0.5 * loss_fruit  # weighted combination

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"✅ Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f}")

    # -----------------------------
    # 5. Save Model
    # -----------------------------
    torch.save(model.state_dict(), "models/multifruit_freshness_resnet18.pt")
    print("✅ Model saved to models/multifruit_freshness_resnet18.pt")
