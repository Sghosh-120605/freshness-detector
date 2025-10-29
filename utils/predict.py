import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

# ----------------------------
# Load Model and Classes
# ----------------------------
MODEL_PATH = "models/multifruit_freshness_resnet18.pt"
FRUIT_CLASSES_PATH = "models/fruit_classes.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load fruit label encoder
fruit_classes = np.load(FRUIT_CLASSES_PATH, allow_pickle=True)

# Model definition (must match training architecture)
class DualHeadResNet(nn.Module):
    def __init__(self, num_fruits):
        super().__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.base.fc.in_features
        self.base.fc = nn.Identity()
        self.freshness_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
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

model = DualHeadResNet(num_fruits=len(fruit_classes)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ----------------------------
# Image Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----------------------------
# Prediction Function
# ----------------------------
def predict_fruit_freshness(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out_fresh, out_fruit = model(img_tensor)
        probs_fresh = torch.softmax(out_fresh, dim=1)
        probs_fruit = torch.softmax(out_fruit, dim=1)

        freshness_idx = torch.argmax(probs_fresh, dim=1).item()
        fruit_idx = torch.argmax(probs_fruit, dim=1).item()

        confidence = probs_fresh[0][freshness_idx].item() * 100
        fruit_name = fruit_classes[fruit_idx]
        freshness_label = "Fresh" if freshness_idx == 0 else "Spoiled"

        # Adaptive freshness time logic
        if freshness_label == "Fresh":
            if confidence > 85:
                use_within = "Safe for 4-5 days"
            elif confidence > 60:
                use_within = "Use within 2-3 days"
            else:
                use_within = "Mild spoilage — consume soon (1 day)"
        else:
            if confidence <= 60:
                use_within = "Use within 1-2 days"
            else:
                use_within = "Already spoiled — do not consume"

    return {
        "fruit": fruit_name,
        "confidence": round(confidence, 2),
        "use_within": use_within
    }
