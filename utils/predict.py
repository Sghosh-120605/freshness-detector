import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------
# Load Model (ResNet18)
# --------------------------
model = models.resnet18(weights=None)
num_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Linear(128, 2)  # 2 classes: Fresh / Rotten
)

model.load_state_dict(torch.load("models/freshness_resnet18.pt", map_location=device))
model.to(device)
model.eval()

# --------------------------
# Image Transform
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --------------------------
# Prediction Function
# --------------------------
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, preds = torch.max(probs, 1)

    # Get numerical confidence %
    confidence = confidence.item() * 100

    # Get class (0 = Fresh, 1 = Rotten)
    label = "Fresh" if preds.item() == 0 else "Rotten"

    # --------------------------
    # Calibrated freshness logic
    # --------------------------
    if label == "Fresh":
        if confidence >= 90:
            days_remaining = 5
        elif confidence >= 80:
            days_remaining = 4
        elif confidence >= 70:
            days_remaining = 3
        elif confidence >= 60:
            days_remaining = 2
        elif confidence >= 70:
            days_remaining = 2
        elif confidence >= 50:
            days_remaining = 1
        else:
            days_remaining = 0
    else:  # Rotten
        if confidence >= 60:
            days_remaining = 0
        else:
            days_remaining = 1

    # --------------------------
    # Draw info on image
    # --------------------------
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = f"{days_remaining} days left ({confidence:.2f}%)"
    color = "green" if label == "Fresh" else "red"
    draw.text((10, 10), text, fill=color, font=font)

    # Save result
    os.makedirs("static/results", exist_ok=True)
    result_path = os.path.join("static/results", os.path.basename(image_path))
    img.save(result_path)
    print(torch.version.cuda)

    # Only return confidence + days
    return confidence, days_remaining, result_path
