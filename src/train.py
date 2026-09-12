import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# 1. Configuration
DATA_DIR = "data/dataset"
MODEL_DIR = "models"
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 2. Data Preprocessing & Loading
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
val_data = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

print(f"[*] Loaded {len(train_data)} training samples, {len(val_data)} validation samples.")
print(f"[*] Detected Classes: {train_data.classes}")


# 3. Model Architecture (Transfer Learning)
def build_model(num_classes=4):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    # Freeze feature extraction layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier head for our 4 classes
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


model = build_model(num_classes=len(train_data.classes)).to(DEVICE)
print(f"[*] Model initialized on device: {DEVICE}")

# 4. Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)


# 5. Training Loop
def train_and_evaluate():
    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_data)
        # Using .float() instead of .double() for MPS compatibility
        train_acc = (correct_train.float() / len(train_data)).item() * 100

        # --- Validation Phase ---
        model.eval()
        correct_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data)

        val_acc = (correct_val.float() / len(val_data)).item() * 100
        print(
            f"Epoch {epoch}/{NUM_EPOCHS} -> Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%"
        )

    # 6. Export to ONNX for C++ Engine
    os.makedirs(MODEL_DIR, exist_ok=True)
    onnx_path = os.path.join(MODEL_DIR, "model.onnx")
    print("\n[*] Exporting trained model to ONNX format...")

    # Move model to CPU for clean, universal ONNX export
    model_cpu = model.to("cpu")
    model_cpu.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device="cpu")

    torch.onnx.export(
        model_cpu,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"[+] Model saved successfully at: {onnx_path}")


if __name__ == "__main__":
    train_and_evaluate()