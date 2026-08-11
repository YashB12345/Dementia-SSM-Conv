import os
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, classification_report

# =========================================================
# CONFIG
# =========================================================

ORIGINAL_DATASET_DIR = "/Users/yashbanerjee/Pythonprojects/ReadADNIMAC/1Yr/Axis0/temp"
# Structure:
# original_dataset/
#    class1/
#    class2/
#    class3/

SPLIT_DATASET_DIR = "split_dataset"

TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
RANDOM_SEED = 42

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

random.seed(RANDOM_SEED)

# =========================================================
# CREATE TRAIN / VAL / TEST SPLITS
# =========================================================

def create_data_splits():
    """
    Splits dataset into:
    - 80% train
    - 10% validation
    - 10% test

    Ensures:
    - No overlapping images
    - Same class structure
    """

    original_dir = Path(ORIGINAL_DATASET_DIR)
    split_dir = Path(SPLIT_DATASET_DIR)

    # Remove old split dataset if it exists
    if split_dir.exists():
        shutil.rmtree(split_dir)

    # Create split folders
    for split in ["train", "val", "test"]:
        (split_dir / split).mkdir(parents=True, exist_ok=True)

    # Iterate through classes
    for class_name in os.listdir(original_dir):

        class_path = original_dir / class_name

        if not class_path.is_dir():
            continue

        images = list(class_path.glob("*"))

        random.shuffle(images)

        total_images = len(images)

        train_count = int(total_images * TRAIN_SPLIT)
        val_count = int(total_images * VAL_SPLIT)

        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]

        split_map = {
            "train": train_images,
            "val": val_images,
            "test": test_images
        }

        # Create class folders and copy images
        for split_name, split_images in split_map.items():

            split_class_dir = split_dir / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)

            for image_path in split_images:
                shutil.copy(image_path, split_class_dir / image_path.name)

        print(f"{class_name}")
        print(f"  Train: {len(train_images)}")
        print(f"  Val:   {len(val_images)}")
        print(f"  Test:  {len(test_images)}")

# =========================================================
# RUN DATA SPLIT
# =========================================================

create_data_splits()

# =========================================================
# TRANSFORMS
# =========================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# DATASETS
# =========================================================

train_dataset = datasets.ImageFolder(
    root=os.path.join(SPLIT_DATASET_DIR, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(SPLIT_DATASET_DIR, "val"),
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    root=os.path.join(SPLIT_DATASET_DIR, "test"),
    transform=test_transform
)

# =========================================================
# DATALOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =========================================================
# CLASS INFO
# =========================================================

class_names = train_dataset.classes
num_classes = len(class_names)

print("\nClasses:", class_names)

# =========================================================
# MODEL
# =========================================================

model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)

# Replace final classification layer
in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    num_classes
)

model = model.to(DEVICE)

# =========================================================
# LOSS + OPTIMIZER
# =========================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# =========================================================
# TRAIN FUNCTION
# =========================================================

def train_one_epoch(model, loader):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    loss = running_loss / len(loader)
    accuracy = correct / total

    return loss, accuracy

# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate(model, loader):

    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    loss = running_loss / len(loader)
    accuracy = correct / total

    return loss, accuracy

# =========================================================
# TRAINING LOOP
# =========================================================

best_val_acc = 0

for epoch in range(NUM_EPOCHS):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader
    )

    val_loss, val_acc = evaluate(
        model,
        val_loader
    )

    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Acc:  {train_acc:.4f}")

    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Val Acc:    {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "best_efficientnet.pth"
        )

        print("Best model saved!")

# =========================================================
# TESTING
# =========================================================

model.load_state_dict(
    torch.load("best_efficientnet.pth")
)

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# =========================================================
# FINAL TEST RESULTS
# =========================================================

test_acc = accuracy_score(
    all_labels,
    all_preds
)

print("\n======================")
print("FINAL TEST RESULTS")
print("======================")

print(f"Test Accuracy: {test_acc:.4f}")

print("\nClassification Report:\n")

print(classification_report(
    all_labels,
    all_preds,
    target_names=class_names
))
