import os
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn.functional as F

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

# Name used to label every CNN-generated artifact (confusion matrix,
# GradCAM images, comparison chart entry) so filenames make clear
# which model produced them.
CNN_MODEL_NAME = "efficientnet_b0"

# k values tried for KNN; best one is picked using the validation set
KNN_NEIGHBOR_CANDIDATES = [3, 5, 7, 9, 11, 15]

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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

# A second copy of the training set, but with the *non-augmented*
# eval-style transform. Used only for feature extraction so the
# classical ML models train on clean (un-jittered) embeddings.
train_dataset_eval = datasets.ImageFolder(
    root=os.path.join(SPLIT_DATASET_DIR, "train"),
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

train_loader_eval = DataLoader(
    train_dataset_eval,
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
best_model_path = f"best_{CNN_MODEL_NAME}.pth"

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
            best_model_path
        )

        print("Best model saved!")

# =========================================================
# TESTING (CNN)
# =========================================================

model.load_state_dict(
    torch.load(best_model_path)
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
# RESULTS HELPERS (shared by CNN + classical models)
# =========================================================

# Collects {model_name: test_accuracy} so a final comparison
# chart can be produced once every model has been evaluated.
model_accuracies = {}


def report_and_confusion_matrix(model_name, y_true, y_pred, class_names):
    """
    Prints accuracy + classification report and saves a confusion
    matrix image named after the model that produced the predictions,
    e.g. confusion_matrix_efficientnet_b0.png, confusion_matrix_knn.png
    """

    acc = accuracy_score(y_true, y_pred)
    model_accuracies[model_name] = acc

    print(f"\n======================")
    print(f"{model_name.upper()} - TEST RESULTS")
    print(f"======================")
    print(f"Test Accuracy: {acc:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm_array = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n")
    print(cm_array)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_array,
        display_labels=class_names
    )

    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=True, values_format="d")
    ax_cm.set_title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()

    save_path = f"confusion_matrix_{model_name}.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig_cm)
    print(f"Saved {save_path}")

    return acc


# =========================================================
# FINAL TEST RESULTS - CNN (EfficientNet)
# =========================================================

report_and_confusion_matrix(CNN_MODEL_NAME, all_labels, all_preds, class_names)

# =========================================================
# FEATURE EXTRACTION FOR CLASSICAL ML MODELS
# =========================================================
# KNN, SVM, Naive Bayes and Random Forest can't consume raw pixels
# usefully, so instead they train on the 1280-dim embeddings produced
# by the trained EfficientNet backbone (everything up to, but not
# including, the final classification layer). This reuses the CNN as
# a fixed feature extractor.

def extract_features(model, loader):
    """
    Runs images through model.features + model.avgpool (i.e. the
    EfficientNet backbone before its classification head) and
    returns flattened embeddings + labels as numpy arrays.
    """

    model.eval()

    features_list = []
    labels_list = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            x = model.features(images)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)

            features_list.append(x.cpu().numpy())
            labels_list.append(labels.numpy())

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return features, labels


print("\nExtracting CNN embeddings for classical ML models...")

X_train, y_train = extract_features(model, train_loader_eval)
X_val, y_val = extract_features(model, val_loader)
X_test, y_test = extract_features(model, test_loader)

print(f"Train embeddings: {X_train.shape}")
print(f"Val embeddings:   {X_val.shape}")
print(f"Test embeddings:  {X_test.shape}")

# Standardize features (helps KNN/SVM; harmless for RF/NB)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# KNN
# =========================================================
# Pick the number of neighbors using validation accuracy, then
# refit on train and evaluate once on test.

print("\nTuning KNN over k =", KNN_NEIGHBOR_CANDIDATES)

best_k = KNN_NEIGHBOR_CANDIDATES[0]
best_k_acc = -1

for k in KNN_NEIGHBOR_CANDIDATES:
    knn_candidate = KNeighborsClassifier(n_neighbors=k)
    knn_candidate.fit(X_train_scaled, y_train)
    val_preds = knn_candidate.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"  k={k}: val accuracy = {val_acc:.4f}")

    if val_acc > best_k_acc:
        best_k_acc = val_acc
        best_k = k

print(f"Best k: {best_k} (val accuracy = {best_k_acc:.4f})")

knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train_scaled, y_train)
knn_preds = knn_model.predict(X_test_scaled)

report_and_confusion_matrix("knn", y_test, knn_preds, class_names)

# =========================================================
# SVM
# =========================================================

svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_SEED)
svm_model.fit(X_train_scaled, y_train)
svm_preds = svm_model.predict(X_test_scaled)

report_and_confusion_matrix("svm", y_test, svm_preds, class_names)

# =========================================================
# NAIVE BAYES
# =========================================================

nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
nb_preds = nb_model.predict(X_test_scaled)

report_and_confusion_matrix("naive_bayes", y_test, nb_preds, class_names)

# =========================================================
# RANDOM FOREST
# =========================================================

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)

report_and_confusion_matrix("random_forest", y_test, rf_preds, class_names)

# =========================================================
# MODEL COMPARISON CHART
# =========================================================

print("\n======================")
print("MODEL COMPARISON (Test Accuracy)")
print("======================")
for name, acc in model_accuracies.items():
    print(f"  {name}: {acc:.4f}")

fig_cmp, ax_cmp = plt.subplots(figsize=(8, 5))
names = list(model_accuracies.keys())
accs = [model_accuracies[n] for n in names]
bars = ax_cmp.bar(names, accs, color="steelblue")
ax_cmp.set_ylim(0, 1)
ax_cmp.set_ylabel("Test Accuracy")
ax_cmp.set_title("Model Comparison - Test Accuracy")
ax_cmp.tick_params(axis="x", rotation=20)

for bar, acc in zip(bars, accs):
    ax_cmp.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{acc:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig_cmp)
print("Saved model_comparison.png")

# =========================================================
# GRAD-CAM (EfficientNet CNN only)
# =========================================================
# Grad-CAM is a gradient-based explanation technique that requires a
# differentiable convolutional network, so it only applies to the
# EfficientNet model above -- KNN, SVM, Naive Bayes and Random Forest
# operate on fixed embeddings and have no equivalent spatial map to
# visualize. Every file this section produces is tagged with
# CNN_MODEL_NAME so it's unambiguous which model generated it, e.g.
# gradcam_efficientnet_b0_<class>_example.png
#
# Target layer: model.features[-1] is the last conv block in
# EfficientNet B0 (1x1 conv -> 1280 channels), producing a
# [B, 1280, 7, 7] spatial feature map for 224x224 input -- this
# is the last tensor before AdaptiveAvgPool2d collapses it.

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)          # (1, num_classes)
        score = output[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]               # (C, H, W)
        activations = self.activations[0]            # (C, H, W)

        weights = gradients.mean(dim=(1, 2))          # (C,)

        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


def unnormalize(img_tensor):
    """Reverse ImageNet normalization for display. img_tensor: (C, H, W)."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def overlay_cam_on_image(rgb_image, cam, alpha=0.4):
    """rgb_image: (H, W, 3) in [0,1]. cam: (h, w) in [0,1], any spatial size."""
    cam_resized = np.array(
        F.interpolate(
            torch.tensor(cam)[None, None, ...],
            size=rgb_image.shape[:2],
            mode="bilinear",
            align_corners=False
        )[0, 0]
    )
    heatmap = cm.jet(cam_resized)[..., :3]
    overlay = (1 - alpha) * rgb_image + alpha * heatmap
    return np.clip(overlay, 0, 1)


def plot_per_class_gradcam(model, input_tensor, rgb_image, class_names,
                            target_layer, model_name, true_label=None,
                            pred_label=None, save_path=None):
    gradcam = GradCAM(model, target_layer)

    fig, axes = plt.subplots(1, len(class_names) + 1, figsize=(4 * (len(class_names) + 1), 4))

    axes[0].imshow(rgb_image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, cls_name in enumerate(class_names):
        cam = gradcam.generate(input_tensor, target_class=i)
        overlay = overlay_cam_on_image(rgb_image, cam)

        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(f"{cls_name}-targeted CAM")
        axes[i + 1].axis("off")

    suptitle = f"Model: {model_name}  "
    if true_label is not None:
        suptitle += f"True: {class_names[true_label]}  "
    if pred_label is not None:
        suptitle += f"Pred: {class_names[pred_label]}"
    fig.suptitle(suptitle, fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# Grad-CAM needs gradients, so re-enable them even though we're
# past the torch.no_grad() eval blocks above.
model.eval()
target_layer = model.features[-1]

# Pick one representative test image per class (first match found)
shown_classes = set()
num_gradcam_examples = num_classes  # one per class

for images, labels in test_loader:
    for i in range(images.size(0)):
        label = labels[i].item()
        if label in shown_classes:
            continue

        input_tensor = images[i:i+1].clone().to(DEVICE)
        input_tensor.requires_grad_(False)  # gradients w.r.t. activations only, not input

        with torch.enable_grad():
            output = model(input_tensor)
            pred_label = output.argmax(dim=1).item()

        rgb_image = unnormalize(images[i])

        save_path = f"gradcam_{CNN_MODEL_NAME}_{class_names[label]}_example.png"
        plot_per_class_gradcam(
            model, input_tensor, rgb_image, class_names,
            target_layer=target_layer,
            model_name=CNN_MODEL_NAME,
            true_label=label, pred_label=pred_label,
            save_path=save_path
        )
        print(f"Saved {save_path}")

        shown_classes.add(label)

    if len(shown_classes) == num_gradcam_examples:
        break