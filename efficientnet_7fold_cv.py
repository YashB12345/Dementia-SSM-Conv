import os
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, ConcatDataset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn.functional as F

# =========================================================
# CONFIG
# =========================================================

ORIGINAL_DATASET_DIR = "/Users/yashbanerjee/Pythonprojects/ReadADNIMAC/1Yr/Axis0/ADNI1YR_N4RBFFN_sagittal_985_pruned"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
RANDOM_SEED = 42
N_SPLITS = 7

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

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
# LOAD FULL DATASET
# =========================================================

full_dataset = datasets.ImageFolder(
    root=ORIGINAL_DATASET_DIR,
    transform=test_transform
)


full_dataset_1yr = datasets.ImageFolder(
    root=ORIGINAL_DATASET_DIR,
    transform=test_transform
)
 
full_dataset_2yr = datasets.ImageFolder(
    root="/Users/yashbanerjee/Pythonprojects/ReadADNIMAC/2Yr/Axis0/ADNI2YR_N4RBFFN_sagittal_pruned_405",
    transform=test_transform
)
 
full_dataset = ConcatDataset([full_dataset_1yr, full_dataset_2yr])

class_names = full_dataset_1yr.classes
num_classes = len(class_names)

print(f"\nClasses: {class_names}")
print(f"Total samples: {len(full_dataset)}\n")

# Extract labels for stratification
labels = []
for idx in range(len(full_dataset)):
    # Get the label from the underlying dataset
    sample, label = full_dataset[idx]
    labels.append(label)
labels = np.array(labels)


# Shuffle dataset indices
shuffle_indices = np.random.permutation(len(full_dataset))
labels_shuffled = labels[shuffle_indices]
 
print(f"Class distribution:")
unique, counts = np.unique(labels_shuffled, return_counts=True)
for cls_idx, count in zip(unique, counts):
    print(f"  {class_names[cls_idx]}: {count} samples\n")
 
# =========================================================
# 7-FOLD STRATIFIED CROSS-VALIDATION SETUP
# =========================================================
 
skfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
 
fold_results = {
    'accuracies': [],
    'losses': [],
    'all_preds': [],
    'all_labels': []
}
 
# =========================================================
# TRAIN FUNCTION
# =========================================================
 
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
 
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
 
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
 
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
 
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
 
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
# GET PREDICTIONS FUNCTION
# =========================================================
 
def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
 
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
 
    return np.array(all_preds), np.array(all_labels)
 
# =========================================================
# CROSS-VALIDATION LOOP
# =========================================================
 
print(f"\nStarting {N_SPLITS}-Fold Stratified Cross-Validation\n")
print("="*70)
 
for fold, (train_idx, test_idx) in enumerate(skfold.split(shuffle_indices, labels_shuffled)):
 
    print(f"\nFOLD {fold + 1}/{N_SPLITS}")
    print("-"*70)
 
    # Map shuffled indices back to original dataset indices
    train_idx_original = shuffle_indices[train_idx]
    test_idx_original = shuffle_indices[test_idx]
 
    # Create subsets
    train_subset = Subset(full_dataset, train_idx_original)
    test_subset = Subset(full_dataset, test_idx_original)
 
    # Split training data into train/val (70/30)
    train_size = int(0.7 * len(train_subset))
    val_size = len(train_subset) - train_size
 
    train_indices = np.random.permutation(len(train_subset))
    train_split_idx = train_indices[:train_size]
    val_split_idx = train_indices[train_size:]
 
    train_dataset = Subset(train_subset, train_split_idx)
    val_dataset = Subset(train_subset, val_split_idx)
 
    # Create DataLoaders with augmented transforms for training
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
        test_subset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
 
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_subset)}")
 
    # Initialize model for this fold
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )
 
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(DEVICE)
 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
 
    # Training loop for this fold
    best_val_acc = 0
    fold_save_path = f"best_efficientnet_fold{fold+1}.pth"
 
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
 
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, DEVICE
        )
 
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}")
 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), fold_save_path)
 
    # Load best model and test
    model.load_state_dict(torch.load(fold_save_path))
    model.eval()
 
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
 
    preds, labels_actual = get_predictions(model, test_loader, DEVICE)
 
    print(f"\nFold {fold + 1} Test Accuracy: {test_acc:.7f}")
 
    fold_results['accuracies'].append(test_acc)
    fold_results['losses'].append(test_loss)
    fold_results['all_preds'].extend(preds)
    fold_results['all_labels'].extend(labels_actual)
 
    # Keep last fold model for Grad-CAM visualization
    if fold < N_SPLITS - 1:
        os.remove(fold_save_path)
    else:
        print(f"Keeping Fold {fold + 1} model for Grad-CAM visualization")
 
print("\n" + "="*70)
 
# =========================================================
# CROSS-VALIDATION RESULTS SUMMARY
# =========================================================
 
accuracies = np.array(fold_results['accuracies'])
losses = np.array(fold_results['losses'])
all_preds = np.array(fold_results['all_preds'])
all_labels = np.array(fold_results['all_labels'])
 
print("\nTest Accuracy by Fold:")
for i, acc in enumerate(fold_results['accuracies']):
    print(f"  Fold {i+1}: {acc:.7f}")
 
print("\n" + "="*70)
print("7-FOLD STRATIFIED CROSS-VALIDATION FINAL RESULTS")
print("="*70)
 
print(f"\nMean Test Accuracy:  {accuracies.mean():.7f}")
print(f"Std Dev:             {accuracies.std():.7f}")
print(f"\nAccuracy Range: [{accuracies.min():.7f}, {accuracies.max():.7f}]")
 
print("\n" + "="*70)
print("AGGREGATE CLASSIFICATION REPORT (All Folds)")
print("="*70 + "\n")
 
print(classification_report(
    all_labels,
    all_preds,
    target_names=class_names
))
 
# =========================================================
# AGGREGATE CONFUSION MATRIX
# =========================================================
 
cm_array = confusion_matrix(all_labels, all_preds)
 
print("\nAggregate Confusion Matrix (All Folds):\n")
print(cm_array)
 
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_array,
    display_labels=class_names
)
 
fig_cm, ax_cm = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax_cm, cmap="Blues", colorbar=True, values_format="d")
ax_cm.set_title("EfficientNet B0 - 7-Fold Cross-Validation Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_7fold_efficientnet.png", dpi=200, bbox_inches="tight")
plt.show()
print("\nSaved confusion_matrix_7fold_efficientnet.png")
 
# =========================================================
# GRAD-CAM CLASS
# =========================================================
 
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
        output = self.model(input_tensor)
        score = output[0, target_class]
        score.backward(retain_graph=True)
 
        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))
 
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
 
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()
 
 
def unnormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img
 
 
def overlay_cam_on_image(rgb_image, cam, alpha=0.4):
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
                            target_layer, true_label=None, pred_label=None,
                            save_path=None):
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
 
    suptitle = ""
    if true_label is not None:
        suptitle += f"True: {class_names[true_label]}  "
    if pred_label is not None:
        suptitle += f"Pred: {class_names[pred_label]}"
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
 
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
 
 
# =========================================================
# GRAD-CAM VISUALIZATION FROM LAST FOLD
# =========================================================
 
print("\n" + "="*70)
print("GENERATING GRAD-CAM VISUALIZATIONS (from Fold 7 Best Model)")
print("="*70)
 
model_last = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
in_features = model_last.classifier[1].in_features
model_last.classifier[1] = nn.Linear(in_features, num_classes)
model_last = model_last.to(DEVICE)
 
# Note: Reload best model from last fold
model_last.load_state_dict(torch.load("best_efficientnet_fold7.pth"))
model_last.eval()
target_layer = model_last.features[-1]
 
# Create new test loader with augmented data for visualization
test_dataset_aug = datasets.ImageFolder(
    root=ORIGINAL_DATASET_DIR,
    transform=test_transform
)
 
test_loader_aug = DataLoader(test_dataset_aug, batch_size=1, shuffle=False)
 
shown_classes = set()
num_gradcam_examples = num_classes
 
for images, labels in test_loader_aug:
    for i in range(images.size(0)):
        label = labels[i].item()
        if label in shown_classes:
            continue
 
        input_tensor = images[i:i+1].clone().to(DEVICE)
        input_tensor.requires_grad_(False)
 
        with torch.enable_grad():
            output = model_last(input_tensor)
            pred_label = output.argmax(dim=1).item()
 
        rgb_image = unnormalize(images[i])
 
        save_path = f"gradcam_7fold_{class_names[label]}_example.png"
        plot_per_class_gradcam(
            model_last, input_tensor, rgb_image, class_names,
            target_layer=target_layer,
            true_label=label, pred_label=pred_label,
            save_path=save_path
        )
        print(f"Saved {save_path}")
 
        shown_classes.add(label)
 
    if len(shown_classes) == num_gradcam_examples:
        break
 
print("\n" + "="*70)
print("Finished 7-Fold Stratified Cross-Validation Analysis")
print("="*70)