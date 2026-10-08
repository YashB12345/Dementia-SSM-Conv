import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import pandas as pd
import random

# =========================================================
# CONFIG
# =========================================================

ORIGINAL_DATASET_DIR = "/Users/yashbanerjee/Pythonprojects/ReadADNIMAC/1Yr/Axis0/ADNI1YR_N4RBFFN_sagittal_985_pruned"
IMAGE_SIZE = 224
BATCH_SIZE = 32
RANDOM_SEED = 42
N_SPLITS = 7

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# =========================================================
# TRANSFORMS
# =========================================================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# LOAD DATASET AND EXTRACT FEATURES
# =========================================================

print("Loading dataset...")
full_dataset = datasets.ImageFolder(
    root=ORIGINAL_DATASET_DIR,
    transform=test_transform
)


full_dataset_1yr = datasets.ImageFolder(
    root=ORIGINAL_DATASET_DIR,
    transform=test_transform
)


 
full_dataset_2yr = datasets.ImageFolder(
    root="/Users/yashbanerjee/Pythonprojects/ReadADNIMAC/2Yr/Axis0/ADNI2YR_N4RBFFN_sagittal_580",
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


# Extract image features (flatten images)
print("Extracting features from images...")
features_list = []
loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)

for images, _ in loader:
    # Flatten images: (B, 3, 224, 224) to (B, 150528)
    flattened = images.view(images.size(0), -1).cpu().numpy()
    features_list.append(flattened)

X = np.concatenate(features_list, axis=0)
y = labels

print(f"Feature shape: {X.shape}")
print(f"Label shape: {y.shape}\n")

# Display class distribution
print("Class distribution:")
unique, counts = np.unique(y, return_counts=True)
for cls_idx, count in zip(unique, counts):
    print(f"  {class_names[cls_idx]}: {count} samples")

# =========================================================
# SHUFFLE DATA FOR STRATIFICATION
# =========================================================

shuffle_indices = np.random.permutation(len(X))
X_shuffled = X[shuffle_indices]
y_shuffled = y[shuffle_indices]

# =========================================================
# DEEP LEARNING MODELS: ResNet34 and DenseNet
# =========================================================
 
print("\n" + "="*80)
print("DEEP LEARNING MODELS: ResNet34 and DenseNet")
print("="*80 + "\n")
 
import torch.nn as nn
import torch.optim as optim
from torchvision import models
 
# Reload dataset with training transforms for deep learning
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
 
# full_dataset_dl = datasets.ImageFolder(
#     root=ORIGINAL_DATASET_DIR,
#     transform=test_transform
# )
 
# Deep learning cross-validation results
dl_results = {
    'ResNet34': {
        'accuracies': [],
        'precisions': [],
        'recalls': [],
        'f1_scores': [],
        'all_preds': [],
        'all_labels': []
    },
    'DenseNet': {
        'accuracies': [],
        'precisions': [],
        'recalls': [],
        'f1_scores': [],
        'all_preds': [],
        'all_labels': []
    }
}
 
DL_NUM_EPOCHS = 15
DL_LEARNING_RATE = 0.001
DL_BATCH_SIZE = 32

# =========================================================
# 7-FOLD STRATIFIED CROSS-VALIDATION
# =========================================================

print("\n" + "="*80)
print("Starting 7-Fold Stratified Cross-Validation")
print("="*80 + "\n")

skfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
 
def train_dl_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    """Train deep learning model and return best validation accuracy"""
    best_val_acc = 0.0
 
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
 
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
 
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
 
        train_acc = correct / total
 
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
 
        val_acc = val_correct / val_total
        best_val_acc = max(best_val_acc, val_acc)
 
    return model
 
def evaluate_dl_model(model, test_loader, device):
    """Evaluate deep learning model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
 
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
 
    return np.array(all_preds), np.array(all_labels)
 
# Deep learning 7-fold cross-validation
print("Training Deep Learning Models (ResNet34 and DenseNet)...\n")
 
for fold, (train_idx, test_idx) in enumerate(skfold.split(shuffle_indices, y_shuffled)):
 
    print(f"FOLD {fold + 1}/{N_SPLITS} (Deep Learning)")
    print("-"*80)
 
    # Get indices for this fold
    train_indices_fold = shuffle_indices[train_idx]
    test_indices_fold = shuffle_indices[test_idx]
 
    # Create datasets
    from torch.utils.data import Subset
    train_subset = Subset(full_dataset, train_indices_fold)
    test_subset = Subset(full_dataset, test_indices_fold)
 
    # Split training into train/val (70/30)
    train_size = int(0.7 * len(train_subset))
    val_size = len(train_subset) - train_size
    train_indices_split = np.random.permutation(len(train_subset))
    train_split_idx = train_indices_split[:train_size]
    val_split_idx = train_indices_split[train_size:]
 
    train_set = Subset(train_subset, train_split_idx)
    val_set = Subset(train_subset, val_split_idx)
 
    # Create DataLoaders
    train_loader_dl = DataLoader(train_set, batch_size=DL_BATCH_SIZE, shuffle=True)
    val_loader_dl = DataLoader(val_set, batch_size=DL_BATCH_SIZE, shuffle=False)
    test_loader_dl = DataLoader(test_subset, batch_size=DL_BATCH_SIZE, shuffle=False)
 
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_subset)}")
 
    # ResNet34
    print("  Training ResNet34...", end=" ", flush=True)
    resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_classes)
    resnet = resnet.to(DEVICE)
 
    criterion_dl = nn.CrossEntropyLoss()
    optimizer_resnet = optim.Adam(resnet.parameters(), lr=DL_LEARNING_RATE)
 
    resnet = train_dl_model(resnet, train_loader_dl, val_loader_dl, criterion_dl, optimizer_resnet, DEVICE, DL_NUM_EPOCHS)
    resnet_preds, resnet_labels = evaluate_dl_model(resnet, test_loader_dl, DEVICE)
 
    resnet_acc = accuracy_score(resnet_labels, resnet_preds)
    resnet_prec = precision_score(resnet_labels, resnet_preds, average='weighted', zero_division=0)
    resnet_rec = recall_score(resnet_labels, resnet_preds, average='weighted', zero_division=0)
    resnet_f1 = f1_score(resnet_labels, resnet_preds, average='weighted', zero_division=0)
 
    dl_results['ResNet34']['accuracies'].append(resnet_acc)
    dl_results['ResNet34']['precisions'].append(resnet_prec)
    dl_results['ResNet34']['recalls'].append(resnet_rec)
    dl_results['ResNet34']['f1_scores'].append(resnet_f1)
    dl_results['ResNet34']['all_preds'].extend(resnet_preds)
    dl_results['ResNet34']['all_labels'].extend(resnet_labels)
 
    print(f"Acc: {resnet_acc:.4f}")
 
    # DenseNet
    print("  Training DenseNet...", end=" ", flush=True)
    densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    num_ftrs = densenet.classifier.in_features
    densenet.classifier = nn.Linear(num_ftrs, num_classes)
    densenet = densenet.to(DEVICE)
 
    optimizer_densenet = optim.Adam(densenet.parameters(), lr=DL_LEARNING_RATE)
 
    densenet = train_dl_model(densenet, train_loader_dl, val_loader_dl, criterion_dl, optimizer_densenet, DEVICE, DL_NUM_EPOCHS)
    densenet_preds, densenet_labels = evaluate_dl_model(densenet, test_loader_dl, DEVICE)
 
    densenet_acc = accuracy_score(densenet_labels, densenet_preds)
    densenet_prec = precision_score(densenet_labels, densenet_preds, average='weighted', zero_division=0)
    densenet_rec = recall_score(densenet_labels, densenet_preds, average='weighted', zero_division=0)
    densenet_f1 = f1_score(densenet_labels, densenet_preds, average='weighted', zero_division=0)
 
    dl_results['DenseNet']['accuracies'].append(densenet_acc)
    dl_results['DenseNet']['precisions'].append(densenet_prec)
    dl_results['DenseNet']['recalls'].append(densenet_rec)
    dl_results['DenseNet']['f1_scores'].append(densenet_f1)
    dl_results['DenseNet']['all_preds'].extend(densenet_preds)
    dl_results['DenseNet']['all_labels'].extend(densenet_labels)
 
    print(f"Acc: {densenet_acc:.4f}\n")
 
# =========================================================
# COMBINED RESULTS: Classical ML + Deep Learning
# =========================================================
 
print("\n" + "="*80)
print("COMBINED RESULTS: Classical ML vs Deep Learning Models")
print("="*80 + "\n")
 
# Merge results
all_results = {**cv_results, **dl_results}
 
combined_results_data = []
 
for model_name in all_results.keys():
    accuracies = np.array(all_results[model_name]['accuracies'])
    precisions = np.array(all_results[model_name]['precisions'])
    recalls = np.array(all_results[model_name]['recalls'])
    f1_scores = np.array(all_results[model_name]['f1_scores'])
 
    combined_results_data.append({
        'Model': model_name,
        'Mean Accuracy': f"{accuracies.mean():.7f}",
        'Std Accuracy': f"{accuracies.std():.7f}",
        'Mean Precision': f"{precisions.mean():.7f}",
        'Mean Recall': f"{recalls.mean():.7f}",
        'Mean F1': f"{f1_scores.mean():.7f}"
    })
 
combined_results_df = pd.DataFrame(combined_results_data)
print(combined_results_df.to_string(index=False))
 
# =========================================================
# COMBINED RANKING
# =========================================================
 
print("\n" + "="*80)
print("MODEL RANKING (Classical ML + Deep Learning)")
print("="*80 + "\n")
 
combined_ranking = sorted(
    [(name, np.array(all_results[name]['accuracies']).mean()) for name in all_results.keys()],
    key=lambda x: x[1],
    reverse=True
)
 
for rank, (name, acc) in enumerate(combined_ranking, 1):
    model_type = "Deep Learning" if name in ['ResNet34', 'DenseNet'] else "Classical ML"
    print(f"{rank}. {name:20} {acc:.7f}  ({model_type})")
 
# =========================================================
# COMBINED CONFUSION MATRICES
# =========================================================
 
print("\n" + "="*80)
print("CONFUSION MATRICES: All Models")
print("="*80 + "\n")
 
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
 
for idx, model_name in enumerate(all_results.keys()):
 
    all_preds = np.array(all_results[model_name]['all_preds'])
    all_labels = np.array(all_results[model_name]['all_labels'])
 
    cm_array = confusion_matrix(all_labels, all_preds)
 
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_array,
        display_labels=class_names
    )
 
    model_type = "DL" if model_name in ['ResNet34', 'DenseNet'] else "ML"
    disp.plot(ax=axes[idx], cmap="Blues", colorbar=True, values_format="d")
    axes[idx].set_title(f"{model_name} ({model_type}) - Confusion Matrix")
 
axes[-1].remove()
 
plt.tight_layout()
plt.savefig("confusion_matrices_combined_all_models.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved confusion_matrices_combined_all_models.png")
 
# =========================================================
# ACCURACY COMPARISON: All Models
# =========================================================
 
fig, ax = plt.subplots(figsize=(12, 7))
 
model_names = list(all_results.keys())
means = [np.array(all_results[m]['accuracies']).mean() for m in model_names]
stds = [np.array(all_results[m]['accuracies']).std() for m in model_names]
 
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
x_pos = np.arange(len(model_names))
ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7, color=colors)
 
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('7-Fold Cross-Validation: Classical ML vs Deep Learning Models', fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation=15, ha='right')
ax.set_ylim([0.65, 0.95])
 
for i, (mean, std) in enumerate(zip(means, stds)):
    ax.text(i, mean + std + 0.01, f'{mean:.4f}', ha='center', va='bottom', fontsize=9)
 
plt.tight_layout()
plt.savefig("accuracy_comparison_all_models.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved accuracy_comparison_all_models.png")


# =========================================================
# MODEL DEFINITIONS
# =========================================================

def get_models():
    """Return dictionary of models to evaluate"""
    return {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    }



# Storage for results
cv_results = {
    model_name: {
        'accuracies': [],
        'precisions': [],
        'recalls': [],
        'f1_scores': [],
        'all_preds': [],
        'all_labels': []
    }
    for model_name in get_models().keys()
}

for fold, (train_idx, test_idx) in enumerate(skfold.split(shuffle_indices, y_shuffled)):

    print(f"FOLD {fold + 1}/{N_SPLITS}")
    print("-"*80)

    # Get train and test data for this fold
    X_train, X_test = X_shuffled[train_idx], X_shuffled[test_idx]
    y_train, y_test = y_shuffled[train_idx], y_shuffled[test_idx]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and evaluate each model
    models = get_models()

    for model_name, model in models.items():

        # Train model
        model.fit(X_train_scaled, y_train)

        # Get predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Store results
        cv_results[model_name]['accuracies'].append(accuracy)
        cv_results[model_name]['precisions'].append(precision)
        cv_results[model_name]['recalls'].append(recall)
        cv_results[model_name]['f1_scores'].append(f1)
        cv_results[model_name]['all_preds'].extend(y_pred)
        cv_results[model_name]['all_labels'].extend(y_test)

        print(f"  {model_name:20} Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")

    print()

# =========================================================
# CROSS-VALIDATION RESULTS SUMMARY
# =========================================================

print("\n" + "="*80)
print("7-FOLD CROSS-VALIDATION RESULTS SUMMARY")
print("="*80 + "\n")

# Create results dataframe
results_data = []

for model_name in cv_results.keys():
    accuracies = np.array(cv_results[model_name]['accuracies'])
    precisions = np.array(cv_results[model_name]['precisions'])
    recalls = np.array(cv_results[model_name]['recalls'])
    f1_scores = np.array(cv_results[model_name]['f1_scores'])

    results_data.append({
        'Model': model_name,
        'Mean Accuracy': f"{accuracies.mean():.7f}",
        'Std Accuracy': f"{accuracies.std():.7f}",
        'Mean Precision': f"{precisions.mean():.7f}",
        'Std Precision': f"{precisions.std():.7f}",
        'Mean Recall': f"{recalls.mean():.7f}",
        'Std Recall': f"{recalls.std():.7f}",
        'Mean F1': f"{f1_scores.mean():.7f}",
        'Std F1': f"{f1_scores.std():.7f}"
    })

results_df = pd.DataFrame(results_data)
print(results_df.to_string(index=False))

# =========================================================
# PER-FOLD ACCURACY COMPARISON
# =========================================================

print("\n" + "="*80)
print("Per-Fold Test Accuracy Comparison")
print("="*80 + "\n")

fold_data = []
for fold_num in range(N_SPLITS):
    fold_dict = {'Fold': fold_num + 1}
    for model_name in cv_results.keys():
        fold_dict[model_name] = f"{cv_results[model_name]['accuracies'][fold_num]:.7f}"
    fold_data.append(fold_dict)

fold_df = pd.DataFrame(fold_data)
print(fold_df.to_string(index=False))

# =========================================================
# DETAILED CLASSIFICATION REPORTS
# =========================================================

print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORTS (Aggregate Across All Folds)")
print("="*80 + "\n")

for model_name in cv_results.keys():
    print(f"\n{model_name}")
    print("-"*80)

    all_preds = np.array(cv_results[model_name]['all_preds'])
    all_labels = np.array(cv_results[model_name]['all_labels'])

    print(classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    ))

# =========================================================
# CONFUSION MATRICES
# =========================================================

print("\n" + "="*80)
print("CONFUSION MATRICES (Aggregate Across All Folds)")
print("="*80 + "\n")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, model_name in enumerate(cv_results.keys()):

    all_preds = np.array(cv_results[model_name]['all_preds'])
    all_labels = np.array(cv_results[model_name]['all_labels'])

    cm_array = confusion_matrix(all_labels, all_preds)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_array,
        display_labels=class_names
    )

    disp.plot(ax=axes[idx], cmap="Blues", colorbar=True, values_format="d")
    axes[idx].set_title(f"{model_name} - Confusion Matrix")

plt.tight_layout()
plt.savefig("confusion_matrices_comparison.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved confusion_matrices_comparison.png")

# =========================================================
# ACCURACY COMPARISON VISUALIZATION
# =========================================================

fig, ax = plt.subplots(figsize=(10, 6))

model_names = list(cv_results.keys())
means = [np.array(cv_results[m]['accuracies']).mean() for m in model_names]
stds = [np.array(cv_results[m]['accuracies']).std() for m in model_names]

x_pos = np.arange(len(model_names))
ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('7-Fold Cross-Validation Accuracy Comparison', fontsize=14)
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names)
ax.set_ylim([0.7, 1.0])

for i, (mean, std) in enumerate(zip(means, stds)):
    ax.text(i, mean + std + 0.01, f'{mean:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved accuracy_comparison.png")

# =========================================================
# BEST MODEL SUMMARY
# =========================================================

print("\n" + "="*80)
print("BEST MODEL SUMMARY")
print("="*80 + "\n")

best_model = max(
    cv_results.keys(),
    key=lambda m: np.array(cv_results[m]['accuracies']).mean()
)

best_accuracies = np.array(cv_results[best_model]['accuracies'])

print(f"Best Model: {best_model}")
print(f"Mean Accuracy: {best_accuracies.mean():.7f}")
print(f"Std Dev: {best_accuracies.std():.7f}")
print(f"Accuracy Range: [{best_accuracies.min():.7f}, {best_accuracies.max():.7f}]")

print(f"\nAccuracy by Fold:")
for fold_num, acc in enumerate(best_accuracies):
    print(f"  Fold {fold_num + 1}: {acc:.7f}")

# =========================================================
# MODEL COMPARISON SUMMARY
# =========================================================

print("\n" + "="*80)
print("MODEL RANKING BY MEAN ACCURACY")
print("="*80 + "\n")

ranking = sorted(
    [(name, np.array(cv_results[name]['accuracies']).mean()) for name in cv_results.keys()],
    key=lambda x: x[1],
    reverse=True
)

for rank, (name, acc) in enumerate(ranking, 1):
    print(f"{rank}. {name:20} {acc:.7f}")

print("\n" + "="*80)
print("Finished 7-Fold Stratified Cross-Validation Analysis")
print("="*80)

 
print("\n" + "="*80)
print("Finished Complete 7-Fold Stratified Cross-Validation Analysis")
print("Classical ML + Deep Learning Models")
print("="*80)
 