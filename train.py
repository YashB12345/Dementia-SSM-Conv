import os
import sys
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import ConcatDataset, Subset
import torch.optim as optim
from tqdm import tqdm
from torchsummary import summary
from MedMamba import VSSM
from torch.utils.data import Dataset, DataLoader, random_split

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.CenterCrop(224),         # Crop 224x224 patch
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        "flip": transforms.Compose([  transforms.CenterCrop(224),         # Crop 224x224 patch
                                     transforms.RandomHorizontalFlip(p=1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.CenterCrop(224),         # Crop 224x224 patch
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])}


    DATA_SET_PATH = '/kaggle/input/datasets/yfinity/adni-1-5t-coronal-2to2-antspynet-flirt-n4-b180-3n4'

    orig_dataset = datasets.ImageFolder(root=DATA_SET_PATH,
                                         transform=data_transform["train"])
    flip_dataset = datasets.ImageFolder(root=DATA_SET_PATH,
                                         transform=data_transform["flip"])

    full_dataset = ConcatDataset([orig_dataset, flip_dataset])

    class_to_idx = orig_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in class_to_idx.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    #train_dataset = tf.data.Dataset.from_tensor_slices(train_x, train_y )
    train_loader = torch.utils.data.DataLoader(full_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    # 2. Define split ratios and calculate lengths
    train_size = int(0.9 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size # Adjust for any rounding issues

    # Ensure reproducibility with a fixed seed
    torch.manual_seed(42)

    # 3. Perform the random split
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # 4. Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Verify the sizes
    print(f"Train set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    #validate_dataset = datasets.ImageFolder(root=DATA_SET_PATH,transform=data_transform["val"])

    val_num = len(val_dataset)
    validate_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation with {} classes".format(len(train_dataset),
                                                                           val_num, len(class_to_idx)))

    model_name = "VSSM"
    save_path = './{}Net.pth'.format(model_name)
    net = VSSM(num_classes=len(class_to_idx))
    net.to(device)
    net.load_state_dict(torch.load("./VSSMNet.pth", map_location=device))
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    # full_dataset is your ConcatDataset object
    all_labels = []

    for dataset in full_dataset.datasets:
        # Check if the sub-dataset has targets (standard for ImageFolder)
        if hasattr(dataset, 'targets'):
            all_labels.extend(dataset.targets)
        # Some datasets use .labels instead of .targets
        elif hasattr(dataset, 'labels'):
            all_labels.extend(dataset.labels)
        else:
            # Fallback: manually iterate if targets aren't exposed (slower)
            print("Warning: Dataset doesn't have .targets attribute. Extracting manually...")
            for _, label in dataset:
                all_labels.append(label)

    all_labels = np.array(all_labels)
    all_indices = np.arange(len(all_labels))

    # Initialize StratifiedKFold
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # To store results across folds
    fold_results = []

    epochs = 25
    best_acc = 0.0
    best_loss = 1.0


    for fold, (train_ids, val_ids) in enumerate(skf.split(all_indices, all_labels)):
        print(f"--- Fold {fold + 1}/{k_folds} ---")
        
        # Create Subsets for this fold
        train_sub = Subset(full_dataset, train_ids)
        val_sub = Subset(full_dataset, val_ids)
        
        # Create DataLoaders
        train_loader = DataLoader(train_sub, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)
    
        train_steps = len(train_loader)
        for epoch in range(epochs):
            # train
            net.train()
            running_loss = 0.0
            train_bar = tqdm(train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                images, labels = data
                optimizer.zero_grad()
                outputs = net(images.to(device))
                loss = loss_function(outputs, labels.to(device))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                #train_bar.desc = "train epoch[{}/{}] loss:{:.7f}".format(epoch + 1,epochs,loss)

            # validate
            net.eval()
            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                val_bar = tqdm(validate_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = net(val_images.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_accurate = acc / val_num
            print('[epoch %d] train_loss: %.7f  val_accuracy: %.7f' %
                (epoch + 1, running_loss / train_steps, val_accurate))

            if (val_accurate > best_acc  or (running_loss / train_steps) < best_loss):
                best_acc = val_accurate
                best_loss = running_loss / train_steps
                torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
