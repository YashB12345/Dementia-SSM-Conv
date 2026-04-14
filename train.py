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
import torch.nn.functional as F

def get_embeddings(model, images):
    # VSSM typically supports forward_features
    features = model.forward_features(images)
    return features.view(features.size(0), -1)


def filter_batch(model, images, labels, centroid, threshold):
    emb = get_embeddings(model, images)
    distances = torch.norm(emb - centroid, dim=1)

    mask = distances < threshold  # ✅ THIS FILTERS OUTLIERS

    return images[mask], labels[mask], mask

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

def is_valid_image(img):
    return not torch.isnan(img).any() and img.std() > 1e-5

def compute_embedding_threshold(model, train_loader, device, percentile=0.9):
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(device)

            # 👇 IMPORTANT: adjust this to your model
            features = model.forward_features(images)  
            embeddings = features.view(features.size(0), -1)

            all_embeddings.append(embeddings.cpu())

    all_embeddings = torch.cat(all_embeddings)

    centroid = all_embeddings.mean(dim=0)

    distances = torch.norm(all_embeddings - centroid, dim=1)

    threshold = torch.quantile(distances, percentile)

    return centroid.to(device), threshold.to(device)
    

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


    DATA_SET_PATH = '/kaggle/input/datasets/yfinity/adni1-1yr-1-5t-1571sam-antspynet-coronal-central'

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
        orig_dataset, [train_size, val_size, test_size]
    )

    #train_dataset = train_dataset.mean(dim=1, keepdim=True)
    #val_dataset = val_dataset.mean(dim=1, keepdim=True)
    #test_dataset = test_dataset.mean(dim=1, keepdim=True)

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
    net = VSSM(num_classes=len(class_to_idx))
    net.to(device)
    loss_function = FocalLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    #summary(net, input_size=(1, 224, 224))

    epochs = 100
    best_acc = 0.0
    save_path = './{}Net.pth'.format(model_name)
    train_steps = len(train_loader)

    print("Computing embedding threshold...")

    centroid, threshold = compute_embedding_threshold(
        net, train_loader, device, percentile=0.9
    )
    
    print("Threshold:", threshold.item())

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            print(type(images))
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.7f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                # 🔥 --- FILTER OUT BAD EMBEDDINGS ---
                val_images, val_labels, mask = filter_batch(
                    net, val_images, val_labels, centroid, threshold
                )
                
                removed = (~mask).sum().item()
                if removed > 0:
                    print(f"Filtered {removed} samples")
                
                # Skip if all removed
                if val_images.size(0) == 0:
                    continue
                
                # Normal forward pass
                outputs = net(val_images)
                predict_y = torch.max(outputs, dim=1)[1]
                
                acc += torch.eq(predict_y, val_labels).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.7f  val_accuracy: %.7f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
