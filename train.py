import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import ConcatDataset
import torch.optim as optim
from tqdm import tqdm
from torchsummary import summary
from MedMamba import VSSM
from torch.utils.data import Dataset, DataLoader, random_split

# Define the threshold (e.g., 10 KB in bytes)
MIN_SIZE_BYTES = 10 * 1024

def check_image_size(path):
    # Check if the file exists and is larger than the minimum size
    return os.path.isfile(path) and os.path.getsize(path) > MIN_SIZE_BYTES


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        "flip": transforms.Compose([  transforms.Resize((224, 224)),
                                     transforms.RandomHorizontalFlip(p=1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])}


    DATA_SET_PATH = '/kaggle/input/datasets/yfinity/adni-phase1-5class/sagittal'

    orig_dataset = datasets.ImageFolder(root=DATA_SET_PATH,
                                        is_valid_file=check_image_size,  # Uses the filter
                                         transform=data_transform["train"])
    flip_dataset = datasets.ImageFolder(root=DATA_SET_PATH,
                                        is_valid_file=check_image_size,  # Uses the filter
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
    net = VSSM(num_classes=len(class_to_idx))
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    #summary(net, input_size=(1, 224, 224))

    epochs = 150
    best_acc = 0.0
    save_path = './{}Net.pth'.format(model_name)
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
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.7f  val_accuracy: %.7f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
