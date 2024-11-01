import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Cityscapes
from torch.optim.lr_scheduler import StepLR
import urllib.request
import tarfile

class FCN8s(nn.Module):
    def __init__(self, n_class):
        super(FCN8s, self).__init__()
        self.n_class = n_class

        # Convolutional layers (based on VGG-16)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        )

        # Fully convolutional layers
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        self.score_fr = nn.Conv2d(4096, n_class, kernel_size=1)

        # Upsampling layers
        self.upscore2 = nn.ConvTranspose2d(n_class, n_class, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)
        self.upscore8 = nn.ConvTranspose2d(n_class, n_class, kernel_size=16, stride=8, padding=4, output_padding=0, bias=False)

        # Skip connections
        self.score_pool4 = nn.Conv2d(512, n_class, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, n_class, kernel_size=1)
        self.upscore_pool4 = nn.ConvTranspose2d(n_class, n_class, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        pool3 = h  # Save pool3 output for skip connection

        h = self.conv4(h)
        pool4 = h  # Save pool4 output for skip connection

        h = self.conv5(h)

        h = self.fc6(h)
        h = self.fc7(h)
        h = self.score_fr(h)
        h = self.upscore2(h)

        # Add skip connection from pool4
        h_pool4 = self.score_pool4(pool4)
        h = h[:, :, :h_pool4.size(2), :h_pool4.size(3)]  # Crop to match dimensions
        h += h_pool4
        h = self.upscore_pool4(h)

        # Add skip connection from pool3
        h_pool3 = self.score_pool3(pool3)
        h = h[:, :, :h_pool3.size(2), :h_pool3.size(3)]  # Crop to match dimensions
        h += h_pool3
        h = self.upscore8(h)
        h = h[:, :, :x.size(2), :x.size(3)]  # Crop to match input dimensions

        return h

def download_cityscapes():
    url = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/cityscapes.tar.gz"  # Replace with the actual URL to download the dataset
    output_path = "./data/cityscapes_dataset.tar.gz"
    if not os.path.exists("./data/cityscapes"):
        os.makedirs("./data", exist_ok=True)
        print("Downloading Cityscapes dataset...")
        urllib.request.urlretrieve(url, output_path)
        print("Extracting Cityscapes dataset...")
        with tarfile.open(output_path, 'r:gz') as tar_ref:
            tar_ref.extractall("./data")
        os.remove(output_path)
        print("Cityscapes dataset downloaded and extracted.")
    else:
        print("Cityscapes dataset already exists.")

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0

    for i, (image, target) in enumerate(dataloader):
        # Move data to the device
        image = image.to(device)
        target = target.to(device)

        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(image)

        # Compute the loss
        loss = criterion(outputs, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()

        # Print loss information
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

def validate(model, dataloader, criterion, device, epoch, num_epochs):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (image, target) in enumerate(dataloader):
            # Move data to the device
            image = image.to(device)
            target = target.to(device)

            # Forward pass
            outputs = model(image)

            # Compute the loss
            loss = criterion(outputs, target)
            val_loss += loss.item()

    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

def main():
    # Download Cityscapes dataset if not available
    download_cityscapes()

    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Define transformations for the Cityscapes dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.286, 0.325, 0.283], std=[0.176, 0.180, 0.177])
    ])

    # Load Cityscapes dataset
    train_dataset = Cityscapes(root='./data', split='train', mode='fine', target_type='semantic', transform=transform, target_transform=transform)
    val_dataset = Cityscapes(root='./data', split='val', mode='fine', target_type='semantic', transform=transform, target_transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    model = FCN8s(n_class=34).to(device)  # Cityscapes has 34 classes
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Add a learning rate scheduler for decay
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
        validate(model, val_loader, criterion, device, epoch, num_epochs)

        # Step the scheduler after each epoch
        scheduler.step()

        # Save model checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/fcn8s_cityscape_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
