import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN, self).__init__()
        
        # Define convolutional layers
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=100)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully convolutional layers
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)

        # Deconvolution layers
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)

        # Scoring layers for pool3 and pool4
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.output_layer = nn.Sequential(
            nn.Conv2d(21, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Downsampling path
        x_original = x
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        pool1 = self.pool1(x)

        x = F.relu(self.conv2_1(pool1))
        x = F.relu(self.conv2_2(x))
        pool2 = self.pool2(x)

        x = F.relu(self.conv3_1(pool2))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        pool3 = self.pool3(x)

        x = F.relu(self.conv4_1(pool3))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        pool4 = self.pool4(x)

        x = F.relu(self.conv5_1(pool4))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        pool5 = self.pool5(x)

        # Fully connected layers
        x = F.relu(self.fc6(pool5))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5, self.training)
        score_fr = self.score_fr(x)

        # Upsampling and fusion
        upscore2 = self.upscore2(score_fr)
        score_pool4 = self.score_pool4(pool4)
        score_pool4 = self._crop(score_pool4, upscore2)
        fuse_pool4 = upscore2 + score_pool4

        upscore_pool4 = self.upscore_pool4(fuse_pool4)
        score_pool3 = self.score_pool3(pool3)
        score_pool3 = self._crop(score_pool3, upscore_pool4)
        fuse_pool3 = upscore_pool4 + score_pool3

        upscore8 = self.upscore8(fuse_pool3)
        out = self._crop(upscore8, x_original)
        out = self.output_layer(out)
        return out

    def _crop(self, input, target):
        # Cropping to match the shape of the target
        _, _, h, w = target.size()
        return input[:, :, :h, :w]

# Usage:
net = FCN(num_classes=21)
input = torch.randn(1, 3, 500, 500)  # Example input tensor
output = net(input)
print(output.shape)  # Expected output shape
