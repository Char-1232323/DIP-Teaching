import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        ### FILL: add more CONV Layers
        self.conv1 = self._conv_block(3, 16)
        self.conv2 = self._conv_block(16, 32)
        self.conv3 = self._conv_block(32, 64)
        self.conv4 = self._conv_block(64, 128)  # 增加更多卷积层

        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function
        self.deconv1 = self._deconv_block(128, 64)
        self.deconv2 = self._deconv_block(64, 32)
        self.deconv3 = self._deconv_block(32, 16)
        self.deconv4 = nn.Sequential(  # 增加反卷积层
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # 输出RGB值
        )

    def _conv_block(self, in_channels, out_channels, dropout_prob=0.1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob)  # 添加Dropout层
        )

    def _deconv_block(self, in_channels, out_channels, dropout_prob=0.1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_prob)  # 添加Dropout层
        )

    def forward(self, x):
        # Encoder forward pass
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Decoder forward pass
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        output = self.deconv4(x)
        ### FILL: encoder-decoder forward pass
        
        return output
    

class FCN8s(nn.Module):
    def __init__(self, num_classes):
        super(FCN8s, self).__init__()
        # Load VGG16 pretrained model
        vgg = models.vgg16(pretrained=True)
        features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
        
        self.features = nn.Sequential(*features)
        
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2, bias=False)
        self.upscore8 = nn.Sequential(nn.ConvTranspose2d(num_classes, 3, kernel_size=14, stride=8, padding=3, bias=False),
                                     nn.Tanh())  # Output size: (N, 3, 256, 256)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2, bias=False)

    def forward(self, x):
        # Input size: (1, 3, 256, 256)
        pool3 = None
        pool4 = None
        for i in range(len(self.features)):
            x = self.features[i](x)
            if i == 16:  # pool3 layer
                pool3 = x  # Size after pool3: (1, 256, 32, 32)
            elif i == 23:  # pool4 layer
                pool4 = x  # Size after pool4: (1, 512, 16, 16)

        x = F.relu(self.fc6(x))  # Size after fc6: (1, 4096, 8, 8)

        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))  # Size after fc7: (1, 4096, 8, 8)
        x = F.dropout(x, 0.5, self.training)
        x = self.score_fr(x)  # Size after score_fr: (1, num_classes, 8, 8)

        x = self.upscore2(x)  # Size after upscore2: (1, num_classes, 16, 16)

        x = x + self.score_pool4(pool4)  # Size after adding score_pool4: (1, num_classes, 16, 16)

        x = self.upscore_pool4(x)  # Size after upscore_pool4: (1, num_classes, 32, 32)

        x = x + self.score_pool3(pool3)  # Size after adding score_pool3: (1, num_classes, 32, 32)
 
        x = self.upscore8(x)  # Size after upscore8: (1, num_classes, 256, 256)

        return x