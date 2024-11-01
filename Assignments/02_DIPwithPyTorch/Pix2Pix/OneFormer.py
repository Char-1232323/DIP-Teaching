import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.attention(x, x, x)[0]
        x = self.dropout1(attention) + x
        x = self.norm1(x)
        
        forward = self.feed_forward(x)
        x = self.dropout2(forward) + x
        x = self.norm2(x)
        
        return x

class OneFormer(nn.Module):
    def __init__(self, num_classes):
        super(OneFormer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 增加通道数
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=3, padding=1),  # 增加通道数
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 增加通道数
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 增加通道数
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.transformer = TransformerBlock(embed_size=256, heads=8, dropout=0.1, forward_expansion=4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, num_classes, kernel_size=4, stride=2, padding=1)
        )
        self.outlayer = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(num_classes, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        # Reshape for transformer (N, C, H, W) to (N, H*W, C)
        N, C, H, W = x.size()
        x = x.view(N, C, -1).permute(0, 2, 1)  # (N, H*W, C)
        
        x = self.transformer(x)
        
        # Reshape back (N, C, H, W)
        x = x.permute(0, 2, 1).view(N, C, H, W)
        x = self.decoder(x)
        out =self.outlayer(x)
        return out
