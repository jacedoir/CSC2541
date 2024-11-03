import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedConvBlock, self).__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.dilated_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, 
            padding=padding, dilation=dilation
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.dilated_conv(x)
        x = self.bn(x)
        return self.activation(x)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)

    def forward(self, x):
        scale = torch.mean(x, dim=(2, 3), keepdim=True)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale

class UniRepLKNetSmall(nn.Module):
    def __init__(self, in_channels, base_channels=96):
        super(UniRepLKNetSmall, self).__init__()
        
        # Initial large-kernel conv layer
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=13, padding=6)
        
        # Dilated convolution layers with varying dilation rates
        self.dilated_convs = nn.ModuleList([
            DilatedConvBlock(base_channels, base_channels, kernel_size=5, dilation=1),
            DilatedConvBlock(base_channels, base_channels, kernel_size=7, dilation=2),
            DilatedConvBlock(base_channels, base_channels, kernel_size=3, dilation=3),
            DilatedConvBlock(base_channels, base_channels, kernel_size=3, dilation=4),
            DilatedConvBlock(base_channels, base_channels, kernel_size=3, dilation=5)
        ])
        
        # Squeeze-and-Excitation (SE) block
        self.se_block = SEBlock(base_channels)
        
        # Down-sampling layers to reduce spatial dimensions
        self.downsampling = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.GELU()
        )

    def forward(self, x):
        # Initial convolution
        x = self.init_conv(x)
        
        # Capture outputs after each dilated convolution
        dilated_outputs = []
        for conv in self.dilated_convs:
            x = conv(x)
            dilated_outputs.append(x)
        
        # Squeeze-and-Excitation
        se_features = self.se_block(x)
        
        # Downsample the feature map
        final_output = self.downsampling(se_features)
        
        return dilated_outputs, final_output
