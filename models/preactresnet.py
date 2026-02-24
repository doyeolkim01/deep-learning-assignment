import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActResnetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        shortcut = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        # Option A - Zero Padding Shortcut
        if self.stride != 1:  # 공간 크기 맞추기(1x1 average pooling)
            shortcut = F.avg_pool2d(shortcut, kernel_size=1, stride=self.stride)
        if self.in_channels != self.out_channels:  # 채널 수 맞추기(zero padding)
            zero_block = torch.zeros(
                shortcut.shape[0],
                self.out_channels - self.in_channels,
                shortcut.shape[2],
                shortcut.shape[3],
                device=shortcut.device
            )
            shortcut = torch.cat((shortcut, zero_block), dim=1)

        out += shortcut
        return out


class PreActResnetBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        mid_channels = out_channels // 4

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()

        # Option B - Projection Shortcut using 1×1 Conv
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.shortcut = nn.Identity()  # 입력이 그대로 출력됨

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, n, block, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv2 = self.construct_layer(16, 16, block, n, 1)  # 32x32, 16 filters
        self.conv3 = self.construct_layer(16, 32, block, n, 2)  # 16x16, 32 filters
        self.conv4 = self.construct_layer(32, 64, block, n, 2)  # 8x8, 64 filters

        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 각 채널마다 평균을 구하여 (N, 64, 8, 8) -> (N, 64, 1, 1)로 압축
        self.num_features = 64
        self.fc = nn.Linear(self.num_features, num_classes)

    def construct_layer(self, in_channels, out_channels, block, num_blocks, stride):
        layers = []
        layers.append(block(in_channels, out_channels, stride))

        for i in range(0, num_blocks - 1):
            layers.append(block(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward_features(self, x):
        out = self.conv1(x)

        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        out = self.bn(out)
        out = self.relu(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        return out

    def forward(self, x):
        features = self.forward_features(x)
        logits = self.fc(features)

        return logits


class PreActResNetEncoder(nn.Module):
    def __init__(self, n, block):
        super().__init__()
        self.backbone = PreActResNet(n=n, block=block, num_classes=10)
        self.num_features = self.backbone.num_features

    def forward(self, x):
        return self.backbone.forward_features(x)


def preactresnet20_encoder():
    return PreActResNetEncoder(n=3, block=PreActResnetBasicBlock)
