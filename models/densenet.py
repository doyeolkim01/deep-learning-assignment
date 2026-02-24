import torch
import torch.nn as nn
import torch.nn.functional as F


class Layer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)

        return torch.cat([x, out], dim=1)  # (N, C, H, W) 에서 C를 축으로 이어 붙인다.


class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        mid_channels = growth_rate * 4

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, padding=0)

        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        return torch.cat([x, out], dim=1)  # (N, C, H, W) 에서 C를 축으로 이어 붙인다.


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        out = self.pool(out)

        return out


class Block(nn.Sequential):
    def __init__(self, in_channels, growth_rate, num_layers, layer):
        layers = []
        channels = in_channels

        for i in range(num_layers):
            layers.append(layer(channels, growth_rate))
            channels += growth_rate

        self.out_channels = channels

        super().__init__(*layers)


class DenseNet(nn.Module):
    def __init__(self, num_blocks, num_layers, layer, growth_rate, num_classes=10):
        super().__init__()
        channels = 16
        self.num_blocks = num_blocks

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.blocks = nn.ModuleList()  # PyTorch에서 사용되는 모듈들을 리스트 형태로 관리하는 클래스
        self.transition_layers = nn.ModuleList()

        for i in range(num_blocks):
            block = Block(channels, growth_rate, num_layers, layer)
            self.blocks.append(block)
            channels = block.out_channels

            if i < num_blocks - 1:
                transition_layer = TransitionLayer(channels, channels // 2)
                self.transition_layers.append(transition_layer)
                channels = channels // 2

        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.num_features = channels
        self.fc = nn.Linear(channels, num_classes)

    def forward_features(self, x):
        out = self.conv1(x)

        for i in range(self.num_blocks - 1):
            out = self.blocks[i](out)
            out = self.transition_layers[i](out)

        out = self.blocks[self.num_blocks - 1](out)

        out = self.bn(out)
        out = self.relu(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        return out

    def forward(self, x):
        features = self.forward_features(x)
        logits = self.fc(features)

        return logits


class DenseNetEncoder(nn.Module):
    def __init__(self, num_blocks, num_layers, layer, growth_rate):
        super().__init__()
        self.backbone = DenseNet(num_blocks, num_layers, layer, growth_rate, num_classes=10)
        self.num_features = self.backbone.num_features

    def forward(self, x):
        return self.backbone.forward_features(x)


def densenet_encoder():
    return DenseNetEncoder(num_blocks=3, num_layers=12, layer=BottleneckLayer, growth_rate=12)
