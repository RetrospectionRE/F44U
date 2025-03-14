import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # 修复：in_channels 正确传递
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x, indices = self.pool(x)
        return x, indices


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x, indices):
        x = self.unpool(x, indices)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x


class SegNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1, base_channels=16, dropout_prob=0.5):
        super().__init__()

        # Encoder
        self.encoder1 = EncoderBlock(input_channels, base_channels, dropout_prob)
        self.encoder2 = EncoderBlock(base_channels, base_channels * 2, dropout_prob)
        self.encoder3 = EncoderBlock(base_channels * 2, base_channels * 4, dropout_prob)
        self.encoder4 = EncoderBlock(base_channels * 4, base_channels * 8, dropout_prob)

        # Decoder
        self.decoder4 = DecoderBlock(base_channels * 8, base_channels * 4, dropout_prob)
        self.decoder3 = DecoderBlock(base_channels * 4, base_channels * 2, dropout_prob)
        self.decoder2 = DecoderBlock(base_channels * 2, base_channels, dropout_prob)
        self.decoder1 = DecoderBlock(base_channels, base_channels, dropout_prob)

        # Final prediction layer
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoding path
        x, idx1 = self.encoder1(x)
        x, idx2 = self.encoder2(x)
        x, idx3 = self.encoder3(x)
        x, idx4 = self.encoder4(x)

        # Decoding path
        x = self.decoder4(x, idx4)
        x = self.decoder3(x, idx3)
        x = self.decoder2(x, idx2)
        x = self.decoder1(x, idx1)

        # Final classification layer
        x = self.final_conv(x)
        return x


# 使用示例
