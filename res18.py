import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights  # 导入权重枚举


class ResNetEncoder(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base_channels = base_channels

        # 修改第一层卷积以适应不同的基础通道数
        self.conv1 = nn.Conv2d(3, self.base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # 替换原始的resnet层以匹配新的基础通道数
        self.layer1 = self._modify_resnet_layer(resnet.layer1, self.base_channels)
        self.layer2 = self._modify_resnet_layer(resnet.layer2, self.base_channels * 2)
        self.layer3 = self._modify_resnet_layer(resnet.layer3, self.base_channels * 4)
        self.layer4 = self._modify_resnet_layer(resnet.layer4, self.base_channels * 8)

    def _modify_resnet_layer(self, layer, output_channels):
        """辅助函数，用于根据指定的输出通道数修改resnet层"""
        for name, module in layer.named_children():
            if isinstance(module, nn.Conv2d):
                new_module = nn.Conv2d(in_channels=module.in_channels,
                                       out_channels=output_channels,
                                       kernel_size=module.kernel_size,
                                       stride=module.stride,
                                       padding=module.padding,
                                       bias=module.bias)
                return nn.Sequential(new_module, *(list(layer.children())[1:]))
        return layer

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x1 = self.maxpool(x0)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return [x5, x4, x3, x2, x1, x0]


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, dropout_prob=0.2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.skip_conv = nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob)  # 添加Dropout层
        )
        self.dropout = nn.Dropout(dropout_prob)  # Dropout层用于处理上采样后的特征

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            skip = self.skip_conv(skip)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
            x = self.conv(x)
        x = self.dropout(x)  # 应用Dropout
        return x


class ResNet18_UNet(nn.Module):
    def __init__(self, n_classes, base_channels=64, dropout_prob=0.5):
        super().__init__()
        self.encoder = ResNetEncoder(base_channels)
        encoder_channels = [base_channels * 8, base_channels * 4, base_channels * 2, base_channels, base_channels,
                            base_channels]
        decoder_channels = [base_channels * 4, base_channels * 2, base_channels, int(base_channels / 2),
                            int(base_channels / 4)]
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(decoder_channels)):
            in_ch = encoder_channels[i] if i == 0 else decoder_channels[i - 1]
            skip_ch = encoder_channels[i + 1]
            self.decoder_blocks.append(DecoderBlock(in_ch, decoder_channels[i], skip_ch, dropout_prob))
        self.final_conv = nn.Conv2d(decoder_channels[-1], n_classes, kernel_size=1)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout(dropout_prob)  # 最终卷积前的Dropout层

    def forward(self, x):
        features = self.encoder(x)
        x = features[0]
        for i, block in enumerate(self.decoder_blocks):
            skip = features[i + 1]
            x = block(x, skip)
        x = self.final_upsample(x)
        x = self.dropout(x)  # 在最终卷积之前应用Dropout
        x = self.final_conv(x)
        return x