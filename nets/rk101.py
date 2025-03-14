import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SKConv(nn.Module):
    def __init__(self, channels, M=2, G=32, r=16):
        super(SKConv, self).__init__()
        d = max(channels // r, 32)
        self.M = M
        self.channels = channels

        # 多尺度卷积
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1 + i, dilation=1 + i, groups=G),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ))

        # 特征融合
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, d, kernel_size=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )

        # 注意力权重生成
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Conv2d(d, channels, kernel_size=1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)

        # 多尺度特征
        feats = [conv(x) for conv in self.convs]
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.channels, feats.size(2), feats.size(3))

        # 特征融合
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        # 生成注意力权重
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.channels, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        # 特征加权融合
        feats_V = torch.sum(feats * attention_vectors, dim=1)
        return feats_V


class ResNetEncoder_SK(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)  # 替换为ResNet101
        self.base_channels = base_channels

        self.conv1 = nn.Conv2d(3, self.base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # 添加SK模块
        self.layer1 = self._modify_resnet_layer(resnet.layer1, self.base_channels)
        self.sk1 = SKConv(self.base_channels)
        self.layer2 = self._modify_resnet_layer(resnet.layer2, self.base_channels * 4)
        self.sk2 = SKConv(self.base_channels * 4)
        self.layer3 = self._modify_resnet_layer(resnet.layer3, self.base_channels * 8)
        self.sk3 = SKConv(self.base_channels * 8)
        self.layer4 = self._modify_resnet_layer(resnet.layer4, self.base_channels * 16)
        self.sk4 = SKConv(self.base_channels * 16)

    def _modify_resnet_layer(self, layer, output_channels):
        for name, module in layer.named_children():
            if isinstance(module, nn.Conv2d):
                new_module = nn.Conv2d(
                    module.in_channels, output_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    bias=module.bias
                )
                return nn.Sequential(new_module, *(list(layer.children())[1:]))
        return layer

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x1 = self.maxpool(x0)

        x2 = self.layer1(x1)
        x2 = self.sk1(x2)
        x3 = self.layer2(x2)
        x3 = self.sk2(x3)
        x4 = self.layer3(x3)
        x4 = self.sk3(x4)
        x5 = self.layer4(x4)
        x5 = self.sk4(x5)

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
            nn.Dropout(dropout_prob)
        )
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            skip = self.skip_conv(skip)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
            x = self.conv(x)
        x = self.dropout(x)
        return x

class R101KNet(nn.Module):  # 修改网络名称以反映使用ResNet101
    def __init__(self, n_classes, base_channels=64, dropout_prob=0.5):
        super().__init__()
        self.encoder = ResNetEncoder_SK(base_channels)
        encoder_channels = [base_channels * 16, base_channels * 8, base_channels * 4,
                            base_channels, base_channels, base_channels]
        decoder_channels = [base_channels * 8, base_channels * 4, base_channels * 2,
                            base_channels, int(base_channels / 2)]

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(decoder_channels)):
            in_ch = encoder_channels[i] if i == 0 else decoder_channels[i - 1]
            skip_ch = encoder_channels[i + 1]
            self.decoder_blocks.append(DecoderBlock(in_ch, decoder_channels[i], skip_ch, dropout_prob))

        self.final_conv = nn.Conv2d(decoder_channels[-1], n_classes, kernel_size=1)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        features = self.encoder(x)
        x = features[0]
        for i, block in enumerate(self.decoder_blocks):
            skip = features[i + 1]
            x = block(x, skip)
        x = self.final_upsample(x)
        x = self.dropout(x)
        x = self.final_conv(x)
        return x