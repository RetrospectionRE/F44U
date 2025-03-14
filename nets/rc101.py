import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import Bottleneck

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_att = self.channel_attention(x)
        x_channel = x * channel_att

        spatial_att = torch.cat([torch.max(x_channel, dim=1, keepdim=True)[0],
                                 torch.mean(x_channel, dim=1, keepdim=True)], dim=1)
        spatial_att = self.spatial_attention(spatial_att)

        return x_channel * spatial_att


class ResNet101Encoder(nn.Module):
    def __init__(self, base_channels=64, pretrained=True):
        super(ResNet101Encoder, self).__init__()
        resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)

        # 初始化通道参数
        self.inplanes = base_channels

        # 初始卷积层（替换原ResNet的卷积）
        self.initial = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            CBAM(base_channels)
        )
        self.maxpool = resnet.maxpool

        # 修正后的各层构建
        self.layer1 = self._make_layer(Bottleneck, planes=base_channels, blocks=3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, planes=base_channels * 2, blocks=4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, planes=base_channels * 4, blocks=23, stride=2) # 更改blocks数量
        self.layer4 = self._make_layer(Bottleneck, planes=base_channels * 8, blocks=3, stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            layers.append(CBAM(self.inplanes))  # 在块后添加CBAM

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.maxpool(x0)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x0, x2, x3, x4, x5

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.5):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CBAM(out_channels),
            nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class R101CNet(nn.Module):
    def __init__(self, num_classes=1, base_channels=16, dropout_prob=0.5):
        super(R101CNet, self).__init__()
        self.encoder = ResNet101Encoder(base_channels=base_channels)

        # 根据拼接后的通道数调整DecoderBlock的输入通道数
        self.up1 = nn.ConvTranspose2d(base_channels * 32, base_channels * 16, 2, stride=2)
        self.decoder1 = DecoderBlock(base_channels * 16 + base_channels * 16, base_channels * 16, dropout_prob)

        self.up2 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.decoder2 = DecoderBlock(base_channels * 8 + base_channels * 8, base_channels * 8, dropout_prob)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.decoder3 = DecoderBlock(base_channels * 4 + base_channels * 4, base_channels * 4, dropout_prob)

        self.up4 = nn.ConvTranspose2d(base_channels * 4, base_channels, 2, stride=2)
        self.decoder4 = DecoderBlock(base_channels * 2, base_channels, dropout_prob)  # 注意此处的拼接

        self.final_conv = nn.Conv2d(base_channels, num_classes, 1)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x0, x2, x3, x4, x5 = self.encoder(x)

        d1 = self.decoder1(torch.cat([self.up1(x5), x4], 1))
        d2 = self.decoder2(torch.cat([self.up2(d1), x3], 1))
        d3 = self.decoder3(torch.cat([self.up3(d2), x2], 1))
        d4 = self.decoder4(torch.cat([self.up4(d3), x0], 1))

        out = self.final_conv(d4)
        return self.final_upsample(out)