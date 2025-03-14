
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import Bottleneck

class SKConv(nn.Module):
    def __init__(self, features, WH=32, M=1, G=16, r=8, stride=1, L=16):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features

        # 定义单一分支的卷积层
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
            )
        ])

        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([nn.Linear(d, features) for _ in range(M)])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        feats = [conv(x) for conv in self.convs]
        feats = torch.stack(feats, dim=1)

        U = sum([feats[:, i] for i in range(self.M)])
        s = U.mean(-1).mean(-1)

        z = self.fc(s)
        attention_vectors = [fc(z) for fc in self.fcs]
        attention_vectors = torch.stack(attention_vectors, dim=1)
        attention_vectors = self.softmax(attention_vectors)

        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        V = (feats * attention_vectors).sum(dim=1)

        return V


class ResNet50Encoder(nn.Module):
    def __init__(self, base_channels=32, pretrained=True):
        super(ResNet50Encoder, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

        self.inplanes = base_channels

        self.initial = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            SKConv(base_channels, WH=32, M=1, G=16, r=8)
        )
        self.maxpool = resnet.maxpool

        self.layer1 = self._make_layer(Bottleneck, planes=base_channels, blocks=3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, planes=base_channels * 2, blocks=4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, planes=base_channels * 4, blocks=6, stride=2)
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
            layers.append(SKConv(self.inplanes, WH=32, M=1, G=16, r=8))

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
    def __init__(self, in_channels, out_channels, dropout_prob=0.1):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class R50KNet(nn.Module):
    def __init__(self, num_classes=1, base_channels=32, dropout_prob=0.1):
        super(R50KNet, self).__init__()
        self.encoder = ResNet50Encoder(base_channels=base_channels)

        # 修改解码器中的上采样层，使其能够接受拼接后的输入
        # 注意：这里假设x5有base_channels * 32个通道，x4有base_channels * 16个通道
        self.up1 = nn.ConvTranspose2d(base_channels * 32, base_channels * 8, kernel_size=2, stride=2)
        self.decoder1 = DecoderBlock(base_channels * 8 + base_channels * 16, base_channels * 8, dropout_prob)

        self.up2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.decoder2 = DecoderBlock(base_channels * 4 + base_channels * 8, base_channels * 4, dropout_prob)

        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.decoder3 = DecoderBlock(base_channels * 2 + base_channels * 4, base_channels * 2, dropout_prob)

        self.up4 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.decoder4 = DecoderBlock(base_channels * 2, base_channels, dropout_prob)

        self.final_conv = nn.Conv2d(base_channels, num_classes, 1)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x0, x2, x3, x4, x5 = self.encoder(x)

        up_x5 = self.up1(x5)  # 对x5进行上采样
        d1 = self.decoder1(torch.cat([up_x5, x4], 1))  # 将上采样后的x5与x4拼接
        up_d1 = self.up2(d1)
        d2 = self.decoder2(torch.cat([up_d1, x3], 1))
        up_d2 = self.up3(d2)
        d3 = self.decoder3(torch.cat([up_d2, x2], 1))
        up_d3 = self.up4(d3)
        d4 = self.decoder4(torch.cat([up_d3, x0], 1))

        out = self.final_conv(d4)
        return self.final_upsample(out)