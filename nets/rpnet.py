import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights



# 金字塔注意力模块
class PyramidAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(PyramidAttention, self).__init__()
        self.channels = channels
        self.reduction = reduction

        # 多尺度卷积
        self.conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.conv3 = nn.Conv2d(channels, channels // reduction, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(channels, channels // reduction, kernel_size=3, padding=2, dilation=2)

        # 注意力生成
        self.fusion = nn.Sequential(
            nn.Conv2d(3 * (channels // reduction), channels, kernel_size=1),  # 输出通道数与输入相同
            nn.Sigmoid()
        )

    def forward(self, x):
        # 多尺度特征提取
        feat1 = F.relu(self.conv1(x))  # 1x1卷积
        feat3 = F.relu(self.conv3(x))  # 3x3卷积
        feat5 = F.relu(self.conv5(x))  # 5x5空洞卷积

        # 拼接多尺度特征
        fused = torch.cat([feat1, feat3, feat5], dim=1)

        # 生成注意力权重并确保通道数与输入相同
        att = self.fusion(fused)

        # 将注意力权重均匀分成三部分，分别对应feat1, feat3, feat5
        att_channels = self.channels // self.reduction
        att_chunks = [att[:, i*att_channels:(i+1)*att_channels, :, :] for i in range(3)]  # 手动分割

        # 确保每个注意力权重图的空间尺寸与对应的特征图相匹配
        att_chunks = [F.interpolate(a, size=(f.shape[2], f.shape[3]), mode='bilinear', align_corners=False)
                      for a, f in zip(att_chunks, [feat1, feat3, feat5])]

        # 应用注意力权重到对应的特征图上
        out = sum([a * f for a, f in zip(att_chunks, [feat1, feat3, feat5])])

        # 如果输出的通道数与输入不一致，则调整输出的通道数以匹配输入
        if out.shape[1] != x.shape[1]:
            out = nn.Conv2d(out.shape[1], x.shape[1], kernel_size=1).to(x.device)(out)

        return out + x  # 残差连接

# 修改后的编码器（保持原始结构）
class ResNetEncoder(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base_channels = base_channels

        self.conv1 = nn.Conv2d(3, self.base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = self._modify_resnet_layer(resnet.layer1, self.base_channels)
        self.layer2 = self._modify_resnet_layer(resnet.layer2, self.base_channels * 2)
        self.layer3 = self._modify_resnet_layer(resnet.layer3, self.base_channels * 4)
        self.layer4 = self._modify_resnet_layer(resnet.layer4, self.base_channels * 8)

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
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return [x5, x4, x3, x2, x1, x0]


# 修改后的解码块（加入金字塔注意力）
class PyramidDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, dropout_prob=0.2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # 金字塔注意力模块
        self.pyramid_att = PyramidAttention(skip_channels)

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1, bias=False),
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
            # 应用金字塔注意力到跳跃连接
            skip = self.pyramid_att(skip)

            # 调整尺寸匹配
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)

            # 特征融合
            x = torch.cat([x, skip], dim=1)
            x = self.conv(x)
        x = self.dropout(x)
        return x


# 最终网络结构
class RPNet(nn.Module):
    def __init__(self, n_classes, base_channels=64, dropout_prob=0.5):
        super().__init__()
        self.encoder = ResNetEncoder(base_channels)
        encoder_channels = [base_channels * 8, base_channels * 4,
                            base_channels * 2, base_channels,
                            base_channels, base_channels]

        decoder_channels = [base_channels * 4, base_channels * 2,
                            base_channels, base_channels // 2,
                            base_channels // 4]

        self.decoder_blocks = nn.ModuleList()
        for i in range(len(decoder_channels)):
            in_ch = encoder_channels[i] if i == 0 else decoder_channels[i - 1]
            skip_ch = encoder_channels[i + 1]
            self.decoder_blocks.append(
                PyramidDecoderBlock(in_ch, decoder_channels[i], skip_ch, dropout_prob)
            )

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