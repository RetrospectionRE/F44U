import torch
from torch import nn
from torch.nn import functional as F


class conv_block(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_prob=0):  # 新增 dropout_prob 参数
        super(conv_block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(p=dropout_prob)  # 在卷积块末尾添加 Dropout
        )

    def forward(self, x):
        return self.layer(x)


class downsample(nn.Module):
    def __init__(self, input_channel, dropout_prob=0):  # 新增 dropout_prob 参数
        super(downsample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, input_channel * 2, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.BatchNorm2d(input_channel * 2),
            nn.Dropout2d(p=dropout_prob)  # 下采样后添加 Dropout
        )

    def forward(self, x):
        return self.layer(x)


class upsample(nn.Module):
    def __init__(self, input_channel, output_channel, dropout_prob=0):  # 新增 dropout_prob 参数
        super(upsample, self).__init__()
        self.trans_conv = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_prob)  # 上采样后添加 Dropout

    def forward(self, x):
        x = self.trans_conv(x)
        return self.dropout(x)


class Unet(nn.Module):
    def __init__(self, num_class, base_channels=4, dropout_prob=0):  # 新增 dropout_prob 参数
        super(Unet, self).__init__()
        ch = base_channels

        # Encoder 部分
        self.con1 = conv_block(3, ch, dropout_prob)
        self.down1 = downsample(ch, dropout_prob)

        self.con3 = conv_block(ch * 2, ch * 4, dropout_prob)
        self.down2 = downsample(ch * 4, dropout_prob)

        # 瓶颈层（此处添加额外 Dropout）
        self.bottleneck = nn.Sequential(
            conv_block(ch * 8, ch * 8, dropout_prob),
            nn.Dropout2d(p=dropout_prob * 1.5)  # 增强瓶颈层的正则化
        )

        # Decoder 部分
        self.up1 = upsample(ch * 8, ch * 4, dropout_prob)
        self.con6 = conv_block(ch * 8, ch * 4, dropout_prob)

        self.up2 = upsample(ch * 4, ch * 2, dropout_prob)
        self.con7 = conv_block(ch * 4, ch * 2, dropout_prob)

        self.up3 = upsample(ch * 2, ch, dropout_prob)
        self.con8 = conv_block(ch * 2, ch, dropout_prob)

        self.out = nn.Conv2d(ch, num_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.con1(x)  # [B, 8, H, W]
        e2 = self.down1(e1)  # [B, 16, H/2, W/2]

        e3 = self.con3(e2)  # [B, 32, H/2, W/2]
        e4 = self.down2(e3)  # [B, 64, H/4, W/4]

        e5 = self.bottleneck(e4)  # [B, 64, H/4, W/4]

        # Decoder
        d1 = self.up1(e5)  # [B, 32, H/2, W/2]
        d1 = F.interpolate(d1, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([e3, d1], dim=1)  # [B, 64, H/2, W/2]
        d1 = self.con6(d1)  # [B, 32, H/2, W/2]

        d2 = self.up2(d1)  # [B, 16, H, W]
        d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([e2, d2], dim=1)  # [B, 32, H, W]
        d2 = self.con7(d2)  # [B, 16, H, W]

        d3 = self.up3(d2)  # [B, 8, 2H, 2W]
        d3 = F.interpolate(d3, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([e1, d3], dim=1)  # [B, 16, 256, 256]
        d3 = self.con8(d3)  # [B, 8, 256, 256]

        return self.out(d3)  # [B, 1, 256, 256]


if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    net = Unet(num_class=1, base_channels=4, dropout_prob=0)
    output = net(x)
    print(f"Output shape: {output.shape}")