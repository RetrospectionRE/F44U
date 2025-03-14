import torch
from torch import nn
from torch.nn import functional as F

# 定义卷积块
class conv_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(conv_block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        return self.layer(x)

# 下采样层
class downsample(nn.Module):
    def __init__(self, input_channel):
        super(downsample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, input_channel * 2, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),  # 增加通道数
            nn.LeakyReLU(),
            nn.BatchNorm2d(input_channel * 2)
        )

    def forward(self, x):
        return self.layer(x)

# 上采样层
class upsample(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(upsample, self).__init__()
        # 使用转置卷积进行上采样，并使用 output_padding 确保尺寸匹配
        self.trans_conv = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.trans_conv(x)
        return x

# ASPP 模块定义
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_3x3_rate6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[0], dilation=rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_3x3_rate12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[1], dilation=rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_3x3_rate18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[2], dilation=rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1)

    def forward(self, x):
        size = x.shape[2:]

        feat1 = self.conv_1x1(x)
        feat2 = self.conv_3x3_rate6(x)
        feat3 = self.conv_3x3_rate12(x)
        feat4 = self.conv_3x3_rate18(x)
        feat5 = self.global_avg_pool(x)
        feat5 = F.interpolate(feat5, size=size, mode='bilinear', align_corners=True)

        out = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        out = self.conv_out(out)

        return out

# 修改 Unet 类以包含 ASPP 模块
class AUnet(nn.Module):
    def __init__(self, num_class, base_channels=8):  # 设置基础通道数为 32
        super(AUnet, self).__init__()
        ch = base_channels

        # Encoder 部分 - 减少了下采样的次数
        self.con1 = conv_block(3, ch)
        self.down1 = downsample(ch)  # 下采样后通道数变为 ch*2

        self.con3 = conv_block(ch*2, ch*4)  # 接受 ch*2 输入，输出 ch*4
        self.down2 = downsample(ch*4)  # 下采样后通道数变为 ch*8

        self.aspp = ASPP(ch*8, ch*8)  # 在瓶颈层添加 ASPP 模块

        # Decoder 部分 - 减少了上采样的次数
        self.up1 = upsample(ch*8, ch*4)
        self.con6 = conv_block(ch*8, ch*4)  # 接受来自 e3 和 d1 的跳跃连接，总共有 ch*8 输入，输出 ch*4

        self.up2 = upsample(ch*4, ch*2)
        self.con7 = conv_block(ch*4, ch*2)  # 接受来自 e2 和 d2 的跳跃连接，总共有 ch*4 输入，输出 ch*2

        self.up3 = upsample(ch*2, ch)
        self.con8 = conv_block(ch*2, ch)  # 接受来自 e1 和 d3 的跳跃连接，总共有 ch*2 输入，输出 ch

        self.out = nn.Conv2d(in_channels=ch, out_channels=num_class, kernel_size=1)  # 使用1x1卷积核来减少参数

    def forward(self, x):
        # Encoder 部分
        e1 = self.con1(x)
        e2 = self.down1(e1)

        e3 = self.con3(e2)
        e4 = self.down2(e3)

        # 使用 ASPP 模块处理编码器的最后一层
        e5 = self.aspp(e4)

        # Decoder 部分
        d1 = self.up1(e5)
        # 确保 d1 和 e3 尺寸匹配
        d1 = F.interpolate(d1, size=e3.size()[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat((e3, d1), dim=1)
        d1 = self.con6(d1)

        d2 = self.up2(d1)
        # 确保 d2 和 e2 尺寸匹配
        d2 = F.interpolate(d2, size=e2.size()[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.con7(d2)

        d3 = self.up3(d2)
        # 确保 d3 和 e1 尺寸匹配
        d3 = F.interpolate(d3, size=e1.size()[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat((e1, d3), dim=1)
        d3 = self.con8(d3)

        output = self.out(d3)
        return output

if __name__ == '__main__':
    x = torch.randn(8, 3, 256, 256)
    net = AUnet(num_class=1, base_channels=8)  # 设置基础通道数为 32
    output = net(x)
    print(f"Output shape: {output.shape}")  # 检查输出形状