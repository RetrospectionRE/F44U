import torch
from torch import nn
from torch.nn import functional as F

class conv_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(conv_block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        return self.layer(x)

class downsample(nn.Module):
    def __init__(self, input_channel):
        super(downsample, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(input_channel, input_channel * 2, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),  # 增加通道数
            nn.LeakyReLU(),
            nn.BatchNorm2d(input_channel * 2)
        )

    def forward(self, x):
        return self.layer(x)

class upsample(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(upsample, self).__init__()
        # 使用转置卷积进行上采样，并使用 output_padding 确保尺寸匹配
        self.trans_conv = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.trans_conv(x)
        return x

class PUnet(nn.Module):
    def __init__(self, num_class, base_channels=32):  # 设置基础通道数为 32
        super(PUnet, self).__init__()
        ch = base_channels

        # Encoder 部分
        self.x_00 = conv_block(3, ch)
        self.x_10 = conv_block(ch, ch*2)
        self.x_20 = conv_block(ch*2, ch*4)
        self.x_30 = conv_block(ch*4, ch*8)

        # Decoder 部分，包括额外的跳跃连接
        self.x_01 = conv_block(ch*2, ch)
        self.x_11 = conv_block(ch*4, ch*2)
        self.x_21 = conv_block(ch*8, ch*4)

        self.x_02 = conv_block(ch*3, ch)
        self.x_12 = conv_block(ch*6, ch*2)

        self.x_03 = conv_block(ch*4, ch)

        # 最终输出层
        self.out = nn.Conv2d(in_channels=ch, out_channels=num_class, kernel_size=1)

        # 上采样操作
        self.up_10_to_01 = upsample(ch*2, ch)
        self.up_20_to_11 = upsample(ch*4, ch*2)
        self.up_30_to_21 = upsample(ch*8, ch*4)
        self.up_11_to_02 = upsample(ch*2, ch)
        self.up_21_to_12 = upsample(ch*4, ch*2)
        self.up_22_to_03 = upsample(ch*2, ch)

    def forward(self, x):
        # Encoder 部分
        x_00 = self.x_00(x)
        x_10 = self.x_10(F.max_pool2d(x_00, kernel_size=2, stride=2))
        x_20 = self.x_20(F.max_pool2d(x_10, kernel_size=2, stride=2))
        x_30 = self.x_30(F.max_pool2d(x_20, kernel_size=2, stride=2))

        # Decoder 部分，包含额外的跳跃连接
        x_01 = self.x_01(torch.cat([x_00, self.up_10_to_01(x_10)], dim=1))
        x_11 = self.x_11(torch.cat([x_10, self.up_20_to_11(x_20)], dim=1))
        x_21 = self.x_21(torch.cat([x_20, self.up_30_to_21(x_30)], dim=1))

        x_02 = self.x_02(torch.cat([x_00, x_01, self.up_11_to_02(x_11)], dim=1))
        x_12 = self.x_12(torch.cat([x_10, x_11, self.up_21_to_12(x_21)], dim=1))

        x_03 = self.x_03(torch.cat([x_00, x_01, x_02, self.up_22_to_03(x_12)], dim=1))

        output = self.out(x_03)
        return output

if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    net = PUnet(num_class=1, base_channels=32)  # 设置基础通道数为 32
    output = net(x)
    print(f"Output shape: {output.shape}")  # 检查输出形状