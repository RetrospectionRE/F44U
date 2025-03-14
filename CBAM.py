import torch
from torch import nn
from torch.nn import functional as F


class CBAM(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CBAM, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv_after_concat = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        module_input = x

        # Channel Attention
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        ca_weights = self.sigmoid_channel(out)
        x = module_input * ca_weights

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        h = torch.cat([avg_out, max_out], dim=1)
        sa_weights = self.sigmoid_spatial(self.conv_after_concat(h))
        x = x * sa_weights

        return x


class conv_block(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.2):  # 添加 dropout_rate 参数
        super(conv_block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1,
                      padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channel),
            CBAM(out_channel),  # 添加CBAM
            nn.Dropout2d(p=dropout_rate)  # 在 CBAM 后面添加 Dropout2d
        )
        if in_channel != out_channel:
            self.residual_connection = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        else:
            self.residual_connection = nn.Identity()

    def forward(self, x):
        identity = self.residual_connection(x)
        out = self.layer(x)
        return out + identity  # 残差连接


class downsample(nn.Module):
    def __init__(self, input_channel):
        super(downsample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, input_channel * 2, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
            # 增加通道数
            nn.LeakyReLU(),
            nn.BatchNorm2d(input_channel * 2)
        )

    def forward(self, x):
        return self.layer(x)


class upsample(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(upsample, self).__init__()
        self.trans_conv = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.trans_conv(x)
        return x


class CBAMUnet(nn.Module):
    def __init__(self, num_class, base_channels=4):  # 设置基础通道数为 32
        super(CBAMUnet, self).__init__()
        ch = base_channels

        # Encoder 部分 - 减少了下采样的次数
        self.con1 = conv_block(3, ch)
        self.down1 = downsample(ch)  # 下采样后通道数变为 ch*2

        self.con3 = conv_block(ch * 2, ch * 4)  # 接受 ch*2 输入，输出 ch*4
        self.down2 = downsample(ch * 4)  # 下采样后通道数变为 ch*8

        self.con5 = conv_block(ch * 8, ch * 8)  # 接受 ch*8 输入，输出 ch*8

        # Decoder 部分 - 减少了上采样的次数
        self.up1 = upsample(ch * 8, ch * 4)
        self.con6 = conv_block(ch * 8, ch * 4)  # 接受来自 e3 和 d1 的跳跃连接，总共有 ch*8 输入，输出 ch*4

        self.up2 = upsample(ch * 4, ch * 2)
        self.con7 = conv_block(ch * 4, ch * 2)  # 接受来自 e2 和 d2 的跳跃连接，总共有 ch*4 输入，输出 ch*2

        self.up3 = upsample(ch * 2, ch)
        self.con8 = conv_block(ch * 2, ch)  # 接受来自 e1 和 d3 的跳跃连接，总共有 ch*2 输入，输出 ch

        self.out = nn.Conv2d(in_channels=ch, out_channels=num_class, kernel_size=1)  # 使用1x1卷积核来减少参数

    def forward(self, x):
        # Encoder 部分
        e1 = self.con1(x)
        e2 = self.down1(e1)

        e3 = self.con3(e2)
        e4 = self.down2(e3)

        e5 = self.con5(e4)

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
    x = torch.randn(1, 3, 256, 256)
    net = CBAMUnet(num_class=1, base_channels=4)  # 设置基础通道数为 32
    output = net(x)
    print(f"Output shape: {output.shape}")  # 检查输出形状