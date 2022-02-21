# UNet.py
# 这里定义了UNet网络结构，用来进行2d的医学图像分割

import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])
import parameter as para

# 卷积块
class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(            
            # 进行2d的卷积操作
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            
            # 进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.BatchNorm2d(out_ch),

            # 这个函数的作用是计算激活函数 relu，即 max(features, 0)。将大于0的保持不变，小于0的数置为0。
            # 参数inplace=True：inplace为True，将会改变输入的数据，否则不会改变原输入，只会产生新的输出
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x

# 上卷积，继承了nn.Module类，也就是进行上采样，还原图片大小
class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            # 上采样，默认为 最近邻元法：在待求象素的四邻象素中，将距离待求象素最近的邻象素灰度赋给待求象素
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

# 继承nn.Module类，重写模型的方法
class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, training):
        # 首先找到Net的父类（比如是类NNet），
        # 然后把类Net的对象self转换为类NNet的对象，然后“被转换”的类NNet对象调用自己的init函数
        super(U_Net, self).__init__()

        self.training = training

        # 通道数设置，输入448*448*3的相邻三层切片图像
        in_ch = 3
        # 输出通道为2
        # 有多少个需要识别的物体，就有多少个输出channel，最后再做一个叠加就是最终我们想分割的结果。
        out_ch = 2
        
        # 通道数设置，初始为32，每次通道数翻倍
        n1 = 32
        channels = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # 4次最大池化层
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 5次下卷积
        self.Conv1 = conv_block(in_ch, channels[0])
        self.Conv2 = conv_block(channels[0], channels[1])
        self.Conv3 = conv_block(channels[1], channels[2])
        self.Conv4 = conv_block(channels[2], channels[3])
        self.Conv5 = conv_block(channels[3], channels[4])

        # 5次上采样+卷积
        self.Up5 = up_conv(channels[4], channels[3])
        self.Up_conv5 = conv_block(channels[4], channels[3])

        self.Up4 = up_conv(channels[3], channels[2])
        self.Up_conv4 = conv_block(channels[3], channels[2])

        self.Up3 = up_conv(channels[2], channels[1])
        self.Up_conv3 = conv_block(channels[2], channels[1])

        self.Up2 = up_conv(channels[1], channels[0])
        self.Up_conv2 = conv_block(channels[1], channels[0])

        # 最后一层，使用1*1卷积将64个分量特征向量映射到所需的类的数量，这里是2个类，肝脏和肿瘤
        self.Conv = nn.Conv2d(channels[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    # 前向传播
    def forward(self, x):
        # print('x',x.shape)
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        # pytorch的cat()，进行两层特征图的拼接
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        # 输出分割结果
        out = self.Conv(d2)

        #d1 = self.active(out)

        return out


# 在init中通过判断模块的类型来进行不同的参数初始化定义类型
def init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, 0.25)
        nn.init.constant_(module.bias.data, 0)

# training=True表示网络进行训练时
net = U_Net(training=True)
# 使用apply函数的方式进行初始化
net.apply(init)

# 计算网络参数
net_total_para = sum(param.numel() for param in net.parameters())
print('Run U_Net， total parameters:', net_total_para)