# losses.py
# 这里定义了训练模型使用的损失函数

import torch
import torch.nn as nn
import torch.nn.functional as F

# 二值交叉熵损失函数
"""
二值交叉熵损失函数
    交叉熵能够衡量同一个随机变量中的两个不同概率分布的差异程度，在机器学习中就表示为真实概率分布与预测概率分布之间的差异。交叉熵的值越小，模型预测效果就越好。
    交叉熵在分类问题中常常与softmax是标配，softmax将输出的结果进行处理，使其多个分类的预测值和为1，再通过交叉熵来计算损失。
"""
class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, input, target):
        return F.binary_cross_entropy_with_logits(input, target)

# Dice损失函数

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1e-5 # 防止除数为0
        input = torch.sigmoid(input)
        num = target.size(0)

        input_1 = input[:,0,:,:]
        input_2 = input[:,1,:,:]
        target_1 = target[:,0,:,:]
        target_2 = target[:,1,:,:]

        input_1 = input_1.view(num, -1)
        input_2 = input_2.view(num, -1)

        target_1 = target_1.view(num, -1)
        target_2 = target_2.view(num, -1)

        intersection_1 = (input_1 * target_1)
        intersection_2 = (input_2 * target_2)

        dice_1 = (2. * intersection_1.sum(1) + smooth) / (input_1.sum(1) + target_1.sum(1) + smooth)
        dice_2 = (2. * intersection_2.sum(1) + smooth) / (input_2.sum(1) + target_2.sum(1) + smooth)

        dice_1 = 1 - dice_1.sum() / num
        dice_2 = 1 - dice_2.sum() / num

        # dice = (dice_1+dice_2)/2.0
        return dice_1

# 混合损失函数
"""
    交叉熵损失函数+dice损失函数
"""
class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()

    def forward(self, input, target):
        # input = torch.sigmoid(input)
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)

        input_1 = input[:,0,:,:]
        input_2 = input[:,1,:,:]
        target_1 = target[:,0,:,:]
        target_2 = target[:,1,:,:]

        input_1 = input_1.view(num, -1)
        input_2 = input_2.view(num, -1)

        target_1 = target_1.view(num, -1)
        target_2 = target_2.view(num, -1)

        intersection_1 = (input_1 * target_1)
        intersection_2 = (input_2 * target_2)

        dice_1 = (2. * intersection_1.sum(1) + smooth) / (input_1.sum(1) + target_1.sum(1) + smooth)
        dice_2 = (2. * intersection_2.sum(1) + smooth) / (input_2.sum(1) + target_2.sum(1) + smooth)

        dice_1 = 1 - dice_1.sum() / num
        dice_2 = 1 - dice_2.sum() / num

        dice = (dice_1+dice_2)/2.0
        return 0.5 * bce + dice