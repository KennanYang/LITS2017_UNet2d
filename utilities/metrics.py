# metrics.py
# 这里定义了评价矩阵，通过一系列指标评估模型的好坏
import torch

import math

import numpy as np
import scipy.spatial as spatial
import scipy.ndimage.morphology as morphology

# 定义了iou评价指标
def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


# 定义了dice评价指标
def dice_coef(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    input_1 = output[:,0,:,:]
    input_2 = output[:,1,:,:]
    target_1 = target[:,0,:,:]
    target_2 = target[:,1,:,:]

    input_1[input_1>0.5] = 1
    input_1[input_1<0.5] = 0
    input_2[input_2>0.5] = 1
    input_2[input_2<0.5] = 0

    intersection_1 = (input_1 * target_1)
    intersection_2 = (input_2 * target_2)

    dice_1 = (2. * intersection_1.sum() + smooth) / (input_1.sum() + target_1.sum() + smooth)
    dice_2 = (2. * intersection_2.sum() + smooth) / (input_2.sum() + target_2.sum() + smooth)
    
    # print('dice_1',dice_1)
    # print('dice_2',dice_2)
    return dice_1,dice_2


"""
计算基于重叠度和距离等九种分割常见评价指标
"""


class Metirc():
    
    def __init__(self, real_mask, pred_mask, voxel_spacing):
        """

        :param real_mask: 金标准
        :param pred_mask: 预测结果
        :param voxel_spacing: 体数据的spacing
        """
        self.real_mask = real_mask
        self.pred_mask = pred_mask
        self.voxel_sapcing = voxel_spacing

        self.real_mask_surface_pts = self.get_surface(real_mask, voxel_spacing)
        self.pred_mask_surface_pts = self.get_surface(pred_mask, voxel_spacing)

        self.real2pred_nn = self.get_real2pred_nn()
        self.pred2real_nn = self.get_pred2real_nn()

    # 下面三个是提取边界和计算最小距离的实用函数
    def get_surface(self, mask, voxel_spacing):
        """

        :param mask: ndarray
        :param voxel_spacing: 体数据的spacing
        :return: 提取array的表面点的真实坐标(以mm为单位)
        """

        # 卷积核采用的是三维18邻域

        kernel = morphology.generate_binary_structure(3, 2)
        surface = morphology.binary_erosion(mask, kernel) ^ mask
        # np.nonzero函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数
        surface_pts = surface.nonzero()
        surface_pts = np.array(list(zip(surface_pts[0], surface_pts[1], surface_pts[2])))
        
        if surface_pts.size == 0:
            surface_pts = [[0,0,0]]
        
        # print(surface_pts) # todo: 这里当遇到没有标签值的空数组时会报错
        # print(np.array(self.voxel_sapcing[::-1]).reshape(1, 3))
        
        # (0.7808688879013062, 0.7808688879013062, 2.5) (88, 410, 512)
        # # 读出来的数据spacing和shape不是对应的,所以需要反向
        return surface_pts * np.array(self.voxel_sapcing[::-1]).reshape(1, 3)

    def get_pred2real_nn(self):
        """

        :return: 预测结果表面体素到金标准表面体素的最小距离
        """

        tree = spatial.cKDTree(self.real_mask_surface_pts)
        nn, _ = tree.query(self.pred_mask_surface_pts)

        return nn

    def get_real2pred_nn(self):
        """

        :return: 金标准表面体素到预测结果表面体素的最小距离
        """
        tree = spatial.cKDTree(self.pred_mask_surface_pts)
        nn, _ = tree.query(self.real_mask_surface_pts)

        return nn

    # 下面的六个指标是基于重叠度的
    def get_dice_coefficient(self):
        """

        :return: dice系数 dice系数的分子 dice系数的分母(后两者用于计算dice_global)
        """
        
        smooth = 1e-5
        intersection = (self.real_mask * self.pred_mask).sum()
        union = self.real_mask.sum() + self.pred_mask.sum()
        dice = (2 * intersection + smooth) / (union + smooth)
        return dice, 2 * intersection, union

    def get_jaccard_index(self):
        """

        :return: 杰卡德系数
        """
        intersection = (self.real_mask * self.pred_mask).sum()
        union = (self.real_mask | self.pred_mask).sum()

        return intersection / union

    def get_VOE(self):
        """

        :return: 体素重叠误差 Volumetric Overlap Error
        """

        return 1 - self.get_jaccard_index()

    def get_RVD(self):
        """

        :return: 体素相对误差 Relative Volume Difference
        """

        return float(self.pred_mask.sum() - self.real_mask.sum()) / float(self.real_mask.sum())

    def get_FNR(self):
        """

        :return: 欠分割率 False negative rate
        """
        fn = self.real_mask.sum() - (self.real_mask * self.pred_mask).sum()
        union = (self.real_mask | self.pred_mask).sum()

        return fn / union

    def get_FPR(self):
        """

        :return: 过分割率 False positive rate
        """
        fp = self.pred_mask.sum() - (self.real_mask * self.pred_mask).sum()
        union = (self.real_mask | self.pred_mask).sum()

        return fp / union

    # 下面的三个指标是基于距离的
    def get_ASSD(self):
        """

        :return: 对称位置平均表面距离 Average Symmetric Surface Distance
        """
        return (self.pred2real_nn.sum() + self.real2pred_nn.sum()) / \
               (self.real_mask_surface_pts.shape[0] + self.pred_mask_surface_pts.shape[0])

    def get_RMSD(self):
        """

        :return: 对称位置表面距离的均方根 Root Mean Square symmetric Surface Distance
        """
        return math.sqrt((np.power(self.pred2real_nn, 2).sum() + np.power(self.real2pred_nn, 2).sum()) /
                         (self.real_mask_surface_pts.shape[0] + self.pred_mask_surface_pts.shape[0]))

    def get_MSD(self):
        """

        :return: 对称位置的最大表面距离 Maximum Symmetric Surface Distance
        """
        return max(self.pred2real_nn.max(), self.real2pred_nn.max())
