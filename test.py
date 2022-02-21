# test.py
# 3.测试和评估（运行 ./test.py） 
# 用网络预测测试集的标签，保存到./data/pred_seg，并将每次预测的结果和金标准比对，产生的评价指标结果存入excel

import os
import torch
import copy
import collections
import datetime
import pandas as pd
import numpy as np 
from time import time
from tqdm import tqdm 
from net.UNet import U_Net
from utilities.metrics import Metirc
import skimage.measure as measure
import skimage.morphology as morphology

import parameter as para
# 读取医学图像信息软件包
import SimpleITK as sitk 
# 为了计算dice_global定义的两个变量
liver_dice_intersection = 0.0  
liver_dice_union = 0.0

tumor_dice_intersection = 0.0  
tumor_dice_union = 0.0

file_name = []  # 文件名称
time_pre_case = []  # 单例数据消耗时间

# 定义评价指标
liver_score = collections.OrderedDict()
liver_score['dice'] = []
liver_score['jacard'] = []
liver_score['voe'] = []
liver_score['fnr'] = []
liver_score['fpr'] = []
liver_score['assd'] = []
liver_score['rmsd'] = []
liver_score['msd'] = []

tumor_score = collections.OrderedDict()
tumor_score['dice'] = []
tumor_score['jacard'] = []
tumor_score['voe'] = []
tumor_score['fnr'] = []
tumor_score['fpr'] = []
tumor_score['assd'] = []
tumor_score['rmsd'] = []
tumor_score['msd'] = []

# 对肝脏进行最大连通域提取,移除细小区域,并进行内部的空洞填充
def post_process(pred_seg,ct, file):
    liver_seg = copy.deepcopy(pred_seg)
    liver_seg[liver_seg>0] =  1 # 全变成肝脏标签
    # print(liver_seg)
    liver_seg = measure.label(liver_seg, 4)
    props = measure.regionprops(liver_seg)
    
    max_area = 0
    max_index = 0
    for index, prop in enumerate(props, start=1):
        if prop.area > max_area:
            max_area = prop.area
            max_index = index
    
    liver_seg[liver_seg != max_index] = 0
    liver_seg[liver_seg == max_index] = 1
    
    liver_seg = liver_seg.astype(np.bool)
    morphology.remove_small_holes(liver_seg, para.maximum_hole, connectivity=2, in_place=True)
    liver_seg = liver_seg.astype(np.uint8)
    # print(liver_seg)
    # 转换格式，写入文件
    liver_seg = sitk.GetImageFromArray(liver_seg)
    liver_seg.SetDirection(ct.GetDirection())
    liver_seg.SetOrigin(ct.GetOrigin())
    liver_seg.SetSpacing(ct.GetSpacing())
    
    sitk.WriteImage(liver_seg, os.path.join(para.pred_seg_path, file.replace('volume', 'liver_segmentation')))

# 网络模型修改这里：
# 定义网络并加载参数，training=False表示网络进行测试时
net = torch.nn.DataParallel(U_Net(training=False)).cuda()
net.load_state_dict(torch.load(para.model))
# model.eval()是保证BN用全部训练数据的均值和方差
net.eval()

# 如果没有预测结果和评价指标存放的文件夹，则创建
if not os.path.exists(para.pred_seg_path):
    os.mkdir(para.pred_seg_path)
if not os.path.exists(para.result_path):
    os.mkdir(para.result_path)

print('测试样本总数为',len(os.listdir(para.test_ct_path)))
# 遍历测试集的每一个ct图像的样本，依次获取预测的分割结果
for file_index, file in enumerate(os.listdir(para.test_ct_path)):
    if file_index ==0:
        continue
    # 获取系统时间
    start = time()
    file_name.append(file)
    # --------------------------------产生预测图像---------------------------------
    print("预测图像分割结果，当前测试样本：%s" % file)
    # ReadImage()获取原始CT图像及金标准seg数据,并读入内存
    ct = sitk.ReadImage(os.path.join(para.test_ct_path, file), sitk.sitkInt16)
    # GetArrayFromeImage()可用于将SimpleITK对象转换为ndarray数组
    ct_array = sitk.GetArrayFromImage(ct)
    # 将灰度值在阈值之外的截断掉（窗位：CT值中心，窗宽：CT值范围。这里窗宽为upper-lower）
    ct_array[ct_array > para.upper] = para.upper
    ct_array[ct_array < para.lower] = para.lower

    # 【归一化】：将肝脏区域设为（0，1）之间的float值
    ct_array = ct_array.astype(np.float32)
    # ct_array = ct_array/200
    ct_array = (ct_array - para.lower) / (para.upper - para.lower)

    # 【裁剪】：只取带肝脏和病灶的部分
    # 继续裁剪每张切片(偶数才行) 448*448 (512*512,选取中间448*448的部分)
    # ct_crop = ct_crop[:,32:480,32:480]
    ct_crop = ct_array[:,para.crop_start:para.crop_end,para.crop_start:para.crop_end]
  
    # 初始化一个和CT图像相同的全0数组，用于保存预测标签
    slice_predictions = np.zeros((ct_array.shape[0],512,512),dtype=np.int16)

    # 被with torch.no_grad()包住的代码，不用跟踪反向梯度计算
    """
        with torch.no_grad()
        在使用pytorch时，并不是所有的操作都需要进行计算图的生成（计算过程的构建，以便梯度反向传播等操作）。而对于tensor的计算操作，默认是要进行计算图的构建的，
        在这种情况下，可以使用 with torch.no_grad():，强制之后的内容不进行计算图构建
    """
    with torch.no_grad():
        # 遍历ct体积内的每一个切片的2d图像
        for i,n_slice in enumerate(tqdm(range(ct_crop.shape[0]-3))):
            # 把ct体积内的相邻3个切片转换成张量
            model_input = ct_crop[n_slice:n_slice+3]
            ct_tensor = torch.FloatTensor(model_input).cuda()
            # 这个函数主要是对数据维度进行扩充, 给指定位置加上维数为1的维度。
            """
            unsqueeze
                这一功能尤其在神经网络输入单个样本时很有用
                由于pytorch神经网络要求的输入都是mini-batch型的，维度为[batch_size, channels, w, h]，
                而一个样本的维度为[c, w, h]，此时用unsqueeze()增加一个维度变为[1, c, w, h]就很方便了。
            """
            ct_tensor = ct_tensor.unsqueeze(dim=0)
            # 模型输入一个样本的tensor，产生预测结果
            output = net(ct_tensor)
            # sigmoid把输出值处理成0-1之间的像素点的概率值，从而可用阈值0.5分离标签
            output = torch.sigmoid(output).data.cpu().numpy()
            # 初始化一个全0矩阵保存标签
            probability_map = np.zeros([1, 448, 448], dtype=np.uint8)
            # print('output',output.shape)
            # 预测值拼接回去
            for idz in range(output.shape[1]):
                for idx in range(output.shape[2]):
                    for idy in range(output.shape[3]):
                        # 肝脏标签为1
                        if (output[0,0, idx, idy] > 0.5):
                            probability_map[0, idx, idy] = 1 
                        # 肿瘤标签为2       
                        if (output[0,1, idx, idy] > 0.5):
                            probability_map[0, idx, idy] = 2

            # 标签存到CT体积中
            slice_predictions[n_slice,32:480,32:480] = probability_map   

        # 将数组转化为Image格式保存到文件
        pred_seg = slice_predictions
        pred_seg = pred_seg.astype(np.uint8)
        
        # 后处理：对肝脏进行最大连通域提取,移除细小区域,并进行内部的空洞填充
        # if file_index!=0:
        #     post_process(pred_seg,ct,file)
                    
        # 转换格式，写入文件
        pred_seg = sitk.GetImageFromArray(pred_seg)
        pred_seg.SetDirection(ct.GetDirection())
        pred_seg.SetOrigin(ct.GetOrigin())
        pred_seg.SetSpacing(ct.GetSpacing())

        sitk.WriteImage(pred_seg, os.path.join(para.pred_seg_path, file.replace('volume', 'segmentation')))
        # 释放torch占用的缓存
        torch.cuda.empty_cache()

# --------------------------------评估预测结果---------------------------------
    # 遍历测试集的每一个ct图像的样本，依次生成评价指标
    # 将金标准读入内存
    seg = sitk.ReadImage(os.path.join(para.test_seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    
    # 将预测读入内存
    pred_seg = sitk.ReadImage(os.path.join(para.pred_seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
    pred_seg_array = sitk.GetArrayFromImage(pred_seg)

    # 计算分割评价指标：
    # 肝脏分割指标计算
    liver_seg_array = copy.deepcopy(seg_array)  # 金标准
    liver_seg = copy.deepcopy(pred_seg_array)   # 预测值
    liver_seg_array[liver_seg_array > 0] = 1  # 肝脏金标准
    liver_seg[liver_seg > 0] = 1              # 肝脏预测标签


    liver_metric = Metirc(liver_seg_array, liver_seg, ct.GetSpacing())

    liver_score['dice'].append(liver_metric.get_dice_coefficient()[0])
    liver_score['jacard'].append(liver_metric.get_jaccard_index())
    liver_score['voe'].append(liver_metric.get_VOE())
    liver_score['fnr'].append(liver_metric.get_FNR())
    liver_score['fpr'].append(liver_metric.get_FPR())
    liver_score['assd'].append(liver_metric.get_ASSD())
    liver_score['rmsd'].append(liver_metric.get_RMSD())
    liver_score['msd'].append(liver_metric.get_MSD())

    liver_dice_intersection += liver_metric.get_dice_coefficient()[1]
    liver_dice_union += liver_metric.get_dice_coefficient()[2]
    print(file, "肝脏预测评估完成")

    # 肿瘤分割指标计算
    tumor_seg_array = copy.deepcopy(seg_array)  # 金标准
    tumor_seg = copy.deepcopy(pred_seg_array)   # 预测值
    tumor_seg_array[tumor_seg_array==1] = 0 # 肿瘤金标准
    tumor_seg_array[tumor_seg_array>1] = 1
    tumor_seg[tumor_seg==1] = 0             # 肿瘤预测标签
    tumor_seg[tumor_seg>1] = 1

    tumor_metric = Metirc(tumor_seg_array, tumor_seg, ct.GetSpacing())

    tumor_score['dice'].append(tumor_metric.get_dice_coefficient()[0])
    tumor_score['jacard'].append(tumor_metric.get_jaccard_index())
    tumor_score['voe'].append(tumor_metric.get_VOE())
    tumor_score['fnr'].append(tumor_metric.get_FNR())
    tumor_score['fpr'].append(tumor_metric.get_FPR())
    tumor_score['assd'].append(tumor_metric.get_ASSD())
    tumor_score['rmsd'].append(tumor_metric.get_RMSD())
    tumor_score['msd'].append(tumor_metric.get_MSD())

    tumor_dice_intersection += tumor_metric.get_dice_coefficient()[1]
    tumor_dice_union += tumor_metric.get_dice_coefficient()[2]
    print(file, "肿瘤预测评估完成")

    # 计时
    speed = time() - start
    time_pre_case.append(speed)

    print(file, 'this case use {:.3f} s'.format(speed))
    print('-----------------------')

# 将评价指标写入到exel中：
timestamp  = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
result_path = '{}/result_{}_{}.xlsx'.format(para.result_path, para.model_name, timestamp)
print("评价指标写入excel文件：", result_path)

# 肝脏评价指标写入excel
liver_data = pd.DataFrame(liver_score, index=file_name)
liver_data['time'] = time_pre_case

liver_statistics = pd.DataFrame(index=['mean', 'std', 'min', 'max'], columns=list(liver_data.columns))
liver_statistics.loc['mean'] = liver_data.mean()
liver_statistics.loc['std'] = liver_data.std()
liver_statistics.loc['min'] = liver_data.min()
liver_statistics.loc['max'] = liver_data.max()

writer = pd.ExcelWriter(result_path)
# print(os.path.join(para.result_path, 'result.xlsx'))
liver_data.to_excel(writer, 'liver')
liver_statistics.to_excel(writer, 'liver_statistics')

# 肿瘤评价指标写入excel
tumor_data = pd.DataFrame(tumor_score, index=file_name)
tumor_data['time'] = time_pre_case

tumor_statistics = pd.DataFrame(index=['mean', 'std', 'min', 'max'], columns=list(tumor_data.columns))
tumor_statistics.loc['mean'] = tumor_data.mean()
tumor_statistics.loc['std'] = tumor_data.std()
tumor_statistics.loc['min'] = tumor_data.min()
tumor_statistics.loc['max'] = tumor_data.max()

tumor_data.to_excel(writer, 'tumor')
tumor_statistics.to_excel(writer, 'tumor_statistics')

writer.save()

# 打印dice global
if liver_dice_union != 0:
    print('liver dice global:', liver_dice_intersection / liver_dice_union)
if tumor_dice_union != 0:
    print('tumor dice global:', tumor_dice_intersection / tumor_dice_union)
