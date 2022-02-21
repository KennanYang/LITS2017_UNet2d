# preprocess.py
# 1.预处理（运行./data_prepare/preprocess.py） 将原始数据集处理成用于训练网络模型的训练集

       
import os
# sys模块提供了一系列有关Python运行环境的变量和函数
"""
sys.path 返回的是一个列表！
    该路径已经添加到系统的环境变量了，当我们要添加自己的搜索目录时，可以通过列表的append()方法；
    对于模块和自己写的脚本不在同一个目录下，在脚本开头加sys.path.append('xxx')：
    (如果没有这个方法，将不能调用parameter)
"""
import sys
sys.path.append(os.path.split(sys.path[0])[0])
from time import time
import numpy as np
# tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，
# 用户只需要封装任意的迭代器 tqdm(iterator)。
from tqdm import tqdm 
# 读取医学图像信息软件包
import SimpleITK as sitk 
# 导入自定义参数
import parameter as para 

# 如果没有预处理训练集后存放的文件夹，则创建
if not os.path.exists(para.train_ct_path):
    os.mkdir(para.train_ct_path)
if not os.path.exists(para.train_seg_path):
    os.mkdir(para.train_seg_path)

# 计算运行时长
start = time()
# 遍历原始训练集的每一个CT体积文件,并输出进度信息(tqdm)
# enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
for index,file in enumerate(tqdm(os.listdir(para.lits_train_ct_path))):
    # ReadImage()获取原始CT图像及金标准seg数据,并读入内存
    ct = sitk.ReadImage(os.path.join(para.lits_train_ct_path, file), sitk.sitkInt16)
    seg = sitk.ReadImage(os.path.join(para.lits_train_seg_path,file.replace('volume', 'segmentation')), sitk.sitkUInt8)

    # GetArrayFromeImage()可用于将SimpleITK对象转换为ndarray数组
    ct_array = sitk.GetArrayFromImage(ct)
    seg_array = sitk.GetArrayFromImage(seg)
    
    # 肝脏和肿瘤标签值设置（0表示背景，1表示肝脏，2表示肿瘤)
    # seg_array[seg_array == 1] = 0  # 肿瘤
    # seg_array[seg_array == 2] = 1

    # 将灰度值在阈值之外的截断掉（窗位：CT值中心，窗宽：CT值范围。这里窗宽为upper-lower）
    ct_array[ct_array > para.upper] = para.upper
    ct_array[ct_array < para.lower] = para.lower

    # 【归一化】：将肝脏区域设为（0，1）之间的float值
    ct_array = ct_array.astype(np.float32)
    # ct_array = ct_array/200
    ct_array = (ct_array - para.lower) / (para.upper - para.lower)

    # 找到金标准的肝脏区域开始和结束的slice（范围在0~1），并各向外扩张slice
    z = np.any(seg_array, axis=(1, 2))
    start_slice, end_slice = np.where(z)[0][[0, -1]] 

    # ---------------------------------2D分割预处理---------------------------------  
    # 2d图像分割：
    # 取出包含肝脏和肿瘤金标准的切片，逐层保留为单独的图片样本，格式为npy格式，命名规则为“ct编号_切片编号.npy”
    # 如 1_0.npy 表示第一个CT体积的包含肝脏病灶的第一个切片

    # 【裁剪】：只取带肝脏和病灶的部分
    # 取包含肝脏病灶的切片的起始索引
    start_slice = max(0, start_slice - 1)
    end_slice = min(seg_array.shape[0] - 1, end_slice + 2)

    # 纵向裁剪切片,只保留带病灶的部分
    ct_crop = ct_array[start_slice:end_slice, :, :]
    seg_crop = seg_array[start_slice+1: end_slice-1,  :, :]

    # 继续裁剪每张切片(偶数才行) 448*448 (512*512,选取中间448*448的部分)
    # ct_crop = ct_crop[:,32:480,32:480]
    ct_crop = ct_crop[:,para.crop_start:para.crop_end,para.crop_start:para.crop_end]
    seg_crop = seg_crop[:,para.crop_start:para.crop_end,para.crop_start:para.crop_end]

    # 切片处理为单张2d图片，并去掉没有病灶的CT体积
    # if 判断当前CT体积的切片中是否存在非0像素的标签
    if int(np.sum(seg_crop))!=0:
        # 遍历每一张切片， seg_crop.size() = [n, 448, 448]
        for n_slice in range(seg_crop.shape[0]):
            
            segImg = seg_crop[n_slice,:,:]

            # 相邻切片合并成一个三通道的图像，为了联系上下文信息
            ctImageArray = np.zeros((ct_crop.shape[1], ct_crop.shape[2], 3), np.float)
            tumor_slice = ct_crop[n_slice , :, :]
            tumor_slice[segImg==0] = 0
            ctImageArray[:, :, 0] = tumor_slice

            tumor_slice = ct_crop[n_slice+1 , :, :]
            tumor_slice[segImg==0] = 0
            ctImageArray[:, :, 1] = tumor_slice

            tumor_slice = ct_crop[n_slice+2 , :, :]
            tumor_slice[segImg==0] = 0
            ctImageArray[:, :, 2] = tumor_slice

            # ctImageArray = ct_crop[n_slice, :, :]

            # 设置2d训练图片的文件名
            imagepath = para.train_ct_path + "/" + str(index) + "_" + str(n_slice) + ".npy"
            segpath = para.train_seg_path + "/" + str(index) + "_" + str(n_slice) + ".npy"

            # 保存2d图像的numpy到文件
            np.save(imagepath, ctImageArray)  # (448，448,3) np.float dtype('float64')
            np.save(segpath, segImg)  # (448，448) dtype('uint8') 值为0 1 2
    else:
        continue

print("Preprocess Done!")