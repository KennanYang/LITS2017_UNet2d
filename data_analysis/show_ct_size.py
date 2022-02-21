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

ct_max_size = 0
seg_max_size = 0

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
    seg_array[seg_array == 2] = 1

    # 将灰度值在阈值之外的截断掉（窗位：CT值中心，窗宽：CT值范围。这里窗宽为upper-lower）
    ct_array[ct_array > para.upper] = para.upper
    ct_array[ct_array < para.lower] = para.lower

    # 【归一化】：将肝脏区域设为（0，1）之间的float值
    ct_array = ct_array.astype(np.float32)
    ct_array = ct_array / 200

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

    ct_max_size = max(ct_max_size, len(ct_array))
    seg_max_size = max(seg_max_size, len(seg_crop))
    print(file, len(ct_array),len(seg_crop))

print("Preprocess Done! %d, %d",ct_max_size,seg_max_size)