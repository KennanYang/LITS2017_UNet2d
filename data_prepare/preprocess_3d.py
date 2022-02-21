# preprocess_3d.py
# 1.预处理（运行./data_prepare/preprocess_3d.py） 将原始数据集处理成用于训练网络模型的训练集

       
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
# SciPy提供了ndimage(n维图像)包, 其中包含许多常规图像处理和分析功能
import scipy.ndimage as ndimage
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
    ct_array = ct_array / 200

    # 找到金标准的肝脏区域开始和结束的slice（范围在0~1），并各向外扩张slice
    z = np.any(seg_array, axis=(1, 2))
    start_slice, end_slice = np.where(z)[0][[0, -1]] 

    # ---------------------------------3D分割预处理---------------------------------  
    # 3d图像分割：
    # 取出只包含肝脏和肿瘤金标准的切片，保存为一个CT体积，命名格式保留

    # 对CT数据在横断面上进行降采样,并进行重采样,将所有数据的z轴的spacing调整到1mm
    # zoom函数的作用是按比例缩放图片，order是样条插值的顺序，默认是3。顺序必须在0-5范围内。
    ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / para.slice_thickness, para.down_scale, para.down_scale), order=3)
    seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / para.slice_thickness, 1, 1), order=0)

    # 取包含肝脏病灶的切片的起始索引，并且两个方向上各扩张expand_slice
    # 这是因为3d卷积需要同时获取3d上下文信息
    start_slice = max(0, start_slice - para.expand_slice)
    end_slice = min(seg_array.shape[0] - 1, end_slice + para.expand_slice)
    
    # 如果这时候剩下的slice数量不足size，直接放弃该数据，这样的数据很少,所以不用担心
    if end_slice - start_slice + 1 < para.size:
        print('!!!!!!!!!!!!!!!!')
        print(file, 'have too little slice', ct_array.shape[0])
        print('!!!!!!!!!!!!!!!!')
        continue
    
    # 截取包含肝脏的切片
    ct_array = ct_array[start_slice:end_slice + 1, :, :]
    seg_array = seg_array[start_slice:end_slice + 1, :, :]

    # 最终将数据保存为nii
    # GetImageFromArray()可用于将ndarray数组转换为SimpleITK对象
    new_ct = sitk.GetImageFromArray(ct_array) 

    new_ct.SetDirection(ct.GetDirection())
    new_ct.SetOrigin(ct.GetOrigin())
    new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / para.down_scale), ct.GetSpacing()[1] * int(1 / para.down_scale), para.slice_thickness))

    new_seg = sitk.GetImageFromArray(seg_array)

    new_seg.SetDirection(ct.GetDirection())
    new_seg.SetOrigin(ct.GetOrigin())
    new_seg.SetSpacing((ct.GetSpacing()[0], ct.GetSpacing()[1], para.slice_thickness))

    # 写入预处理后的nii格式的CT体积到文件
    sitk.WriteImage(new_ct, os.path.join(para.train_ct_path, file))
    sitk.WriteImage(new_seg, os.path.join(para.train_seg_path, file.replace('volume', 'segmentation').replace('.nii', '.nii.gz')))

print("Preprocess Done!")