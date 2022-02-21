# 查看测试后生成的nii文件
import SimpleITK as sitk
import cv2
import os 
import numpy as np
# 用户只需要封装任意的迭代器 tqdm(iterator)。
from tqdm import tqdm 
# 导入自定义参数
import sys
sys.path.append(os.path.split(sys.path[0])[0])
import parameter as para 
from PIL import Image

# vscode的相对路径默认为项目的根目录
# # 读取当前路径
# print('current path: ', os.getcwd())
# # 改变相对路径到该py文件的目录
# os.chdir('data_analysis')

target = para.show_nii
if not os.path.exists(target):
    os.makedirs(target)

for index,file in enumerate(tqdm(os.listdir(para.test_ct_path))):
    nii_subfile = target +'/'+ file.split('.')[0]
    print('\nnii_subfile',nii_subfile)
    # 创建当前nii对应的子文件夹
    if not os.path.exists(nii_subfile):
        os.makedirs(nii_subfile)
    
    # 从文件读取并保存为数组
    image = sitk.ReadImage(os.path.join(para.test_ct_path, file))
    image = sitk.GetArrayFromImage(image)
    mask = sitk.ReadImage(os.path.join(para.test_seg_path, file.replace('volume','segmentation')))
    mask = sitk.GetArrayFromImage(mask)
    pred = sitk.ReadImage(os.path.join(para.pred_seg_path, file.replace('volume','segmentation').replace('nii','nii.gz')))
    pred = sitk.GetArrayFromImage(pred)
    print(image.shape,mask.shape,pred.shape)

    z = np.any(mask, axis=(1, 2))
    start_slice, end_slice = np.where(z)[0][[0, -1]] 

    for n_slice in range(start_slice,end_slice):
        cv2.imwrite(nii_subfile+'/'+ file.split('.')[0] + '_' + str(n_slice) + '_img'+".png",image[n_slice,:,:])
    
        if para.valuate == True:
            # 展示金标准肝脏标签
            cv2.imwrite(nii_subfile+'/'+ file.split('.')[0] +  '_' +str(n_slice)+ '_liver' +".png",mask[n_slice,:,:])
            # 展示金标准肿瘤标签
            mask[mask==1]=0
            cv2.imwrite(nii_subfile+'/'+ file.split('.')[0]+  '_' +str(n_slice)+'_tumor' +".png",mask[n_slice,:,:])

         # 展示预测的肝脏标签
        cv2.imwrite(nii_subfile+'/'+ file.split('.')[0] +  '_' +str(n_slice) + '_pred_liver'+".png",pred[n_slice,:,:])
 
        # 展示预测的肿瘤标签
        pred[pred==1]=0
        cv2.imwrite(nii_subfile+'/'+ file.split('.')[0]+  '_' +str(n_slice) +'_pred_tumor'+".png",pred[n_slice,:,:])
    
    print(file,'转换png完成')