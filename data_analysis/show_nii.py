# encoding=utf8
'''
查看和显示nii文件，可以直观的看到三个维度的图像
'''
 
import matplotlib
matplotlib.use('TkAgg')
 
from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D

import os 
import sys
sys.path.append(os.path.split(sys.path[0])[0])
import parameter as para

example_filename = para.test_ct_path +'volume-27.nii'
 
img = nib.load(example_filename)
print (img)
print (img.header['db_name'])   #输出头信息

width,height,queue=img.dataobj.shape

OrthoSlicer3D(img.dataobj).show()
 
# num = 1
# for i in range(0,queue,10):
 
#     img_arr = img.dataobj[:,:,i]
#     plt.subplot(5,4,num)
#     plt.imshow(img_arr,cmap='gray')
#     num +=1
 
# # 输出三个维度的图像
# plt.show()

