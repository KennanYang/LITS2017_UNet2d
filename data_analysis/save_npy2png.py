# 查看预处理后生成的npy文件
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 
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

target = para.show_npy
if not os.path.exists(target):
    os.makedirs(target)

for index,file in enumerate(tqdm(os.listdir(para.train_ct_path))):
    print('file',str(file))
    image = np.load(os.path.join(para.train_ct_path, file))
    mask = np.load(os.path.join(para.train_seg_path, file))
    
    # if image[:,:,0].sum() == image[:,:,1].sum():
    #     print('1\n1\n1\n')
    # else:
    #     print('2\n2\n2\n')
    # plt.imshow(image[:,:],cmap='gray')
    # cv2.imwrite(target+'/'+ file.split('.')[0]  +'_img'+".png",image*400)    
    for i in range(3):
        plt.imshow(image[:,:,i],cmap='gray')
        cv2.imwrite(target+'/'+ file.split('.')[0] + '_'+ str(i) +'_img'+".png",image*200)
    # plt.show()
    # 展示npy格式的肝脏标签
    plt.imshow(mask,cmap='gray')
    cv2.imwrite(target+'/'+ file.split('.')[0] + '_liver'+".png",mask*127.5)
    # plt.show()

    # 展示npy格式的肿瘤标签
    mask[mask==1]=0
    plt.imshow(mask,cmap='gray')
    cv2.imwrite(target+'/'+ file.split('.')[0]+'_tumor'+".png",mask*255)
    # plt.show()