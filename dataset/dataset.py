# dataset.py
# torch中的Dataset定义脚本，为了传入Dataloader使用

import parameter as para
import numpy as np

from torch.utils.data import Dataset as dataset 

# 这里重写了一个pytorch中定义的Dataset抽象类，所有数据集代表一个从key到data samples的map。
class Dataset(dataset):
    # 传入参数：处理后的训练集的ct图像和分割标签的路径
    def __init__(self, ct_list, seg_list,transform = None):
        # 获取训练集的所有文件名的列表
        self.ct_list = ct_list
        self.seg_list = seg_list
        self.transform = transform
    
    # 获取数据集的大小
    def __len__(self):
        return len(self.ct_list)

    # 获取一个给定key的数据样本
    def __getitem__(self,index):
        # 获得单个ct图像和分割标签的路径
        ct_path = self.ct_list[index]
        seg_path = self.seg_list[index]
 
        # print("index ",index)
        # print(self.ct_paths)
        # print(ct_path)
        # print(self.seg_paths)
        # print(seg_path)

        # 读取numpy数据(.npy)
        npct = np.load(ct_path)
        npseg = np.load(seg_path)
        
        # 获取截取的图片大小（448*448*3）
        npct.resize(para.img_size, para.img_size, 3)

        # # 改为（3，448，448）
        npct = npct.transpose((2, 0, 1))

        # 处理标签变成2分类问题
        # 肝脏分割时，把肿瘤标签也设定为1
        liver_label = npseg.copy()
        liver_label[npseg == 2] = 1
        liver_label[npseg == 1] = 1

        # 肿瘤分割时，把肝脏标签设定为0
        tumor_label = npseg.copy()
        tumor_label[npseg == 1] = 0
        tumor_label[npseg == 2] = 1

        # 建立一个高维数组存标签，shape为 448*448*2， 一层存肝脏标签，一层存肿瘤标签
        nplabel = np.empty((para.img_size,para.img_size,2))
        nplabel[:, :, 0] = liver_label
        nplabel[:, :, 1] = tumor_label
        nplabel = nplabel.transpose((2,0,1))
        
        # 把ct中肝脏之外的部分置为背景
        # npct = npct * liver_label
        # npct [npct==0] = -1

        nplabel = nplabel.astype("float32")
        npct = npct.astype("float32")

        # 数据增强：给数据增加随机的旋转/缩放
        if self.transform is not None:
            npct, nplabel = self.transform(npct,nplabel)

        # 返回值为根据索引index分别获取图像，标签
        return npct,nplabel