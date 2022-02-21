# parameter.py
# 0.文件描述了项目运行过程，以及这个工程需要自定义的参数和路径设置

# 步骤：
# 1.预处理（运行./data_prepare/preprocess.py） 将原始数据集处理成用于训练网络模型的2d或3d训练集
# 2.训练（运行 ./train.py） 用训练集训练网络模型，保存到./models
# 3.测试（运行 ./test.py） 用网络预测测试集的标签，保存到./data/pred_seg，并评估预测分割结果，通过评价指标判断分割精度

# -----------------------路径相关参数---------------------------------------------------
# 原始LITS2017 dataset的trainset路径，用于preprocess
# 服务器
lits_train_ct_path = "../../dataset/lits2017/train/ct/"      # 原始训练集的CT图像，volumn-*.nii
lits_train_seg_path = "../../dataset/lits2017/train/seg/"    # 原始训练集的分割标注，segmentation-*.nii
# pc端
# lits_train_ct_path = "D:/projects/LITS/train/TrainingDataset/train/ct/"      # 原始训练集的CT图像，volumn-*.nii
# lits_train_seg_path = "D:/projects/LITS/train/TrainingDataset/train/seg/"    # 原始训练集的分割标注，segmentation-*.nii

# 用于网络模型训练的trainset路径，用于train
# 服务器131
# train_ct_path = "/home/planck/Desktop/yzw/LITS2017-main1/data/trainImage_k1_1217/"       # 预处理后的训练集的CT图像，volumn-*.nii
# train_seg_path = "/home/planck/Desktop/yzw/LITS2017-main1/data/trainMask_k1_1217/"      # 预处理后的训练集的分割标注，segmentation-*.nii

# 服务器109
train_ct_path = "/home/planck/Desktop/yzw/LITS2017-yzw/data/train_ct_onlyliver/"       # 预处理后的训练集的CT图像，volumn-*.nii
train_seg_path = "/home/planck/Desktop/yzw/LITS2017-yzw/data/train_seg_onlyliver/"      # 预处理后的训练集的分割标注，segmentation-*.nii
# pc端
# train_ct_path = "E:/train_ct"       # 预处理后的训练集的CT图像，volumn-*.nii
# train_seg_path = "E:/train_seg"      # 预处理后的训练集的分割标注，segmentation-*.nii

# 用于保存预处理结果可查看的png格式
show_npy = '../../show_npy/'
show_nii = '../../show_nii/'

# 网络模型预测分割结果的test set路径，用于test
# 服务器22
# test_ct_path = "/home/planck/Desktop/yzw/dataset/lits2017/test/ct/"       # 测试集的CT图像，volumn-*.nii
# test_seg_path = "/home/planck/Desktop/yzw/dataset/lits2017/test/seg/"      # 测试集的分割标注，segmentation-*.nii
# pc端22
test_ct_path = "E:/BaiduNetdiskWorkspace/MICCAI-LITS2017-master/test/CT/"       # 测试集的CT图像，test-volumn-*.nii
test_seg_path = "E:/BaiduNetdiskWorkspace/MICCAI-LITS2017-master/test/seg/"      # 测试集的分割标注，segmentation-*.nii
# pc端70
# test_ct_path = "D:/projects/LITS/test/ct/"
# test_seg_path = "D:/projects/LITS/test/seg/"
pred_seg_path = "./data/pred_seg/"      # 预测的分割标注，segmentation-*.nii

# 保存评价指标的路径，用于val
result_path = "./data/val_result/"      # 保存评价指标（分割的精度的结果）

# 保存网络模型的路径, 格式为 ./models/[model name]/[date time]/[net].pth
model_path = "./models"     # 模型总目录   
model_name = "UNet2d"       # 模型子目录，以网络模型命名
model = "./models/UNet2d/2021-11-10_23-07-37/net72-0.000-0.001.pth"      # 获取一个网络模型文件，用于测试集的分割预测

# -----------------------训练数据预处理相关参数---------------------------------------------------
upper, lower = 200, -200  # CT数据灰度截断窗口

# 2d图像分割参数： 默认(512*512,选取中间448*448的部分)
crop_start = 32  # 裁剪切片的像素起点
crop_end = 480   # 裁剪切片的像素终点
img_size = 448

# 3d图像分割参数：
size = 48  # 使用48张连续切片作为网络的输入
down_scale = 0.5  # 横断面降采样因子
expand_slice = 20  # 仅使用包含肝脏以及肝脏上下20张切片作为训练样本
slice_thickness = 1  # 将所有数据在z轴的spacing归一化到1mm


# -----------------------网络结构相关参数------------------------------------

drop_rate = 0.3  # dropout随机丢弃概率

# -----------------------网络训练相关参数--------------------------------------

gpu = '0,1'  # 使用的显卡序号

# 设置 torch.backends.cudnn.benchmark=True为整个网络的每个卷积层搜索最适合它的卷积实现算法
"""
【cudnn.benchmark】将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
    适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。
    反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间。
"""
cudnn_benchmark = True

# 定义epoch数量,这里发现150以后，dice值就不再更新了
"""
当一个完整的数据集通过了神经网络一次并且返回了一次，这个过程称为一个 epoch
随着 epoch 数量增加，神经网络中的权重的更新次数也增加，曲线从欠拟合变得过拟合。
"""
Epoch = 150

# 定义batch size大小
"""
Batch Size定义：一次训练所选取的样本数. 
    Batch Size的大小影响模型的优化程度和速度
    Batch Size从小到大的变化对网络影响
    1、没有Batch Size，梯度准确，只适用于小样本数据库
    2、Batch Size=1，梯度变来变去，非常不准确，网络很难收敛。
    3、Batch Size增大，梯度变准确，
    4、Batch Size增大，梯度已经非常准确，再增加Batch Size也没有用

    注意：Batch Size增大了，要到达相同的准确度，必须要增大epoch。
"""
batch_size = 20

# 设置加载数据的线程数目
"""
num_workers是加载数据（batch）的线程数目

    当加载batch的时间 < 数据训练的时间

    　　GPU每次训练完都可以直接从CPU中取到next batch的数据

    　　无需额外的等待，因此也不需要多余的worker，即使增加worker也不会影响训练速度

    当加载batch的时间 > 数据训练的时间

    　　GPU每次训练完都需要等待CPU完成数据的载入

    　　若增加worker，即使worker_1还未就绪，GPU也可以取worker_2的数据来训练

    仅限单线程训练情况
"""
num_workers = 4
# 设置锁页内存
"""
pin_memory就是锁页内存
    创建DataLoader时，
    设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，
    这样将内存的Tensor转义到GPU的显存就会更快一些。
"""
pin_memory = True

# 学习率
"""
【学习率】即步长，它控制着算法优化的速度。
    使用的学习率过大，虽然在算法优化前期会加速学习，使得损失能够较快的下降，但在优化后期会出现波动现象以至于不能收敛。
    如果使用的学习率偏小，那么极有可能训练时loss下降得很慢，算法也很难寻优。
"""
learning_rate = 3e-4

# 学习率衰减
"""
多间隔调整学习率 MultiStepLR:
    学习率调整的间隔并不是相等的，
    参数：milestone(list): 一个列表参数，表示多个学习率需要调整的epoch值，如milestones=[10, 30, 80].epoch=10时调整一次，epoch=30时调整一次，epoch=80时调整一次
"""
learning_rate_decay = [100, 150]


# -----------------------模型测试相关参数--------------------------------------
maximum_hole = 5e4  # 最大的空洞面积
