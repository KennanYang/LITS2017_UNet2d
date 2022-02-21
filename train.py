# train.py
# 2.训练（运行 ./train.py） 用训练集训练网络模型，保存到./models


import os
from time import time
import datetime
import numpy as np

# 【PyTorch】 是一个 Python 优先的深度学习框架，能够在强大的 GPU 加速基础上实现张量和动态神经网络
"""
# pytorch 的数据加载到模型的操作顺序是这样的：
# 1. 创建一个 Dataset 对象
# 2. 创建一个 DataLoader 对象
# 3. 循环这个 DataLoader 对象，将img, label加载到模型中进行训练
"""
import torch
# cuDNN 是英伟达专门为深度神经网络所开发出来的 GPU 加速库，针对卷积、池化等等常见操作做了非常多的底层优化，比一般的 GPU 程序要快很多.
import torch.backends.cudnn as cudnn
# torch.optim是一个实现了多种优化算法的包

# glob模块可以查找符合特定规则的文件路径名
from glob import glob
# train_test_split将原始数据按照比例分割为“测试集”和“训练集”
from sklearn.model_selection import train_test_split

from dataset.dataset import Dataset
# DataLoader将自定义的Dataset根据batch size大小、是否shuffle等封装成一个Batch Size大小的Tensor，用于后面的训练。
from torch.utils.data import DataLoader


# 自定义loss函数
"""
【损失函数】：真实值与预测值差别越大，Loss越大，我们的优化的目标就是减小Loss。
    cnn进行前向传播阶段，依次调用每个Layer的Forward函数，得到逐层的输出，
    最后一层与目标函数比较得到损失函数，计算误差更新值，
    通过反向传播逐层到达第一层，所有权值在反向传播结束时一起更新。
    损失函数是在前向传播计算中得到的，同时也是反向传播的起点。
"""

from utilities.losses import BCELoss
from utilities.losses import DiceLoss
from utilities.losses import HybridLoss
from net.UNet import U_Net, net, net_total_para

from scipy.ndimage import zoom
from PIL import Image
from tqdm import tqdm
import parameter as para
from utilities.metrics import iou_score,dice_coef

# 重写collate_fn函数，即用于对单个样本生成batch的函数, 其输入为一个batch的sample数据
# 在dataloader按照batch进行取数据的时候, 是取出大小等同于batch size的index列表; 
# 然后将列表列表中的index输入到dataset的getitem()函数中,取出该index对应的数据; 
# 最后, 对每个index对应的数据进行堆叠, 就形成了一个batch的数据
def collate_fn(batch):
    # print("type(batch), len(batch):", type(batch), len(batch))
    
    # 增加一个维度为batch size
    img_sequence_list = []
    label_sequence_list = []
    for i in range(len(batch)):
        img, label= batch[i][0], batch[i][1]
        img_sequence_list.append(img)
        label_sequence_list.append(label)
    
    # print(" len(batch),img_sequence)",len(batch),len(img_sequence_list)) # 20 (20,3,448,448)

    img_sequence = torch.Tensor(list(img_sequence_list))
    label_sequence = torch.Tensor(list(label_sequence_list))

    # print('img_sequence')
    # print("Tensor--- img_sequence.shape, label_sequence.shape, mask_sequence.shape, sign_list.shape, center_point, name, size:", \
    #     type(img_sequence), img_sequence.shape, label_sequence.shape, mask_sequence.shape, sign_list.shape, center_point, name, size)
    return img_sequence, label_sequence # maybe map change to entorpy

# 用于在训练过程中评价网络模型
def validate(val_dl, net):
    # switch to evaluate mode
    net.eval()
    total=len(val_dl)
    ious=0
    dice_1s=0
    dice_2s=0
    with torch.no_grad():
        for i, (input,target) in tqdm(enumerate(val_dl), total=len(val_dl)):
            # print('input,target',input.shape)
            input = input.cuda()
            target = target.cuda()

            output = net(input)
            
            ious = ious + iou_score(output, target)
            dice_1s = dice_1s + dice_coef(output, target)[0]
            dice_2s = dice_2s + dice_coef(output, target)[1]

    iou = ious / total
    dice_1 = dice_1s / total
    dice_2 = dice_2s / total
    return iou,dice_1,dice_2

# 数据增强：给数据增加随机的旋转/缩放
def data_tf(img, label):
    # print('img.shape,label.shape:',img.shape,label.shape)
    # img:448*448   label: 2*448*448
    # 缩放尺寸数（数值 = 原图尺寸-新图尺寸）
    mode_size = np.random.randint(0,25)
    # 旋转角度数
    mode_angle = np.random.randint(-10,10)
    mode_Symbol = np.random.choice([-1,0,1])
    # 随机缩放
    if mode_Symbol == 1:
        pad = mode_size
        img = np.pad(img, pad_width=((0, 0), (pad, pad), (pad, pad)), mode="constant")
        label = np.pad(label, pad_width=((0, 0), (pad, pad), (pad, pad)), mode="constant")

    elif mode_Symbol == -1:
        pad = mode_size
        img = img[:,pad:para.img_size - pad, pad:para.img_size - pad]
        label = label[:, pad:para.img_size - pad, pad:para.img_size - pad]

    # 顺逆旋转随机角度
    img_list = []
    label_list = []
    for i in range(len(img)):
        img_tem = Image.fromarray(img[i])        
        img_tem = img_tem.rotate(mode_angle)
        img_list.append(np.array(img_tem))
    for i in range(len(label)):
        label_tem = Image.fromarray(label[i])
        label_tem = label_tem.rotate(mode_angle)
        label_list.append(np.array(label_tem))
        #print("旋转了{}度".format(mode_Symbol*mode_list[mode]))
    img = np.array(img_list)
    label = np.array(label_list)

    # print('img.shape,label.shape:',img.shape,label.shape)
    # 按比例还原图片大小
    factor = (3/img.shape[0], para.img_size / img.shape[1], para.img_size / img.shape[2])
    img = zoom(img, factor,order=2)
    factor = (2/label.shape[0], para.img_size / label.shape[1], para.img_size / label.shape[2])
    label = zoom(label, factor, order=0)

    # img = img[np.newaxis, :, :]
    # print('img.shape,label.shape:',img.shape,label.shape)
    return img, label

# 主函数 训练网络模型，结果存入文件
if __name__ == '__main__':
    # 设置显卡相关，指定使用CUDA的显卡设备为para.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu

    # 设置 torch.backends.cudnn.benchmark为整个网络的每个卷积层搜索最适合它的卷积实现算法
    cudnn.benchmark = para.cudnn_benchmark

    # 定义网络,使用pytorch框架，用cuda进行数据并行处理加速
    """
    DataParallel 并行计算仅存在在前向传播
    DataParallel 会自动帮我们将数据切分 load 到相应 GPU，将模型复制到相应 GPU，进行正向传播计算梯度并汇总。
    """
    net = torch.nn.DataParallel(net).cuda()
    # model.train()是保证BN层用每一批数据的均值和方差
    """
    这两个方法是针对在网络train和eval时采用不同方式的情况，比如Batch Normalization和Dropout。
        如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()，在测试时添加model.eval()。
        其中model.train()是保证BN层用每一批数据的均值和方差，
        而model.eval()是保证BN用全部训练数据的均值和方差；
        而对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接。
    """
    net.train()

    # 分割训练集和验证集
    ct_paths = glob(para.train_ct_path+'/*')
    seg_paths = glob(para.train_seg_path+'/*')
    train_ct_paths, val_ct_paths, train_seg_paths, val_seg_paths = \
        train_test_split(ct_paths, seg_paths, test_size=0.3, random_state=39)
    # train_ct_paths = glob(para.train_ct_path+'/*')
    # train_seg_paths = glob(para.train_seg_path+'/*')
    # val_ct_paths = glob(para.test_ct_path+'/*')
    # val_seg_paths = glob(para.test_seg_path+'/*')

    print("train_num:%s" % str(len(train_ct_paths)))
    print("val_num:%s" % str(len(val_ct_paths)))

    # 定义Dateset
    train_ds = Dataset(train_ct_paths, train_seg_paths, transform=data_tf)
    val_ds = Dataset(val_ct_paths, val_seg_paths)

    # 定义训练和验证的数据加载
    """
    DataLoader(object)的参数:
        dataset(Dataset): 传入的数据集
        batch_size(int, optional): 每个batch有多少个样本
        shuffle(bool, optional): 在每个epoch开始的时候，对数据进行重新排序
        sampler(Sampler, optional): 自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False
        batch_sampler(Sampler, optional): 与sampler类似，但是一次只返回一个batch的indices（索引），需要注意的是，一旦指定了这个参数，那么batch_size,shuffle,sampler,drop_last就不能再制定了（互斥——Mutually exclusive）
        num_workers (int, optional): 这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0）
        collate_fn (callable, optional): 将一个list的sample组成一个mini-batch的函数
        pin_memory (bool, optional)： 如果设置为True，那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存（CUDA pinned memory）中.
    """
    train_dl = DataLoader(
        train_ds, 
        para.batch_size, 
        shuffle = True, 
        num_workers=para.num_workers, 
        pin_memory=para.pin_memory,
        drop_last=True)
    val_dl = DataLoader(
        val_ds, 
        para.batch_size, 
        shuffle = False, 
        num_workers=para.num_workers, 
        pin_memory=para.pin_memory,
        drop_last=False)

    # 挑选损失函数
    loss_func_list = [BCELoss(),DiceLoss(), HybridLoss()]
    loss_func = loss_func_list[1]

    # 定义优化器
    """
    【优化器】为了使用torch.optim，需先构造一个优化器对象Optimizer，用来保存当前的状态，并能够根据计算得到的梯度来更新参数。
        optimzier优化器的作用, 形象地来说，优化器就是需要根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用.

        Adam(Adaptive Moment Estimation)本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。   
        它的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。
    """
    optimizer = torch.optim.Adam(net.parameters(), lr=para.learning_rate)

    # 学习率衰减
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(optimizer, para.learning_rate_decay)

    # 如果没有保存网络模型的文件夹，则创建
    if not os.path.exists(para.result_path):
        os.mkdir(para.result_path)
    
    # 格式为./models/[model name]/[datetime]/[net].pth ,即目录依次按照 主目录/模型名称/训练完成时间/
    timestamp  = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # 创建保存模型的文件夹
    model_path = '{}/{}/{}'.format(para.model_path,para.model_name,timestamp)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # 创建日志，保存模型参数，'a'表示追加内容
    with open('{}/{}.log'.format(model_path, timestamp), 'a') as f:
        print('model name: %s' %(para.model_name), file=f)
        print('train_num: %s' % str(len(train_ct_paths)), file=f)
        print('val_num: %s' % str(len(val_ct_paths)), file=f)
        
        print('batch size: %s' %(para.batch_size), file=f)
        print('num workers: %s' %(para.num_workers), file=f)
        print('learning_rate: %s' %(para.learning_rate), file=f)
        print('learning_rate_decay size: %s' %(para.learning_rate_decay), file=f)
        print('net total parameters: %s' %(net_total_para), file=f)
        print('-----------------------------------------\n', file=f)

    # 开始训练网络
    start = time()

    # 加载训练未完成的网络
    # net.load_state_dict(torch.load(para.model))

    # 初始化肝脏分割的dice值为0
    best_liver_dice = 0 

    # 依次产生每一个epoch
    for epoch in range(para.Epoch):
        # 学习率更新
        lr_decay.step()

        # 记录每一个epoch的所有batch的loss，求均值
        mean_loss = []

        # 遍历每一个minibatch，根据步长获取数据，并计算损失
        for step, (ct,seg) in enumerate(train_dl):
            # 数据放到GPU
            ct = ct.cuda()
            seg = seg.cuda()
            
            # print('ct',ct.shape)
            # 通过网络获得分割结果
            outputs = net(ct)
            
            # print('outputs.type,outputs.shape',outputs.type,outputs.shape)
            # 对比预测值和金标准，评估损失
            loss = loss_func(outputs,seg) 
            mean_loss.append(loss.item())
            
            # optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
            """
            在向后传递之前，使用优化器对象使它将要更新的变量的所有梯度归零(这是模型的可学习权值)。
            这是因为在默认情况下，每当调用.backward()时，梯度会在缓冲区中累积(即不会被覆盖)。
            """
            optimizer.zero_grad()
            # 向后传递:计算损失相对于模型参数的梯度
            """
            训练神经网络的目的就是通过反向传播过程来实现可训练参数的更新，这正是loss.backward()的目的。
            """
            loss.backward()
            # 在优化器上调用step函数会更新它的参数
            optimizer.step()

            # 打印损失函数
            if step % 5 == 0:
                print('epoch:{}, step:{}, loss:{:.3f}, time:{:.3f} min'
                        .format(epoch, step, loss.item(), (time() - start) / 60))

        # 求loss的均值
        mean_loss = sum(mean_loss) / len(mean_loss)

        # 验证集验证模型的精度
        val_log = validate(val_dl, net)

        print('epoch:{}, mean_loss: {:.4f}, iou: {:.8f}, liver_dice: {:.8f}, tumor_dice: {:.8f}, time:{:.3f} min'
                    .format(epoch, mean_loss, val_log[0], val_log[1], val_log[2], (time() - start) / 60))
        print('-----------------------------------------\n')


        tumor_dice = val_log[2]
        if tumor_dice > best_liver_dice and epoch != 0:
            best_liver_dice = tumor_dice # 更新最优dice值
            # 【保存模型】    
            # 网络模型的命名方式为：epoch轮数+当前minibatch的loss+本轮epoch的平均loss
            torch.save(net.state_dict(), '{}/net{}-{:.3f}-{:.3f}.pth'.format(model_path, epoch, loss, mean_loss))

            # 参数写入网络模型对应的[datetime].log
            with open('{}/{}.log'.format(model_path, timestamp), 'a') as f:
                print('epoch: %s' %(epoch), file=f)
                print('loss: %.4f' %(loss.item()), file=f)
                print('mean_loss: %.4f' %(mean_loss), file=f)

                print('iou: %.8f' %(val_log[0]), file=f)
                print('liver_dice: %.8f' %(val_log[1]), file=f)
                print('tumor_dice: %.8f' %(val_log[2]), file=f)
                
                print('time: %.3f min' %((time() - start) / 60), file=f)
                print('-----------------------------------------\n', file=f)

