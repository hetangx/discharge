#!/usr/bin/env python
# coding: utf-8

# # 原型阶段：方法暴露，不要封装，先完成基本流程

# ## 导入包

# In[1]:


from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
# 一些函数
import tools


# ## 加载数据

# In[2]:


path = r"D:\Codes\keyan\peidian\train_list.txt"
mms, stand = tools.SK_scaler(tools.LinkData(path))


# In[3]:


class PdDataset(Data.Dataset): # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, root, csvfile): # 初始化需要传入的参数
        super(PdDataset,self).__init__()
        fh = open(root + csvfile, 'r') #按照传入的路径和txt名，打开文本读取内容
        csvs = [] # 创建空列表
        for line in fh: # 按行循环txt文本中的内容
            line = line.rstrip() # 删除本行string字符串末尾的指定字符
            words = line.split() # 通过指定分隔符对字符串进行切片
            csvs.append((words[0],words[1])) # 把txt里的内容读入csv列表保存，[0]为文件路径，[1]是label
        
        self.csvs = csvs
        
    def __getitem__(self, index): #按照索引读取每个元素的具体内容
        fn, label = self.csvs[index] #fn和label分别获得csvs[index]也即是刚才每行中word[0]和word[1]的信息
        csv = pd.read_csv(fn, header=None)
        csv = tools.DataTrans(csv.values.astype(float), mms, stand)
        csv = torch.from_numpy(csv)
        csv = csv.permute(1, 0).float()
        return csv, label # return返回哪些内容，在训练时循环读取每个batch时就能获得哪些内容
 
    def __len__(self): #返回数据集的长度，也就是多少个文件，要和loader的长度作区分
        return len(self.csvs)


# In[4]:


root = "d:\\Codes\\keyan\\peidian\\"
#根据自己定义的PdDataset创建数据集
train_data=PdDataset(root, "train_list.txt")
test_data=PdDataset(root, "test_list.txt")


# In[5]:


classes_name = ['corona_discharge', 'float_discharge', 'interference', 'internal_discharge', 'surface_discharge']


# In[6]:


train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=4, shuffle=False, num_workers=0)


# ## 定义网络

# In[7]:


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 100, 10)
        self.pool1 = nn.MaxPool1d(3)
        self.conv2 = nn.Conv1d(100, 100, 10)
        self.pool2 = nn.MaxPool1d(3)
        self.conv3 = nn.Conv1d(100, 160, 10)
        self.pool3 = nn.MaxPool1d(3)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(160*10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop1(x)
        x = x.view(-1, 160*10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def initialize_weights(self):
        # 权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


# In[8]:


net = CNNNet()
net = net.cuda()
net.initialize_weights()


# ## 损失函数与优化器

# In[9]:


criterion = nn.CrossEntropyLoss()                                                   # 选择损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, dampening=0.1)    # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)     # 设置学习率下降策略


# ## 训练

# In[10]:


device = torch.device("cuda")


# In[24]:


max_epoch = 20


# In[25]:


for epoch in range(max_epoch):

    loss_sigma = 0.0    # 记录一个epoch的loss之和
    correct = 0.0
    total = 0.0
    scheduler.step()  # 更新学习率

    for i, data in enumerate(train_loader):
        # if i == 30 : break
        # 获取图片和标签
        inputs, labels = data
        inputs = inputs.to(device)
        labels = torch.tensor(list(map(int, labels)))
        labels = labels.to(device)
        # inputs, labels = Variable(inputs), Variable(labels)

        # forward, backward, update weights
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 统计预测信息
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        # correct += (predicted == labels).squeeze().sum().numpy()
        correct += (predicted == labels).squeeze().sum().cpu().numpy()
        loss_sigma += loss.item()

        # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
        if i % 10 == 9:
            loss_avg = loss_sigma / 10
            loss_sigma = 0.0
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch + 1, max_epoch, i + 1, len(train_loader), loss_avg, correct / total))

            # # 记录训练loss
            # writer.add_scalars('Loss_group', {'train_loss': loss_avg}, epoch)
            # # 记录learning rate
            # writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch)
            # # 记录Accuracy
            # writer.add_scalars('Accuracy_group', {'train_acc': correct / total}, epoch)

    # # 每个epoch，记录梯度，权值
    # for name, layer in net.named_parameters():
    #     writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
    #     writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

    # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
    if epoch % 2 == 0:
        loss_sigma = 0.0
        cls_num = len(classes_name)
        conf_mat = np.zeros([cls_num, cls_num])  # 混淆矩阵
        net.eval()
        for i, data in enumerate(test_loader):

            # 获取图片和标签
            # images, labels = data
            # images, labels = Variable(images), Variable(labels)
            inputs, labels = data
            inputs = inputs.to(device)
            labels = torch.tensor(list(map(int, labels)))
            labels = labels.to(device)

            # forward
            outputs = net(inputs)
            outputs.detach_()

            # 计算loss
            loss = criterion(outputs, labels)
            loss_sigma += loss.item()

            # 统计
            _, predicted = torch.max(outputs.data, 1)
            # labels = labels.data    # Variable --> tensor

            # 统计混淆矩阵
            for j in range(len(labels)):
                # cate_i = labels[j].numpy()
                cate_i = labels[j].cpu().numpy()
                # pre_i = predicted[j].numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.0

        print('{} set Accuracy:{:.2%}'.format('Valid', conf_mat.trace() / conf_mat.sum()))
        # # 记录Loss, accuracy
        # writer.add_scalars('Loss_group', {'valid_loss': loss_sigma / len(test_loader)}, epoch)
        # writer.add_scalars('Accuracy_group', {'valid_acc': conf_mat.trace() / conf_mat.sum()}, epoch)
print('Finished Training')


# In[ ]:




