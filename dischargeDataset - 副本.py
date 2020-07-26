#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.utils.data as Data
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader


# In[2]:


class PdDataset(Data.Dataset): # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, root, csvfile, transform=None, target_transform=None): # 初始化需要传入的参数
        super(PdDataset,self).__init__()
        fh = open(root + csvfile, 'r') #按照传入的路径和txt名，打开文本读取内容
        csvs = [] # 创建空列表
        for line in fh: # 按行循环txt文本中的内容
            line = line.rstrip() # 删除本行string字符串末尾的指定字符
            words = line.split() # 通过指定分隔符对字符串进行切片
            csvs.append((words[0],words[1])) # 把txt里的内容读入csv列表保存，[0]为文件路径，[1]是label
        
        self.csvs = csvs
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index): #按照索引读取每个元素的具体内容
        fn, label = self.csvs[index] #fn和label分别获得csvs[index]也即是刚才每行中word[0]和word[1]的信息
        csv = pd.read_csv(fn, header=None)
        csv = torch.from_numpy(csv.values)
        csv = csv.permute(1, 0).float()
        return csv, label # return返回哪些内容，在训练时循环读取每个batch时就能获得哪些内容
 
    def __len__(self): #返回数据集的长度，也就是多少个文件，要和loader的长度作区分
        return len(self.csvs)


# In[3]:


root = "d:\\Codes\\keyan\\peidian\\"
#根据自己定义的PdDataset创建数据集
train_data=PdDataset(root, "test_list.txt")
test_data=PdDataset(root, "train_list.txt")


# In[4]:


#调用DataLoader创建dataloader，loader的长度是有多少个batch，和batch_size有关
train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=False, num_workers=0)


# In[5]:


import torch.nn as nn
import torch.nn.functional as F


# In[6]:


device = torch.device("cuda")


# In[7]:


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv = nn.Sequential()
        self.conv.add_module("conv1", nn.Conv1d(in_channels=3, out_channels=500, kernel_size=50))
        self.conv.add_module("Relu1", nn.ReLU())
        self.conv.add_module("conv2", nn.Conv1d(in_channels=500, out_channels=500, kernel_size=50))
        self.conv.add_module("pool1", nn.MaxPool1d(3))
        self.conv.add_module("conv3", nn.Conv1d(in_channels=500, out_channels=800, kernel_size=50))
        self.conv.add_module("Relu2", nn.ReLU())
        self.conv.add_module("conv4", nn.Conv1d(in_channels=800, out_channels=800, kernel_size=50))
        self.conv.add_module("pool2", nn.AvgPool1d(2))
        self.dense = nn.Sequential()
        self.dense.add_module("dense1", nn.Linear(800, 360))
        self.dense.add_module("Relu3", nn.ReLU())
        self.dense.add_module("dense2", nn.Linear(360, 5))
    
    def forward(self, x):
        conv_out = self.conv(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense(res)
        return out
        


# In[8]:


net = CNNNet()
net = net.to(device)


# In[9]:


print(net)


# In[10]:


import torch.optim as optim


# In[11]:


criterion = nn.CrossEntropyLoss()
optimizier = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# In[12]:


def tuple2tensor_char(x):
    return torch.tensor(list(map(int, x)))


# In[13]:


for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = tuple2tensor_char(labels)
        labels = labels.to(device)
 
        optimizier.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizier.step()

        running_loss += loss.item()
        if i % 2000 ==1999:
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    
print('finish')


# In[16]:


correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        csvs, labels = data
        csvs = csvs.to(device)
        labels = tuple2tensor_char(labels)
        labels = labels.to(device)
        outputs = net(csvs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
 
print('Accuracy of the network on the test csvs: %d %%' % (100 * correct / total))


# In[17]:


class_correct = list(0. for i in range(5))
class_total = list(0. for i in range(5))
with torch.no_grad():
    for data in test_loader:
        csvs, labels = data
        csvs = csvs.to(device)
        labels = tuple2tensor_char(labels)
        labels = labels.to(device)
        outputs = net(csvs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
 
classes = ('0', '1', '2', '3', '4')
for i in range(5):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


# In[ ]:




