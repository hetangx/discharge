import numpy as np
import pandas as pd
from sklearn import preprocessing
import tools
# 运行一次，得到数据的最值、均值、标准差
filepath = r"D:\Codes\keyan\peidian\train_list.txt" # 训练集路径

DATA_ori = tools.LinkData(filepath) # 读取训练集，合并为np.array

DATA_mmscale = preprocessing.MinMaxScaler().fit_transform(DATA_ori) 
DATA_sk_stand = tools.SK_Standard(DATA_ori)
# scale = preprocessing.StandardScaler()
# scale.fit(DATA_ori)
# scale.scale_ = np.std(DATA_ori, axis=0, ddof=1)
# DATA_standard = scale.transform(DATA_ori)

MMMS_ori = tools.MaxMinMeanStd(DATA_ori)
MMMS_mms = tools.MaxMinMeanStd(DATA_mmscale)
MMMS_stand = tools.MaxMinMeanStd(DATA_sk_stand)

print("original data mmms :")
print(MMMS_ori)
print("min max scale data mmms: ")
print(MMMS_mms)
print("standard(zscore normalize) data mmms: ")
print(MMMS_stand)