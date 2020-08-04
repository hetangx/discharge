import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


# dataset_dir = os.path.join("..")

# 显示数据

def plot3(data, title):
    sns.set_style('dark')
    f, ax = plt.subplots()
    ax.set(ylabel='frequency')
    ax.set(xlabel='a(blue) / b(green) / c(red)')
    ax.set(title=title)
    sns.distplot(data[:, 0:1], color='blue')
    sns.distplot(data[:, 1:2], color='green')
    sns.distplot(data[:, 2:3], color='red')
    plt.show()

# 数据预处理 
# 1. Rescaling (min-max normalization)
# 2. Standardization(Z-score normalization)
def LinkData(filepath): # 拼接数据
    """
    从filepath中读取csv文件，拼接，返回dataframe.values(即ndarray)
    """
    csvs = []
    fh = open(filepath, 'r')
    for line in fh:
        line = line.rstrip()
        paths = line.split()
        csvs.append(paths[0])
    
    for csv in csvs:
        data = pd.read_csv(csv, header=None)
        if(csv == csvs[0]):
            ret = data
        else:
            ret = pd.concat([ret, data], ignore_index=True)
    
    return ret.values        

def MaxMinMeanStd(x):
    """
    返回最大值、最小值、均值、方差(n-1)
    """
    ret = []
    _max = np.max(x, axis=0)
    _min = np.min(x, axis=0)
    _mean = np.mean(x, axis=0)
    _std = np.std(x, axis=0, ddof=1)
    ret = np.array([_max, _min, _mean, _std])
    return ret

def MaxMinNormalize(x, mx):
    """
    暂定为x*3型数据
    x为np数组，mx为x的最值
    对x进行缩放(min max normalization)
    """
    ret = np.zeros_like(x)
    for i in range(3):
        _max = mx[0][i]
        _min = mx[1][i]
        mid = _max - _min
        ret[:, i] = (x[:, i] - _min) / mid
    return ret

def ZScoreNormalize(x, mx):
    """
    标准化，Standardization(Z-score normalization)
    x = (x_i - mean(x)) / sigma
    (sigma指标准差)
    """
    ret = np.zeros_like(x)
    for i in range(3):
        _mean = mx[0][i]
        _std = mx[1][i]
        ret[:, i] = (x[:, i] - _mean) / _std
    return ret

def SK_Standard(x):
    """
    sklearn.preprocessing中的StandardScaler采用的标准差是有偏的
    通过修改scaler的scale_参数，使之变为无偏估计
    """
    scale = preprocessing.StandardScaler()
    scale.fit(x)
    scale.scale_ = np.std(x, axis=0, ddof=1)
    return scale.transform(x)

def SK_scaler(data_l):
    """
    读取data_l(ndarray), 返回两个预处理器：
    1. min max scaler
    2. standard scaler
    """
    # 归一化
    s_mms = preprocessing.MinMaxScaler()
    s_mms.fit(data_l)
    # 标准化
    data_temp = s_mms.transform(data_l)
    s_std = preprocessing.StandardScaler()
    s_std.fit(data_temp)
    s_std.scale_ = np.std(data_temp, axis=0, ddof=1)

    return s_mms, s_std

def DataTrans(x, s_mms, s_std):
    """
    归一化、标准化变换
    x: ndarray
    s_mms: min max scaler
    s_std: standard scaler
    """
    return s_std.transform(s_mms.transform(x))

# 杂项 - 记录网上找的各种方法
def print_object_attr(obj):
    """
    打印对象的所有属性
    """
    print('\n'.join(['%s:%s' % item for item in obj.__dict__.items()]))