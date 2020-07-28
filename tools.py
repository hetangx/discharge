import pandas as pd
import numpy as np


# 先写测试脚本，然后封装为方法
# 脚本1 - 分割数据集

# dataset_dir = os.path.join("..")


# 数据预处理 - 采用sklearn

def LinkData(filepath): # 拼接数据
    """
    从filepath中读取csv文件，拼接，返回dataframe.values
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