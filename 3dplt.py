import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def CsvPlot3D(path):
    csv = pd.read_csv(path, header=None)
    csv = np.array(csv.values)
    csv = csv.T

    A = csv[:1] # 幅值
    P = csv[1:2] # 相位
    F = csv[2:3] # 次数

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P,F,A)
    plt.show()

path = r"D:\Codes\keyan\data\internal_discharge\t12.csv"
CsvPlot3D(path)