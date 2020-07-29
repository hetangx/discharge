import os
# 扫描data文件夹下各种放电类型的csv文件，并保存路径与类型至txt文件

dirPath = "D:\\Codes\\keyan\\discharge"
resultFile = "D:\\Codes\\keyan\\CONV\\csv_label_digit.txt"
f = open(resultFile, 'w') # 打开文件

csvLabel = os.listdir(dirPath) # 得到标签
for i in range(5):
    labelDirPath = dirPath + "\\" + csvLabel[i]
    gDirPath = os.walk(labelDirPath) # 文件名迭代器

    for root, dirs, files in gDirPath:
        for f in files:
            # f.write(labelDirPath + "\\" + f + ' ' + csvLabel[i] + '\n') 
            f.write(labelDirPath + "\\" + f + ' ' + str(i) + '\n') 