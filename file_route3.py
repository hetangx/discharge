import os

dirPath = "d:\\Codes\\keyan\\peidian\\discharge"
resultFile = "d:\\Codes\\keyan\\peidian\\csv_label_digit.txt"
fresultFile = open(resultFile, 'w') # 打开文件

csvLabel = os.listdir(dirPath) # 得到标签
for i in range(5):
    labelDirPath = dirPath + "\\" + csvLabel[i]
    gDirPath = os.walk(labelDirPath) # 文件名迭代器

    for root, dirs, files in gDirPath:
        for f in files:
            # fresultFile.write(labelDirPath + "\\" + f + ' ' + csvLabel[i] + '\n')
            fresultFile.write(labelDirPath + "\\" + f + ' ' + str(i) + '\n')