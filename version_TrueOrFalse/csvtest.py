#!/usr/bin/env python 
# -*- coding:utf-8 -*-
"""
csv 操作测试
"""
import csv
# print(read_text_data(r""))
data_path = r"C:\Users\Wuzg\Desktop\PeapleData.csv"
Model_Save_path = r"C:\Users\Wuzg\Desktop\PpModel.h5"
# data_path = str(input("PeapleData数据地址"))
# Model_Save_path = str(input("PpModel模型保存地址"))
# data = input("查询历史数据"))

# f = open(r'C:\Users\Wuzg\Desktop\text.csv', 'r')  # 打开csv文件
# csv_reader = csv.reader(f)  # 将打开的文件装换成csv可读的对象
# # for each in csv_reader:             # 打印,结果是个列表
# #     print(each)
# line=f.readline()
#
# while line :
#
#     print(line[33:], end='')
#     line = f.readline()[1:]
# f.close()
#
# # filename=r'C:\Users\Wuzg\Desktop\text.csv'
# import pandas as pd
# odata = pd.read_csv(filename,encoding="gbk")
# y = odata['child_id']
# x = odata.drop(['child_id'], axis=1) #除去label列之外的所有feature值
def read_text_data(path):
    f = open(path, 'r')

    next(f)
    line=f.readline()
    arr=[]
    while line:
        lines=line[33:].split(",")
        listeee=map(eval,lines)
        listees=list(listeee)
        arr.append(listees)
        line=f.readline()
    return arr


print(read_text_data(r""))
import sys
import keras as K
import tensorflow as tf

py_ver = sys.version
k_ver = K.__version__
tf_ver = tf.__version__

print("Using Python version " + str(py_ver))
print("Using Keras version " + str(k_ver))
print("Using TensorFlow version " + str(tf_ver))



