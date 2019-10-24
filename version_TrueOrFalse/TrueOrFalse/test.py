#!/usr/bin/env python
# -*- coding:utf-8 -*-
from keras.models import load_model
import numpy as np
import pandas as pd

"""
模型的使用
"""
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

data_path = str(input("TrueFalse数据地址"))
Model_Save_path = str(input("TFModel模型保存地址"))
target_path=str(input("目标数据地址"))

def Model_Use(data_path, Model_path,target_path):
    """
    模型使用
    :param data_path: 数据地址
    :param Model_path: 模型地址
    :return: no return
    """
    AIYOUData = pd.read_csv(data_path, encoding="gbk")
    # 目标变量
    target_var = "flag"
    # 数据集特征
    features = list(AIYOUData.columns)
    features.remove(target_var)
    # 目标变量的类别
    Class = AIYOUData[target_var].unique()
    # 目标变量的类别字典
    Class_dict = dict(zip(Class, range(len(Class))))
    # 增加一列target用于编码
    load_modell = load_model(Model_path)

    dataarr =read_text_data(target_path)
    for i in dataarr:
        unknown = np.array([i], dtype=np.float32)
        predicted = load_modell.predict(unknown)
        species_dict = {v: k for k, v in Class_dict.items()}
        print(species_dict[np.argmax(predicted)])


Model_Use(data_path, Model_Save_path,target_path)
