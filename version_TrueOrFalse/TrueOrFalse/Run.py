#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os

import keras as k
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

"""
训练模型
"""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def Model_Load_Data(CLEAN_DATA_PATH):
    AIYOUData = pd.read_csv(CLEAN_DATA_PATH, encoding="gbk")
    # 目标变量
    targetVar = "flag"
    # 数据集特征
    features = list(AIYOUData.columns)
    features.remove(targetVar)
    # 目标变量的类别
    Class = AIYOUData[targetVar].unique()

    # 目标变量的类别字典
    Class_dic = dict(zip(Class, range(len(Class))))
    # 增加一列target用于编码
    AIYOUData['target'] = AIYOUData[targetVar].apply(lambda x: Class_dic[x])
    # 对目标变量进行0-1编码(One-hot Encoding)
    lb = LabelBinarizer()
    lb.fit(list(Class_dic.values()))
    transformed_labels = lb.transform(AIYOUData['target'])
    # 对多分类进行0-1编码的变量
    y_bin_labels = []
    for i in range(transformed_labels.shape[1]):
        y_bin_labels.append('y' + str(i))
        AIYOUData['y' + str(i)] = transformed_labels[:, i]
        # 将数据集分为训练集和测试集
    train_x, test_x, train_y, test_y = train_test_split(AIYOUData[features], AIYOUData[y_bin_labels],
                                                        train_size=0.7, test_size=0.3, random_state=1)
    return train_x, test_x, train_y, test_y, Class_dic


def Train_Model(Data_Path, Model_Save_path):
    """
    构建神经网络
    :param Data_Path: 原始数据
    :param Model_Save_path: 模型保存路径h5文件
    :return: no return
    """
    np.random.seed(7)
    # tf.set_random_seed(9)
    train_x, test_x, train_y, test_y, Class_dict = Model_Load_Data(Data_Path)
    simple_adam = k.optimizers.Adam(lr=0.00001)
    # simple_adam=k.optimizers.SGD(lr=0.01)
    model = k.Sequential()

    model.add(k.layers.Dense(
        units=16,
        input_dim=8,
        kernel_initializer=k.initializers.glorot_uniform(seed=1),
        activation="tanh"
    ))
    model.add(k.layers.Dense(
        units=4,
        input_dim=16,
        kernel_initializer=k.initializers.glorot_uniform(seed=1),
        activation="tanh"
    ))
    model.add(k.layers.Dense(
        units=3,
        kernel_initializer=k.initializers.glorot_uniform(seed=1),
        activation="softmax"
    ))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=simple_adam,
        metrics=["accuracy"]
    )

    model.fit(
        train_x,
        train_y,
        batch_size=16,
        epochs=10000,
        shuffle=True,
        verbose=1,

    )

    # 模型保存
    model.save(Model_Save_path)
    eval = model.evaluate(test_x, test_y, verbose=0)
    # print("Evaluation on test data: loss = %0.6f accuracy = %0.2f%% \n" \
    #       % (eval[0], eval[1] * 100))
    # 测试数据
    # dataarr = [[14, 1, 1, 296464.35, 0, 3, 10719.62, 4],
    #            [27, 1, 3, 4023161.37, 1, 14, 5967478.06, 0],
    #            [37, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
    #            [41, 0, 5, 6571728.98, 1, 10, 23221395.32, 9],
    #            [3, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
    #            [33, 0, 3, 951102.38, 1, 15, 3114262.86, 0],
    #            [7, 1, 1, 296464.35, 0, 3, 10719.62, 4],
    #            [11, 0, 1, 656844.39, 0, 8, 6745063.26, 0],
    #            [38, 1, 3, 58516.78, 1, 16, 612.69, 0],
    #            [32, 1, 3, 3833327.59, 1, 4, 276.42, 3],
    #            [6, 0, 2, 1460203.65, 1, 10, 21468770.94, 0],
    #            [9, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
    #            [6, 0, 1, 1337067.2, 1, 9, 20675941.72, 1],
    #            [6, 0, 1, 60049.77, 1, 2, 0, 0],
    #            [7, 1, 3, 3833327.59, 1, 4, 276.42, 3],
    #            [17, 1, 3, 951102.38, 1, 15, 3114262.86, 0],
    #            [25, 1, 1, 296464.35, 0, 3, 10719.62, 4],
    #            [19, 1, 3, 4023161.37, 1, 14, 5967478.06, 0],
    #            [10, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
    #            [10, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
    #            [6, 1, 3, 4023161.37, 1, 14, 5967478.06, 0],
    #            [10, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
    #            [6, 0, 3, 58516.78, 1, 16, 612.69, 0],
    #            [27, 1, 1, 296464.35, 0, 3, 10719.62, 4],
    #            [10, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
    #            [8, 1, 1, 296464.35, 0, 3, 10719.62, 4],
    #            [10, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
    #            [29, 1, 1, 296464.35, 0, 3, 10719.62, 4],
    #            [10, 1, 2, 5004073.17, 1, 5, 7199826.8, 0],
    #            [15, 0, 3, 951102.38, 1, 15, 3114262.86, 0],
    #            [7, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
    #            [7, 1, 1, 59062.21, 1, 4, 4259296.82, 0],
    #            [10, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
    #            [7, 1, 2, 1460203.65, 1, 10, 21468770.94, 0],
    #            [6, 0, 1, 60049.77, 1, 2, 0, 0],
    #            [7, 1, 3, 4427045.04, 1, 5, 813.01, 0],
    #            [23, 1, 3, 4023161.37, 1, 14, 5967478.06, 0],
    #            [5, 1, 1, 296464.35, 0, 3, 10719.62, 4],
    #            [27, 1, 3, 4023161.37, 1, 14, 5967478.06, 0],
    #            [7, 1, 1, 296464.35, 0, 3, 10719.62, 4],
    #            [18, 1, 3, 4023161.37, 1, 14, 5967478.06, 0],
    #            [6, 0, 1, 60049.77, 1, 2, 0, 0],
    #            [6, 1, 2, 1460203.65, 1, 10, 21468770.94, 0],
    #            [7, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
    #            [9, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
    #            [3, 0, 2, 5004073.17, 1, 5, 7199826.8, 0],
    #            [19, 1, 1, 296464.35, 0, 3, 10719.62, 4],
    #            [12, 0, 1, 656844.39, 0, 8, 6745063.26, 0],
    #            [35, 1, 3, 4023161.37, 1, 14, 5967478.06, 0],
    #            [15, 1, 1, 296464.35, 0, 3, 10719.62, 4],
    #
    #            ]
    #
    # for i in dataarr:
    #     # np.set_printoptions(precision=4)
    #     unknown = np.array([i], dtype=np.float32)
    #     predicted = model.predict(unknown)
    #     species_dict = {v: k for k, v in Class_dict.items()}
    #     print(species_dict[np.argmax(predicted)])



data_path = str(input("TrueFalse数据地址"))
Model_Save_path = str(input("TFModel模型保存地址"))
Train_Model(data_path, Model_Save_path)
