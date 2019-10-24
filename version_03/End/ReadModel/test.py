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
    target_var = "donate_psn_name"
    # 数据集特征
    features = list(AIYOUData.columns)
    features.remove(target_var)
    # 目标变量的类别
    Class = AIYOUData[target_var].unique()

    # 目标变量的类别字典
    Class_dict = dict(zip(Class, range(len(Class))))
    # 增加一列target用于编码
    AIYOUData['target'] = AIYOUData[target_var].apply(lambda x: Class_dict[x])
    # 对目标变量进行0-1编码(One-hot Encoding)
    lb = LabelBinarizer()
    lb.fit(list(Class_dict.values()))
    transformed_labels = lb.transform(AIYOUData['target'])
    # 对多分类进行0-1编码的变量
    y_bin_labels = []
    for i in range(transformed_labels.shape[1]):
        y_bin_labels.append('y' + str(i))
        AIYOUData['y' + str(i)] = transformed_labels[:, i]
        # 将数据集分为训练集和测试集
    train_x, test_x, train_y, test_y = train_test_split(AIYOUData[features], AIYOUData[y_bin_labels],
                                                        train_size=0.6, test_size=0.4, random_state=1)
    return train_x, test_x, train_y, test_y, Class_dict


def Train_Model(Data_Path, outnum):
    np.random.seed(7)
    tf.set_random_seed(9)
    train_x, test_x, train_y, test_y, Class_dict = Model_Load_Data(Data_Path)
    model = k.Sequential()

    model.add(k.layers.Dense(
        units=25,
        input_dim=10,
        kernel_initializer=k.initializers.glorot_uniform(seed=1),
        activation="tanh"
    ))
    model.add(k.layers.Dense(
        units=outnum,
        kernel_initializer=k.initializers.glorot_uniform(seed=1),
        activation="softmax"
    ))
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    model.fit(
        train_x,
        train_y,
        batch_size=8,
        epochs=10000,
        shuffle=True,
        verbose=1,
    )

    dataarr = [
        [22, 0, 1, 20, 2279, 4, 1, 3324.41, 40000, 0],
        [20, 0, 1, 42, 1549, 9, 0, 40000, 40000, 0],
        [8, 0, 1, 83, 670, 10, 0, 9240.44, 9240.44, 0],
        [5, 0, 1, 8, 559, 2, 0, 20925.07, 20925.07, 0],
        [7, 0, 1, 46, 547, 11, 0, 50000, 50000, 0],
        [7, 0, 1, 45, 451, 14, 0, 9000, 9000, 0],
        [22, 1, 11, 20, 156, 4, 1, 2954.96, 0, 1],
        [27, 0, 1, 20, 2246, 1, 1, 40000, 40000, 0],
        [8, 0, 1, 45, 380, 10, 0, 9677.63, 9677.63, 0],
        [22, 0, 1, 20, 1949, 23, 1, 5887.86, 40000, 0],
        [22, 1, 4, 20, 632, 4, 1, 2416.52, 0, 0],
        [27, 0, 1, 20, 660, 26, 1, 12915.9, 40000, 0],
        [27, 0, 1, 80, 574, 28, 0, 18000, 18000, 0],
        [27, 0, 1, 8, 477, 28, 0, 27555, 27555, 0],
        [22, 1, 2, 20, 1640, 32, 1, 7018.71, 0, 0],
        [27, 0, 1, 83, 639, 33, 0, 17500, 17500, 0],
        [27, 0, 1, 45, 566, 3, 0, 12500, 12500, 0],
        [13, 0, 2, 33, 1361, 26, 1, 19434.19, 0, 0],
        [22, 1, 15, 20, 1675, 4, 1, 1918.47, 0, 1],
        [27, 0, 1, 45, 1886, 28, 0, 36006, 36006, 0],
        [16, 1, 17, 20, 1482, 38, 1, 228.98, 0, 0],
        [21, 0, 2, 20, 904, 26, 1, 796.29, 0, 0],
        [13, 0, 1, 5, 287, 23, 1, 2204.07, 40000, 0],
        [21, 0, 1, 20, 130, 32, 1, 40000, 40000, 0],
        [23, 0, 8, 20, 696, 2, 1, 6247.07, 0, 0],
        [13, 1, 11, 20, 1988, 2, 1, 1485.35, 0, 0],
        [27, 0, 1, 83, 1141, 33, 0, 27000, 27000, 0],
        [12, 1, 15, 20, 1998, 38, 1, 692, 0, 0],
        [5, 1, 12, 54, 744, 23, 1, 2765.15, 0, 0],
        [5, 0, 1, 72, 298, 35, 0, 20404.18, 20404.18, 0],
        [21, 1, 9, 84, 1945, 26, 1, 4912.84, 0, 0],
        [5, 0, 1, 83, 161, 35, 0, 6364.31, 6364.31, 0],
        [13, 0, 1, 35, 1274, 28, 0, 29301, 29301, 0],
        [21, 1, 6, 20, 1671, 26, 1, 539.84, 0, 0],
        [27, 0, 1, 8, 1921, 28, 0, 33147, 33147, 0],
        [27, 0, 1, 45, 12, 28, 0, 21000, 21000, 0],
        [18, 0, 1, 83, 1007, 10, 0, 9875.76, 9875.76, 0],
        [27, 0, 1, 84, 1755, 26, 1, 25021.86, 40000, 0],
        [8, 0, 1, 81, 2335, 10, 0, 4868.19, 4868.19, 0],
        [1, 0, 5, 20, 767, 2, 1, 574.4, 0, 0],
        [18, 0, 1, 79, 2173, 10, 0, 15274.19, 15274.19, 0],
        [10, 0, 1, 45, 628, 14, 0, 9000, 9000, 0],
        [7, 0, 1, 83, 1480, 14, 0, 10400, 10400, 0],
        [7, 0, 1, 45, 1374, 14, 0, 19658.11, 19658.11, 0],
        [27, 0, 1, 42, 2188, 33, 0, 9000, 9000, 0],
        [13, 1, 13, 20, 569, 2, 1, 1079.71, 0, 0],
        [7, 0, 1, 45, 2151, 14, 0, 15767.71, 15767.71, 0],

    ]
    for i in dataarr:
        unknown = np.array([i], dtype=np.float32)
        predicted = model.predict(unknown)
        species_dict = {v: k for k, v in Class_dict.items()}
        print(species_dict[np.argmax(predicted)])


writeBig = r""
writeMid = r""
writeSma = r""

Train_Model(writeBig, 5)
