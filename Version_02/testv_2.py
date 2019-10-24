#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os

import keras as k
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def Model_Noem_Data():
    CSV_READ_PATH = r""
    CSV_WRITE_PATH = r""
    data = pd.DataFrame(pd.read_csv(CSV_READ_PATH, encoding="gbk", error_bad_lines=False)
                        .sample(frac=1))
    data.fillna(value=0)
    list_DQ = data.loc[:, "DQ"].unique()

    for i in range(len(data)):
        hashdataDQ = hash(data["DQ"][i]) % (len(list(list_DQ)))
        data.replace(data["DQ"][i], hashdataDQ, inplace=True)
    list_hospital_cd = data.loc[:, "hospital_cd"].unique()  # no hash
    list_proj_cd = data.loc[:, "proj_cd"].unique()  # no hash
    for i in range(len(list_hospital_cd)):
        data.loc[:, "hospital_cd"] = data.loc[:, "hospital_cd"].apply(lambda x: i if x == list_hospital_cd[i] else x)

    for i in range(len(list_proj_cd)):
        data.loc[:, "proj_cd"] = data.loc[:, "proj_cd"].apply(lambda x: i if x == list_proj_cd[i] else x)
    data.to_csv(CSV_WRITE_PATH)

    return CSV_WRITE_PATH


def Model_Load_Data(CSV_WRITE_PATH):
    AIYOUData = pd.read_csv(CSV_WRITE_PATH, encoding="gbk")
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
                                                        train_size=0.7, test_size=0.3, random_state=1)
    return train_x, test_x, train_y, test_y, Class_dict


def main():
    # 0开始
    print("\nTensorFlow 开始运行......")
    np.random.seed(7)
    tf.set_random_seed(13)
    intdata = 2933
    # 1读取数据集
    print("加载数据到内存中......")
    CSV_WRITE_PATH = r""
    train_x, test_x, train_y, test_y, Class_dict = Model_Load_Data(CSV_WRITE_PATH)
    # 2定义模型
    # init = k.initializers.glorot_uniform(seed=1)
    # simple_adam = k.optimizers.Adam()
    # model = k.models.Sequential()
    # model.add(k.layers.Dense(units=30, input_dim=9, kernel_initializer=init, activation='relu'))
    # model.add(k.layers.Dense(units=60, kernel_initializer=init, activation='relu'))
    # model.add(k.layers.Dense(units=40, kernel_initializer=init, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer=simple_adam, metrics=['accuracy'])
    model = k.models.Sequential()

    model.add(k.layers.Dense(
        units=24,
        input_dim=8,
        activation="tanh"
    ))
    model.add(k.layers.Dropout(0.5))

    model.add(k.layers.Dense(
        units=40,
        activation='softmax'
    ))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]

    )
    # 3 训练模型k
    b_size = 10
    max_epochs = 100
    print("开始训练......")
    model.fit(
        train_x,
        train_y,
        batch_size=b_size,
        epochs=max_epochs,
        shuffle=True,
        verbose=1,
    )

    arrays = [[5, 0, 1, 0, 0, 15424.03, 15424.03, 0],
              [18, 0, 1, 1, 0, 15747, 15747, 0],
              [18, 0, 1, 2, 0, 32418, 32418, 0],
              [18, 0, 1, 3, 0, 40000, 40000, 0],
              [12, 0, 1, 4, 0, 16769.63, 16769.63, 0],
              [5, 0, 1, 5, 0, 27000, 27000, 0],
              [21, 0, 1, 6, 1, 30000, 30000, 0],
              [21, 0, 1, 7, 0, 11235.85, 11235.85, 0],
              [21, 0, 1, 8, 0, 11611, 11611, 0],
              [0, 0, 1, 9, 0, 9600.48, 9600.48, 0],
              [21, 0, 1, 10, 0, 8928.05, 8928.05, 0],
              [19, 1, 17, 11, 2, 522.41, 0, 0],
              [18, 0, 1, 12, 0, 12500.1, 12500.1, 0],
              [27, 0, 1, 13, 1, 23436, 23436, 0],
              [15, 0, 1, 14, 0, 8291.9, 8291.9, 0],
              [28, 0, 1, 15, 0, 21000, 21000, 0],
              [24, 0, 1, 16, 1, 9294, 9294, 0],
              [18, 0, 1, 17, 0, 8089.4, 8089.4, 0],
              [5, 0, 1, 5, 0, 27000, 27000, 0],
              [18, 0, 1, 18, 0, 10850, 10850, 0],
              [28, 0, 1, 19, 0, 15533, 15533, 0],
              [17, 0, 1, 20, 0, 30000, 30000, 0],
              [1, 0, 1, 21, 0, 11494.49, 11494.49, 0],
              [18, 0, 1, 12, 0, 12500, 12500, 0],
              [28, 1, 12, 22, 2, 14299.37, 0, 0],
              [28, 0, 1, 23, 1, 49999.98, 49999.98, 0],
              [18, 0, 1, 24, 0, 13000, 13000, 0],
              [5, 0, 1, 12, 0, 10202.92, 10202.92, 0],
              [21, 0, 1, 25, 0, 20359.46, 20359.46, 0],
              [28, 1, 17, 26, 2, 830.32, 0, 0],
              [28, 0, 1, 23, 1, 28296, 28296, 0],
              [28, 1, 5, 22, 2, 1029.14, 0, 0],
              [9, 1, 7, 27, 2, 932.78, 0, 0],
              [28, 1, 7, 22, 2, 1891.22, 0, 0],
              [9, 0, 1, 27, 0, 13462.87, 13462.87, 0],
              [22, 0, 1, 16, 0, 14299, 14299, 0],
              [9, 0, 1, 27, 0, 37710.37, 37710.37, 0],
              [19, 0, 1, 16, 1, 16715.05, 16715.05, 0],
              [28, 0, 1, 2, 0, 13000, 13000, 0],
              [9, 0, 1, 27, 0, 40000, 40000, 0],
              [0, 0, 1, 9, 0, 18186.53, 18186.53, 0],
              [15, 0, 1, 28, 0, 40000, 40000, 0],
              [28, 1, 16, 22, 2, 1676.15, 0, 0],
              [18, 0, 1, 17, 0, 6078.28, 6078.28, 0],
              [14, 1, 7, 29, 2, 1340.86, 0, 0],
              [0, 0, 1, 9, 0, 9843.05, 9843.05, 0],
              [1, 1, 17, 29, 2, 286.86, 0, 0],
              [9, 1, 3, 27, 2, 9425.82, 0, 0],
              [12, 0, 1, 4, 0, 14913.66, 14913.66, 0],
              [5, 0, 1, 0, 0, 22173.46, 22173.46, 0],

              ]

    for i in arrays:
        unknown = np.array([i], dtype=np.float32)
        predicted = model.predict(unknown)
        species_dict = {v: k for k, v in Class_dict.items()}
        print(species_dict[np.argmax(predicted)])


main()
