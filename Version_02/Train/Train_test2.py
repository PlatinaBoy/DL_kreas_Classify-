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
    """
    DQ	:地区
    if_special_sickchildren	：特殊患病儿童x
    if_die	是否死亡x
    if_exist_donor	是否多期已有x
    period	 期次x
    disease_type_cd	    病种
    ops_cd	    手术id
    hospital_cd	    医院id
    proj_cd	    协议id
    need_donate_fee	 钱x
    deduct_amount	钱x
    if_designated  是否指定x
    donate_psn_name  捐赠人
    """
    list_DQ = data.loc[:, "DQ"].unique()
    list_disease_type_cd = data.loc[:, "disease_type_cd"].unique()
    list_ops_cd = data.loc[:, "ops_cd"].unique()

    for i in range(len(data)):
        hashdataDQ = hash(data["DQ"][i]) % (len(list(list_DQ)))
        data.replace(data["DQ"][i], hashdataDQ, inplace=True)

    for i in range(len(data)):
        hashdataDisease = hash(data["disease_type_cd"][i]) % (len(list_disease_type_cd))
        data.replace(data["disease_type_cd"][i], hashdataDisease, inplace=True)

    for i in range(len(data)):
        hashdataOps = hash(data["ops_cd"][i]) % (len(list_ops_cd))
        data.replace(data["ops_cd"][i], hashdataOps, inplace=True)

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
    intdata = 333
    # 1读取数据集
    print("加载数据到内存中......")
    CSV_WRITE_PATH = r""
    train_x, test_x, train_y, test_y, Class_dict = Model_Load_Data(CSV_WRITE_PATH)
    # 2定义模型
    """
    神经元激活函数为ReLU函数，
    损失函数为交叉熵（cross entropy），
    迭代的优化器（optimizer）选择Adam，
    最初各个层的连接权重（weights）和偏重（biases）是随机生成的
    k.layers.advanced_activations.LeakyReLU(alpha=0.3)
    """

    model = k.models.Sequential()

    model.add(k.layers.Dense(
        units=24,
        input_dim=9,
        # kernel_initializer=k.initializers.glorot_uniform(seed=1),
        # activation=k.layers.advanced_activations.LeakyReLU(alpha=0.3),
        activation="tanh"
    ))

    model.add(k.layers.Dropout(0.5))

    model.add(k.layers.Dense(
        units=40,
        kernel_initializer=k.initializers.glorot_uniform(seed=1),
        activation='softmax'
    ))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]

    )
    # 3 训练模型k
    b_size = 10
    max_epochs = 50
    print("开始训练......")
    model.fit(
        train_x,
        train_y,
        batch_size=b_size,
        epochs=max_epochs,
        shuffle=True,
        verbose=1,
    )

    # 4 模型评估
    """
    输出该模型的损失函数的值以及在测试集上的准确率
    """
    eval = model.evaluate(test_x, test_y, verbose=1)
    print("测试数据评估: loss = %0.6f 精确度 = %0.2f%% \n" % (eval[0], (eval[1] * intdata)))
    arrays = [[13, 0, 2, 43, 3735, 5, 0, 1436.27, 0],
              [2, 0, 1, 18, 1021, 7, 1, 40000, 40000],
              [6, 1, 14, 43, 698, 0, 0, 996.7, 0],
              [13, 1, 5, 44, 1566, 5, 0, 3483.59, 0],
              [13, 1, 7, 43, 697, 5, 0, 1804.29, 0],
              [23, 0, 6, 10, 732, 18, 0, 699.6, 0],
              [5, 0, 1, 89, 2273, 12, 1, 10361.25, 10361.25],
              [14, 0, 3, 43, 4378, 7, 0, 4163, 0],
              [11, 0, 1, 56, 3093, 6, 1, 19671.61, 19671.61],
              [12, 0, 1, 37, 2461, 25, 2, 13600, 13600],
              [12, 0, 3, 44, 2810, 32, 0, 4788.42, 0],
              [0, 1, 14, 43, 872, 20, 0, 390.9, 0],
              [5, 0, 1, 89, 4345, 12, 1, 11004.23, 11004.23],
              [1, 0, 1, 23, 3169, 10, 1, 19426.79, 19426.79],
              [16, 0, 1, 43, 2322, 49, 2, 14615.44, 14615.44],
              [2, 0, 7, 43, 3183, 7, 0, 756, 0],
              [11, 0, 1, 56, 666, 6, 1, 27000, 27000],
              [1, 0, 1, 89, 3647, 42, 1, 13728.23, 13728.23],
              [1, 0, 1, 43, 3547, 26, 2, 12752, 12752],
              [0, 0, 1, 23, 2219, 7, 1, 13000, 13000],
              [14, 0, 2, 22, 2127, 27, 0, 2563, 0],
              [8, 0, 1, 56, 2826, 35, 1, 15000, 15000],
              [11, 0, 1, 23, 2171, 6, 1, 5002.23, 5002.23],
              [15, 0, 13, 43, 437, 18, 0, 3271.9, 0],
              [11, 0, 1, 23, 1726, 6, 1, 18956.94, 18956.94],
              [6, 1, 3, 43, 2310, 32, 0, 4053.61, 0],
              [6, 1, 6, 43, 529, 0, 0, 898.26, 0],
              [8, 0, 1, 10, 2161, 35, 1, 11494.49, 11494.49],
              [8, 1, 10, 43, 1464, 18, 0, 398.1, 0],
              [1, 0, 1, 23, 71, 19, 1, 40000, 40000],
              [13, 1, 3, 43, 1324, 5, 0, 1652.42, 0],
              [2, 0, 1, 23, 2035, 19, 1, 11654.42, 11654.42],
              [8, 0, 1, 10, 1600, 35, 1, 11275.78, 11275.78],
              [1, 0, 1, 96, 2785, 2, 1, 4866.13, 4866.13],
              [0, 1, 10, 43, 3988, 20, 0, 3453.37, 0],
              [20, 1, 9, 43, 3677, 3, 0, 1167.84, 0],
              [1, 0, 1, 81, 3201, 28, 1, 13000, 13000],
              [12, 0, 1, 23, 3285, 24, 1, 15830, 15830],
              [1, 1, 10, 43, 2793, 8, 0, 140.81, 0],
              [1, 0, 1, 10, 2086, 42, 1, 11617.6, 11617.6],
              [11, 0, 1, 41, 1076, 6, 1, 11516.24, 11516.24],
              [1, 0, 1, 61, 3401, 11, 1, 27000, 27000],
              [5, 0, 1, 23, 2032, 12, 1, 24294.71, 24294.71],
              [16, 0, 1, 96, 626, 1, 1, 7476.09, 7476.09],
              [1, 0, 1, 86, 613, 42, 1, 11732.27, 11732.27]
              ]

    for i in arrays:
        unknown = np.array([i], dtype=np.float32)
        predicted = model.predict(unknown)
        species_dict = {v: k for k, v in Class_dict.items()}
        print(species_dict[np.argmax(predicted)])
main()