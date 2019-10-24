# !/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import numpy as np
import keras as k
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras import metrics


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def Model_Norm_Data(CSV_READ_PATH,CSV_WRITE_PATH):
    """
    数据处理：包括处理空字段，对分类较多的数据做特征哈希处理
    对目标字段做one_hot  编码
    """
    data = pd.DataFrame(pd.read_csv(CSV_READ_PATH, encoding="gbk", error_bad_lines=False))
    data.fillna(value=0)
    #显示所有列
    pd.set_option('display.max_columns', None)
    #显示所有行
    pd.set_option('display.max_rows', None)
    #设置value的显示长度为100，默认为50
    pd.set_option('max_colwidth',100)
    # hash listUdq = data.loc[:, "donate_psn_name"].unique()
    list_DQ = data.loc[:, "DQ"].unique()  # hash
    list_disease_type_cd = data.loc[:, "disease_type_cd"].unique()  # hash
    list_ops_cd = data.loc[:, "ops_cd"].unique()  # hash
    for i in range(len(data)):
        hashdataDQ = hash(data["DQ"][i]) % (len(list_DQ))
        data.replace(data["DQ"][i], hashdataDQ, inplace=True)

    for i in range(len(data)):
        hashdataDisease = hash(data["disease_type_cd"][i]) % (len(list_disease_type_cd))
        data.replace(data["disease_type_cd"][i], hashdataDisease, inplace=True)

    for i in range(len(data)):
        hashdataOps = hash(data["ops_cd"][i]) % (len(list_ops_cd))
        data.replace(data["ops_cd"][i], hashdataOps, inplace=True)

    # nohash listUdq = data.loc[:, "donate_psn_name"].unique()
    list_batch_cd = data.loc[:, "batch_cd"].unique()  # nohash
    list_hospital_cd = data.loc[:, "hospital_cd"].unique()  # no hash
    list_proj_cd = data.loc[:, "proj_cd"].unique()  # no hashi

    for i in range(len(list_batch_cd)):
        data.loc[:, "batch_cd"] = data.loc[:, "batch_cd"].apply(lambda x: i if x == list_batch_cd[i] else x)
    for i in range(len(list_hospital_cd)):
        data.loc[:, "hospital_cd"] = data.loc[:, "hospital_cd"].apply(lambda x: i if x == list_hospital_cd[i] else x)
    for i in range(len(list_proj_cd)):
        data.loc[:, "proj_cd"] = data.loc[:, "proj_cd"].apply(lambda x: i if x == list_proj_cd[i] else x)
    data.to_csv(CSV_WRITE_PATH)
    return CSV_WRITE_PATH



def Model_Load_Data(CSV_WRITE_PATH):
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    # 设置value的显示长度为100，默认为50
    pd.set_option('max_colwidth', 100)
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
                                                        train_size=0.7, test_size=0.3, random_state=0)
    return train_x, test_x, train_y, test_y, Class_dict


def main():
    # 0开始
    print("\nTensorFlow 开始运行......")
    np.random.seed(1)
    tf.set_random_seed(13)
    # 1读取数据集
    print("加载数据到内存中......")
    CSV_WRITE_PATH = r"C:\Users\Wuzg\Desktop\Result.csv"
    train_x, test_x, train_y, test_y, Class_dict = Model_Load_Data(CSV_WRITE_PATH)
    # 2定义模型
    init = k.initializers.glorot_uniform(seed=1)

    simple_adam = k.optimizers.Adam()

    model = k.models.Sequential()
    """

    """
    model.add(k.layers.Dense(
                units=17,
                input_dim=15,
                kernel_initializer=init,
                activation=k.layers.advanced_activations.LeakyReLU(alpha=0.3)
    ))


    model.add(k.layers.Dense(
                units=18,
                kernel_initializer=init,
                activation=k.layers.advanced_activations.LeakyReLU(alpha=0.3)
    ))

    model.add(k.layers.Dense(
                units=14,
                kernel_initializer=init,
                activation='softmax'
    ))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=simple_adam,
        metrics=[metrics.mae, metrics.categorical_accuracy]
    )

    # 3 训练模型k
    # 批处理参数
    b_size = 1
    max_epochs = 10000
    print("开始训练......")
    h = model.fit(
        train_x,
        train_y,
        batch_size=b_size,
        epochs=max_epochs,
        shuffle=True, verbose=1
    )
    print("训练完成......")
    # 4 模型评估
    eval = model.evaluate(test_x, test_y, verbose=1)
    print("测试数据评估: loss = %0.6f 精确度 = %0.2f%% \n" % (eval[0], eval[1] * 100))




main()


