#!/usr/bin/env python
# -*- coding:utf-8 -*-
from keras.models import load_model
import numpy as np
import pandas as pd

AIYOUData = pd.read_csv(r""
                        , encoding="gbk")
target_var = "donate_psn_name"
features = list(AIYOUData.columns)
features.remove(target_var)
Class = AIYOUData[target_var].unique()
Class_dict = dict(zip(Class, range(len(Class))))
load_modell = load_model(r"C:\Users\Wuzg\Desktop\aiyou.h5")
dataarr = [[18, 1, 3, 4023161.37, 1, 14, 5967478.06, 0],
           [15, 1, 3, 951102.38, 1, 15, 3114262.86, 0],
           [2, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
           [32, 1, 1, 296464.35, 0, 3, 10719.62, 4],
           [10, 1, 2, 1460203.65, 1, 10, 21468770.94, 0],
           [14, 1, 1, 296464.35, 0, 3, 10719.62, 4],
           [37, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
           [2, 1, 3, 58516.78, 1, 16, 612.69, 0],
           [19, 1, 0, 0, 0, 0, 0, 0],
           [10, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
           [2, 1, 3, 4427045.04, 1, 5, 813.01, 0],
           [2, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
           [34, 0, 1, 150764.38, 0, 3, 1528439.86, 0],
           [38, 1, 3, 58516.78, 1, 16, 612.69, 0],
           [25, 1, 1, 296464.35, 0, 3, 10719.62, 4],
           [10, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
           [21, 1, 3, 4023161.37, 1, 14, 5967478.06, 0],
           [7, 1, 2, 1460203.65, 1, 10, 21468770.94, 0],
           [38, 1, 3, 58516.78, 1, 16, 612.69, 0],
           [11, 1, 0, 0, 0, 0, 0, 0],
           [6, 0, 3, 4023161.37, 1, 14, 5967478.06, 0],
           [10, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
           [2, 1, 3, 4023161.37, 1, 14, 5967478.06, 0],
           [9, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
           [27, 1, 5, 1893620.01, 1, 7, 1500000, 11],
           [14, 0, 5, 6571728.98, 1, 10, 23221395.32, 9],
           [9, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
           [10, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
           [11, 1, 0, 0, 0, 0, 0, 0],
           [7, 1, 1, 0, 0, 4, 5000000, 0],
           [14, 1, 1, 296464.35, 0, 3, 10719.62, 4],
           [33, 0, 1, 296464.35, 0, 3, 10719.62, 4],
           [7, 1, 3, 58516.78, 1, 16, 612.69, 0],
           [19, 1, 0, 0, 0, 0, 0, 0],
           [10, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
           [6, 0, 1, 1337067.2, 1, 9, 20675941.72, 1],
           [8, 1, 1, 279041.52, 0, 2, 349.74, 0],
           [6, 1, 2, 2727554.05, 0, 3, 10719.62, 4],
           [10, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
           [10, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
           [3, 0, 3, 4427045.04, 1, 5, 813.01, 0],
           [7, 1, 3, 4427045.04, 1, 5, 813.01, 0],
           [15, 0, 4, 126990.55, 0, 15, 47584.54, 0],
           [19, 1, 1, 296464.35, 0, 3, 10719.62, 4],
           [10, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
           [10, 1, 5, 6571728.98, 1, 10, 23221395.32, 9],
           [32, 1, 1, 296464.35, 0, 3, 10719.62, 4],
           [15, 1, 3, 951102.38, 1, 15, 3114262.86, 0],
           [14, 0, 3, 951102.38, 1, 15, 3114262.86, 0],
           [7, 1, 1, 6482878.52, 0, 28, 3325056.81, 5],
           ]
for i in dataarr:
    np.set_printoptions(precision=4)
    unknown = np.array([i], dtype=np.float32)
    predicted = load_modell.predict(unknown)
    species_dict = {v: k for k, v in Class_dict.items()}
    print(species_dict[np.argmax(predicted)])
