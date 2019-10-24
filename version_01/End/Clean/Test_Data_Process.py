#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# !/usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd

"""
测试数据处理
类别编码处理
"""


def Model_Noem_Data(data, writePath):
    list_DQ = data.loc[:, "DQ"].unique()
    list_disease_type_cd = data.loc[:, "disease_type_cd"].unique()
    list_ops_cd = data.loc[:, "ops_cd"].unique()
    for i in range(len(data)):
        hashdataDQ = hash(data["DQ"][i]) % (len(list(list_DQ)))
        data.replace(data["DQ"][i], hashdataDQ, inplace=True)

    for i in range(len(data)):
        hashdataDisease = hash(data["disease_type_cd"][i]) % (len(list(list_disease_type_cd)))
        data.replace(data["disease_type_cd"][i], hashdataDisease, inplace=True)

    for i in range(len(data)):
        hashdataOps = hash(data["ops_cd"][i]) % (len(list(list_ops_cd)))
        data.replace(data["ops_cd"][i], hashdataOps, inplace=True)

    list_hospital_cd = data.loc[:, "hospital_cd"].unique()  # no hash
    list_proj_cd = data.loc[:, "proj_cd"].unique()  # no hash
    for i in range(len(list_hospital_cd)):
        data.loc[:, "hospital_cd"] = data.loc[:, "hospital_cd"]\
            .apply(lambda x: i if x == list_hospital_cd[i] else x)

    for i in range(len(list_proj_cd)):
        data.loc[:, "proj_cd"] = data.loc[:, "proj_cd"]\
            .apply(lambda x: i if x == list_proj_cd[i] else x)
    data.to_csv(writePath)


def Test_Data_Process():
    seepPath = r""
    writeSeepBig = r""
    writeSeepMid = r""
    writeSeepSma = r""

    # 读取数据and乱序
    seepData = pd.DataFrame(pd.read_csv(seepPath, encoding="gbk", error_bad_lines=False).sample(frac=1))
    seepData.fillna(value=0)

    Model_Noem_Data(seepData, writeSeepBig)
    Model_Noem_Data(seepData, writeSeepMid)
    Model_Noem_Data(seepData, writeSeepSma)

Test_Data_Process()
