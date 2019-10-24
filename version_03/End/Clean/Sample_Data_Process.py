#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd

"""
样本数据处理
对分类数据进行样本编码
"""


def Model_Noem_Data(data, writePath):
    list_DQ = data.loc[:, "DQ"].unique()
    for i in range(len(data)):
        hashdataDQ = hash(data["DQ"][i]) % (len(list(list_DQ)))
        data.replace(data["DQ"][i], hashdataDQ, inplace=True)
    # no hash
    list_hospital_cd = data.loc[:, "hospital_cd"].unique()
    # no hash
    list_proj_cd = data.loc[:, "proj_cd"].unique()

    for i in range(len(list_hospital_cd)):
        data.loc[:, "hospital_cd"] = data.loc[:, "hospital_cd"].apply(lambda x: i if x == list_hospital_cd[i] else x)

    for i in range(len(list_proj_cd)):
        data.loc[:, "proj_cd"] = data.loc[:, "proj_cd"].apply(lambda x: i if x == list_proj_cd[i] else x)
    data.to_csv(writePath)


def Sample_Data_Process():
    bigPath = r""
    midPath = r""
    smaPath = r""
    writeBig = r""
    writeMid = r""
    writeSma = r""
    # 读取数据and乱序
    bigData = pd.DataFrame(pd.read_csv(bigPath, encoding="gbk", error_bad_lines=False).sample(frac=1))
    midData = pd.DataFrame(pd.read_csv(midPath, encoding="gbk", error_bad_lines=False).sample(frac=1))
    smaData = pd.DataFrame(pd.read_csv(smaPath, encoding="gbk", error_bad_lines=False).sample(frac=1))
    bigData.fillna(value=0)
    midData.fillna(value=0)
    smaData.fillna(value=0)
    Model_Noem_Data(bigData, writeBig)
    Model_Noem_Data(midData, writeMid)
    Model_Noem_Data(smaData, writeSma)


Sample_Data_Process()
