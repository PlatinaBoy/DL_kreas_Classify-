#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import pandas
import pandas_profiling
import math
import numpy

"""
生成数据报告.html
这可真的太香了，真的好用
"""
data = pandas.read_csv(r"USEDATA.csv", encoding="gbk")
result = data.describe()
profile = data.profile_report(title='DataEvaluation')
profile.to_file(output_file=r'数据评估.html')
