# -*- coding:utf-8 -*-
"""
作者:朱昊喆
日期:2024年12月19日
"""
# -*- coding:utf-8 -*-
"""
作者:朱昊喆
日期:2023年11月01日
"""
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from gensim.models import Word2Vec, KeyedVectors
import torch.nn.functional as F
import csv
from datetime import datetime


def get_date(data):
    pos = data.find("/")
    year = data[:pos]
    data = data[pos + 1:]
    pos = data.find("/")
    month = data[:pos]
    data = data[pos + 1:]
    pos = data.find(" ")
    day = data[:pos]
    data = data[pos + 1:]
    pos = data.find(":")
    hour = data[:pos]
    minute = data[pos + 1:]
    # print(f"year:{year},month:{month},day:{day},hour:{hour},minute{minute}")
    return year, month, day, hour, minute


out_path = "../final_data/load_data_train_all_8_10_d.csv"
w_csv = csv.writer(open(out_path, 'w', encoding='gb2312', newline=''))
input_path_ele = "../final_data/load_data_train_all_8_10.csv"
w_csv.writerow(['DATE', 'LOAD', 'TEMP', 'CLASS', 'CLASS_PRE', 'WEEK', 'HOUR'])
with open(input_path_ele, "r", encoding='gb2312') as f:
    reader = csv.reader(f)
    for row in reader:
        # print(row)
        if row[0] == 'DATE':
            continue
        year, month, day, hour, minute = get_date(row[0])

        row.append(str(datetime.date(datetime(year=int(year), month=int(month), day=int(day))).weekday() + 1))
        row.append(hour)
        w_csv.writerow(row)
