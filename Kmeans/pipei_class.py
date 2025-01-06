# -*- coding:utf-8 -*-
"""
作者:朱昊喆
日期:2024年12月13日
"""
import pandas as pd
import numpy as np
import csv
from datetime import datetime
from dtw import dtw

# 曼哈顿距离定义，各点相减的绝对值
manhattan_distance = lambda x, y: np.abs(x - y)
# 聚类中心
model_centroid = [[[185.60849151],
                   [168.42261458],
                   [160.63817468],
                   [158.22154006],
                   [158.29424686],
                   [154.77930703],
                   [145.29962121],
                   [127.58038238]]

                  ,[[89.23237274],
    [101.10565838],
    [97.4372905],
    [93.52228662],
    [95.52185053],
    [95.43923513],
    [98.41886679],
    [113.19478946]],

                  [[138.78856239],
    [160.79077239],
    [173.52868284],
    [180.42183365],
    [180.53592417],
    [181.61752097],
    [193.04119298],
    [219.21448701]],

                  [[127.57096606],
    [119.85835173],
    [130.12726161],
    [131.07672956],
    [131.93156597],
    [142.61914406],
    [135.53013699],
    [116.56878026]],

                  [[252.13559055],
    [231.75085803],
    [223.42957746],
    [223.83703704],
    [224.48174442],
    [223.65452675],
    [211.62914923],
    [188.6946865]]]
centers = []
for i in model_centroid:
    tmp = []
    for j in i:
        tmp.append(j[0])
    centers.append(tmp)
print(centers)
# 读取数据
dataset = 'test'
type_name = 'load'
kmeans_cal = 'dtw'
df = pd.read_csv('../Data_pre/'+dataset+'_'+type_name+'_full_dim2_8.csv', parse_dates=['Time'])


def cal_distance(node, centor):
    # return np.sqrt(np.sum(np.square(node - centor)))
    return dtw(node, centor, dist=manhattan_distance)[0]


# 计算欧氏距离并找到最小的距离对应的标签
def assign_label(row, centers):
    # 提取数据中的数值部分（忽略时间）
    values = row[1:].values
    distances = [cal_distance(values, center) for center in centers]
    return np.argmin(distances)  # 返回距离最小的聚类中心索引


# 添加新的列 'load_label'
df['load_label'] = df.apply(lambda row: assign_label(row, centers), axis=1)
df.fillna(df.mean(), inplace=True)
print(df)
# 保存新的CSV文件
df.to_csv(dataset+'_'+type_name+'_full_dim2_8_'+kmeans_cal+'.csv', index=False)
#
# # 查看结果
# print(df.head())
# 读取 cleaned_combined_load.csv 文件
input_filename = '../Data_pre/'+dataset+'_load_full.csv'
output_filename = dataset+'_'+type_name+'_with_labels_full_'+kmeans_cal+'.csv'
w_csv = csv.writer(open(output_filename, 'w', encoding='gb2312', newline=''))
i = 0

target_format = '%Y-%m-%d %H:%M:%S'


# 打开 CSV 文件并读取数据
def convert_date_format(date_str, original_format, target_format):
    # 将原始格式的日期字符串转换为 datetime 对象
    date_obj = datetime.strptime(date_str, original_format)
    # 将 datetime 对象格式化为目标格式的字符串
    return date_obj.strftime(target_format)


w_csv.writerow(['DATE', 'LOAD', 'Temp', 'P0', 'P', 'wed', 'wind', 'cloud', 'see', 'Td', 'CLASS'])
with open(input_filename, mode='r', newline='') as infile:
    reader = csv.reader(infile)
    rows = reader
    print(df.iloc[i])
    next_time = df.iloc[i + 1]['Time']
    for row in rows:
        if row[0] == 'Date':
            continue
        converted_date1 = convert_date_format(row[0], "%Y/%m/%d %H:%M", target_format)
        print(converted_date1, next_time)
        if str(converted_date1) == str(next_time):
            i = i + 1
            next_time = df.iloc[i + 1]['Time']
            print('---------------')
        row.append(df.iloc[i]['load_label'])
        print(row)
        w_csv.writerow(row)
        pass
    i += 1
print(f"处理完成，结果已保存到 {output_filename}")
