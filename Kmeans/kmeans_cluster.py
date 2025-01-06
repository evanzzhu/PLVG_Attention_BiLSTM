# -*- coding:utf-8 -*-
"""
作者:朱昊喆
日期:2024年12月12日
"""
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import csv
from dtw import dtw

# 曼哈顿距离定义，各点相减的绝对值
manhattan_distance = lambda x, y: np.abs(x - y)


def cal_distance(node, centor):
    return dtw(node, centor, dist=manhattan_distance)[0]


def random_centor(data, k):
    data = list(data)
    return random.sample(data, k)


def random_centor1(data, k):
    n = len(data[0])  # n维
    centor = np.array([[0] * n for _ in range(k)])  # 一定要将列表转换为数组
    for j in range(n):
        min_j = np.min(data[:, j])
        max_j = np.max(data[:, j])
        centor[:, j] = np.random.rand(k) * (max_j - min_j) + min_j
    return centor


def get_cluster(data, centor):
    cluster_dict = dict()
    k = len(centor)
    for node in data:
        cluster_class = -1
        min_distance = float('inf')
        for i in range(k):
            dist = cal_distance(node, centor[i])
            if dist < min_distance:
                cluster_class = i
                min_distance = dist
        if cluster_class not in cluster_dict.keys():
            cluster_dict[cluster_class] = []
        cluster_dict[cluster_class].append(node)
    return cluster_dict


def get_centor(cluster_dict, k):
    new_centor = []
    for i in range(k):
        centor = np.mean(cluster_dict[i], axis=0)
        new_centor.append(centor)
    return new_centor


def cal_varience(cluster_dict, centor):
    vsum = 0
    for i in range(len(centor)):
        cluster = cluster_dict[i]
        for j in cluster:
            vsum += cal_distance(j, centor[i])
    return vsum


def k_means(data, k):
    centor = random_centor(data, k)
    # print(centor)
    cluster_dict = get_cluster(data, centor)
    new_varience = cal_varience(cluster_dict, centor)
    old_varience = 1
    while abs(old_varience - new_varience) > 0.1:
        print(old_varience, new_varience)
        centor = get_centor(cluster_dict, k)
        cluster_dict = get_cluster(data, centor)
        old_varience = new_varience
        new_varience = cal_varience(cluster_dict, centor)
    return cluster_dict, centor

fig, ax = plt.subplots(2, 3)
time = []
for i in range(8):
    time.append(i)
light_k = 4
input_path_light = "../Data_pre/train_load_full_dim2_8.csv"
df = pd.read_csv(input_path_light, encoding='gb2312')
df = df.drop('Time', axis=1)
#df = df.drop('class', axis=1)
df.fillna(df.mean(), inplace=True)
# df = df.set_index('Time').interpolate(method='linear', axis=0)
x = df.to_numpy()
a, b = k_means(x, light_k)
print(b)
ax[0][0].set_ylabel("Energy consumption(kW*h)")
ax[0][0].set_xlabel("Time(hour)")
ax[0][0].set_title("Class1")
ax[0][2].set_ylabel("Energy consumption(kW*h)")
ax[0][2].set_xlabel("Time(hour)")
ax[0][2].set_title("Class3")
ax[0][1].set_ylabel("Energy consumption(kW*h)")
ax[0][1].set_xlabel("Time(hour)")
ax[0][1].set_title("Class2")
ax[1][0].set_ylabel("Energy consumption(kW*h)")
ax[1][0].set_xlabel("Time(hour)")
ax[1][0].set_title("Class4")
ax[1][1].set_ylabel("Energy consumption(kW*h)")
ax[1][1].set_xlabel("Time(hour)")
ax[1][1].set_title("Class5")
ax[1][2].set_ylabel("Energy consumption(kW*h)")
ax[1][2].set_xlabel("Time(hour)")
ax[1][2].set_title("Class6")
# ax[1][1].set_ylabel("Energy consumption(kW*h)")
# ax[1][1].set_xlabel("Time(hour)")
# ax[1][1].set_title("Lighting load cluster result")
y_max=50
ax[0][0].set_ylim((0, y_max))
ax[0][1].set_ylim((0, y_max))
ax[0][2].set_ylim((0, y_max))
#ax[0][3].set_ylim((0, 300))
ax[1][2].set_ylim((0, y_max))
ax[1][0].set_ylim((0, y_max))
ax[1][1].set_ylim((0, y_max))
#ax[1][3].set_ylim((0, 300))
for i in range(light_k):
    for data in a[i]:
        if i == 0:
            # ax[1][1].plot(time, data, 'b:')
            ax[0][0].plot(time, data, 'b:')
        elif i == 1:
            # ax[1][1].plot(time, data, 'r:')
            ax[0][1].plot(time, data, 'r:')
        elif i == 2:
            # ax[1][1].plot(time, data, 'g:')
            ax[0][2].plot(time, data, 'g:')
        elif i == 3:
            ax[1][0].plot(time, data, 'c:')
        elif i == 4:
            ax[1][1].plot(time, data, 'm:')
        elif i == 5:
            ax[1][2].plot(time, data, 'y:')

plt.show()
