# -*- coding:utf-8 -*-
"""
作者:朱昊喆
日期:2024年12月17日
"""
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import csv
from chinese_calendar import is_workday, is_holiday, get_holiday_detail, is_in_lieu
from dtw import dtw
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator

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
center = []
for i in model_centroid:
    tmp = []
    for j in i:
        tmp.append(j[0])
    center.append(tmp)
print(center)

manhattan_distance = lambda x, y: np.abs(x - y)
load_ch_dtw_109 =[7753.645706653793, 5220.665153798438, 5463.720535366485, 5492.5512596166645, 5478.2255066965945, 5097.2226168023235, 5178.7173155857945, 4947.123175227886]
temp_ch_train = [12541.134331464355, 13336.999398509837, 14178.568946385876, 14308.094800160656, 14234.41385256925,
                 13722.61299839044, 13048.999774364514, 12326.353962169493]

def cal_distance(node, centor):
    return dtw(node, centor, dist=manhattan_distance)[0]


def get_class(node, centor):
    cluster_dict = dict()
    k = len(centor)
    cluster_class = -1
    min_distance = float('inf')
    for i in range(k):
        dist = cal_distance(node, centor[i])
        if dist < min_distance:
            cluster_class = i
            min_distance = dist
    # if cluster_class not in cluster_dict.keys():
    #     cluster_dict[cluster_class] = []
    # cluster_dict[cluster_class].append(node)
    return cluster_class


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
    return year, month, day, hour, minute


input_path = "../Data_pre/train_load_dim2_8.csv"
center_dict={}
for i in range(len(center)):
    center_dict[str(i)]=[]
with open(input_path, "r", encoding='gb2312') as f:
    reader = csv.reader(f)
    for row in reader:
        load_row = row[1:]
        if row[1] == '0:00':
            continue
        for i in range(len(load_row)):
            load_row[i] = float(load_row[i])
        print(load_row)
        class_num = get_class(load_row, center)
        center_dict[str(class_num)].append(load_row)
        print(class_num)
        row.append(str(class_num))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24
gs = gridspec.GridSpec(2, 6)
gs.update(hspace=0.3,wspace=1.2)
ax0 = plt.subplot(gs[0, :2])
ax1 = plt.subplot(gs[0, 2:4])
ax2 = plt.subplot(gs[0, 4:6])
ax3 = plt.subplot(gs[1, :2])
ax4 = plt.subplot(gs[1, 2:4])
ax5 = plt.subplot(gs[1, 4:6])
#plt.figure(figsize=(6,4))
time = []
for i in range(8):
    time.append(i)

y_max=3000
# ax0.grid(True)
ax0.xaxis.set_major_locator(MultipleLocator(1))
ax0.set_ylabel("CH")
ax0.set_xlabel("$K_{L}$")

ax1.set_ylim((0, y_max))
ax1.set_ylabel("Energy consumption(kW*h)")
ax1.set_xlabel("Time(hour)")
ax1.xaxis.set_major_locator(MultipleLocator(1))

ax2.set_ylim((0, y_max))
ax2.set_ylabel("Energy consumption(kW*h)")
ax2.set_xlabel("Time(hour)")
ax2.xaxis.set_major_locator(MultipleLocator(1))

ax3.set_ylim((0, y_max))
ax3.set_ylabel("Energy consumption(kW*h)")
ax3.set_xlabel("Time(hour)")
ax3.xaxis.set_major_locator(MultipleLocator(1))

#ax[0][3].set_ylim((0, 300))
ax4.set_ylim((0, y_max))
ax4.set_ylabel("Energy consumption(kW*h)")
ax4.set_xlabel("Time(hour)")
ax4.xaxis.set_major_locator(MultipleLocator(1))

ax5.set_ylim((0, y_max))
ax5.set_ylabel("Energy consumption(kW*h)")
ax5.set_xlabel("Time(hour)")
ax5.xaxis.set_major_locator(MultipleLocator(1))

ax0.plot(range(2, 9), load_ch_dtw_109[:], label="CH", color='blue',lw=2,marker='*')
ax0.scatter(range(2, 9), load_ch_dtw_109[:], color='blue')
#ax[1][1].set_ylim((0, y_max))
for i in range(len(center)):
    if i == 0:
        # ax[1][1].plot(time, data, 'b:')
        for j in center_dict[str(i)]:
            ax1.plot(time, j, 'silver',lw=0.25)
        ax1.plot(time, center[i], 'blue',lw=2,marker='*')
    elif i == 1:
        for j in center_dict[str(i)]:
            ax2.plot(time, j, 'silver',lw=0.25)
        ax2.plot(time, center[i], 'royalblue',lw=2,marker='*')
    elif i == 2:
        for j in center_dict[str(i)]:
            ax3.plot(time, j, 'silver',lw=0.25)
        ax3.plot(time, center[i], 'cornflowerblue',lw=2,marker='*')
    elif i == 3:
        for j in center_dict[str(i)]:
            ax4.plot(time, j, 'silver',lw=0.25)
        ax4.plot(time, center[i], 'deepskyblue',lw=2,marker='*')
    elif i == 4:
        for j in center_dict[str(i)]:
            ax5.plot(time, j, 'silver',lw=0.25)
        ax5.plot(time, center[i], 'skyblue',lw=2,marker='*')
plt.show()
