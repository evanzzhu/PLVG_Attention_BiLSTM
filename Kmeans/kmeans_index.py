# -*- coding:utf-8 -*-
"""
作者:朱昊喆
日期:2024年12月12日
"""
# -*- coding:utf-8 -*-
"""
作者:朱昊喆
日期:2023年04月05日
"""
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import calinski_harabasz_score, pairwise_distances, davies_bouldin_score
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from dtw import dtw
import random
from tslearn.clustering import TimeSeriesKMeans

# 曼哈顿距离定义，各点相减的绝对值
manhattan_distance = lambda x, y: np.abs(x - y)


def cal_distance(node, centor):
    return np.sqrt(np.sum(np.square(node - centor)))
    # return dtw(node, centor, dist=manhattan_distance)[0]


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


def get_calinski_harabasz(X, labels):
    n_samples = X.shape[0]
    n_clusters = np.unique(labels).shape[0]
    betw_disp = 0.  # 所有的簇间距离和
    within_disp = 0.  # 所有的簇内距离和
    global_centroid = np.mean(X, axis=0)  # 全局簇中心
    for k in range(n_clusters):  # 遍历每一个簇
        x_in_cluster = X[labels == k]  # 取当前簇中的所有样本
        centroid = np.mean(x_in_cluster, axis=0)  # 计算当前簇的簇中心
        within_disp += np.sum((x_in_cluster - centroid) ** 2)
        betw_disp += len(x_in_cluster) * np.sum((centroid - global_centroid) ** 2)
    return 1. if within_disp == 0. else betw_disp * (n_samples - n_clusters) / (within_disp * (n_clusters - 1.))


ch_value = []
ch_num = []


def test_calinski_harabasz_score(model_dict):
    for i in range(2, 9):
        # dba_kmeans = TimeSeriesKMeans(n_clusters=i, metric="euclidean", verbose=True, random_state=42)
        # model = dba_kmeans.fit(x)
        y_pred = model_dict[str(i)].predict(x)
        print(x, y_pred)
        print(f"方差比 by sklearn: {calinski_harabasz_score(x, y_pred)}")
        print(f"方差比 by ours: {get_calinski_harabasz(x, y_pred)}")
        ch_value.append(calinski_harabasz_score(x, y_pred))
        ch_num.append(i)


def get_davies_bouldin(X, labels):
    n_clusters = np.unique(labels).shape[0]
    centroids = np.zeros((n_clusters, len(X[0])), dtype=float)
    s_i = np.zeros(n_clusters)
    for k in range(n_clusters):  # 遍历每一个簇
        x_in_cluster = X[labels == k]  # 取当前簇中的所有样本
        centroids[k] = np.mean(x_in_cluster, axis=0)  # 计算当前簇的簇中心
        s_i[k] = pairwise_distances(x_in_cluster, [centroids[k]]).mean()  #
    centroid_distances = pairwise_distances(centroids)  # [K,K]
    combined_s_i_j = s_i[:, None] + s_i  # [K,k]
    centroid_distances[centroid_distances == 0] = np.inf
    scores = np.max(combined_s_i_j / centroid_distances, axis=1)
    return np.mean(scores)


db_value = []
db_num = []


def test_davies_bouldin_score(model_dict):
    for i in range(2, 9):
        # dba_kmeans = TimeSeriesKMeans(n_clusters=i, metric="euclidean", verbose=True, random_state=42)
        # model = dba_kmeans.fit(x)
        y_pred = model_dict[str(i)].predict(x)
        print(x, y_pred)
        print(f"db_score by sklearn: {davies_bouldin_score(x, y_pred)}")
        print(f"db_score by ours: {get_davies_bouldin(x, y_pred)}")
        db_value.append(davies_bouldin_score(x, y_pred))
        db_num.append(i)


if __name__ == '__main__':
    model_dict1 = {}
    x = []
    for i in range(2, 9):
        print(f"第{i}次迭代")
        input_path_ele = "../Data_pre/train_load_full_dim2_8.csv"
        df = pd.read_csv(input_path_ele, encoding='gb2312')
        df = df.drop('Time', axis=1)
        # df = df.drop('class', axis=1)
        df.fillna(df.mean(), inplace=True)
        x = df.to_numpy()
        dba_kmeans = TimeSeriesKMeans(n_clusters=i, metric="dtw", verbose=True, random_state=65)
        model_dict1[str(i)] = dba_kmeans.fit(x)
    test_calinski_harabasz_score(model_dict1)
    test_davies_bouldin_score(model_dict1)
    plt.plot(ch_num, ch_value, label="CH")
    print(ch_num)
    print(ch_value)
    plt.legend()
    # plt.ylim((0, 20000))
    plt.show()
    # #db_value[0]=0.74
    plt.plot(db_num, db_value, label="DB")
    plt.legend()
    # plt.ylim((0, 20000))
    plt.show()
    score = []
    for i in range(2, 9):
        score.append(silhouette_score(x, model_dict1[str(i)].labels_))
    plt.show()
    plt.plot(range(2, 9), score, label="SC")
    plt.xlim([2, 9])
    # 给k最大的位置加虚线   #idxmax()[] 取最大索引  因为这边从k=2开始 所以+2
    plt.axvline(pd.DataFrame(score).idxmax()[0] + 2, ls=':')  # 加一条虚线方便查看
    plt.legend()
    plt.show()
