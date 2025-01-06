# -*- coding:utf-8 -*-
"""
作者:朱昊喆
日期:2023年09月13日
"""

# %config InlineBackend.figure_format = 'retina'
# %matplotlib inline
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.utils.data as Data
# from torch.autograd import Variable
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# from sklearn.metrics import cohen_kappa_score
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
# import torch.nn.functional as F
# import csv
#
# ##模型保存
# # torch.save(model,"PATH")
# ##模型加载
# word_model = KeyedVectors.load("../model/word2vec")
# dtype = torch.FloatTensor
# embedding_dim = 10
# n_hidden = 10
# num_classes = 6  # 0 or 1
# # sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
# # labels = [1, 1, 1, 0, 0, 0]
# # word_list = " ".join(sentences).split()
# # word_list = list(set(word_list))
# # word_dict = {w: i for i, w in enumerate(word_list)}
# vocab_size = 10
# flag = True
# batch_size = 32
#
#
# class BiLSTM_Attention(nn.Module):
#     def __init__(self):
#         super(BiLSTM_Attention, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
#         self.out = nn.Linear(n_hidden * 2, num_classes)
#
#     def attention_net(self, lstm_output, final_state):
#         hidden = final_state.view(-1, n_hidden * 2, 1)
#         attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
#         soft_attn_weights = F.softmax(attn_weights, 1)
#         context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
#         return context, soft_attn_weights.data.numpy()
#
#     def forward(self, X):
#         # print(X)
#         input = self.embedding(X)
#         # print(input)
#         # input = []
#         # for i in X:
#         #     tmp = []
#         #     for j in i:
#         #         tmp.append(word_model.wv.get_vector(str(int(j))))
#         #     input.append(tmp)
#         # input = torch.Tensor(input)
#         # print(input)
#         input = input.permute(1, 0, 2)
#         hidden_state = Variable(torch.zeros(1 * 2, len(X), n_hidden))
#         cell_state = Variable(torch.zeros(1 * 2, len(X), n_hidden))
#         output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
#         output = output.permute(1, 0, 2)
#         attn_output, attention = self.attention_net(output, final_hidden_state)
#         return self.out(attn_output), attention
#
#
# def evaluation(y_test, y_predict):
#     accuracy = classification_report(y_test, y_predict, output_dict=True)['accuracy']
#     s = classification_report(y_test, y_predict, output_dict=True)['weighted avg']
#     precision = s['precision']
#     recall = s['recall']
#     f1_score = s['f1-score']
#     # kappa=cohen_kappa_score(y_test, y_predict)
#     return accuracy, precision, recall, f1_score  # , kappa
#
#
# seq_index = []
# test_seq_index = []
# with open('load_class_8_10.txt', 'r') as f:
#     content = f.read()
#     for i in content:
#         if i != ' ':
#             seq_index.append(int(i))
#         # seq_index.append(int(i))
#     # print(seq_index)
# # print(seq_index)
# seq_length = len(seq_index)
# window = 22
# # 生成输入数据
# batch_x = []
# batch_y = []
# for i in range(0, seq_length - window + 1, 2):
#     if i + window + 1 >= seq_length:
#         # y = word2index[' ']
#         break
#         pass
#     else:
#         y = seq_index[i + window + 1]
#         # print(y)
#     x = seq_index[i: i + window + 1]
#     # print(x)
#
#     batch_x.append(x)
#     batch_y.append(y)
# with open('load_class_8_10_test.txt', 'r') as f:
#     content = f.read()
#     for i in content:
#         if i != ' ':
#             test_seq_index.append(int(i))
#         # seq_index.append(int(i))
#     # print(test_seq_index)
# # print(test_seq_index[:10])
# seq_length = len(test_seq_index)
# window = 22
# # 生成输入数据
# test_batch_x = []
# test_batch_y = []
# for i in range(0, seq_length - window + 1, 2):
#     if i + window + 1 >= seq_length:
#         # y = word2index[' ']
#         break
#         pass
#     else:
#         y = test_seq_index[i + window + 1]
#         # print(y)
#     x = test_seq_index[i: i + window + 1]
#     # print(x)
#
#     test_batch_x.append(x)
#     test_batch_y.append(y)
# # X_train = batch_x[:int(0.8 * len(batch_x))]
# # y_train = batch_y[:int(0.8 * len(batch_y))]
# # X_test = batch_x[int(0.8 * len(batch_x)):]
# # y_test = batch_y[int(0.8 * len(batch_y)):]
# X_train = batch_x
# y_train = batch_y
# X_test = test_batch_x
# y_test = test_batch_y
# print('test_seq_index:',test_seq_index[:50])
# print('train', X_train[:10], y_train[:10])
# print('test', X_test[:10], y_test[:10])
# # 训练数据
# batch_x, batch_y = Variable(torch.LongTensor(X_train)), Variable(torch.LongTensor(y_train))
# batch_x_test, batch_y_test = Variable(torch.LongTensor(X_test)), Variable(torch.LongTensor(y_test))
# dataset = Data.TensorDataset(batch_x, batch_y)
# loader = Data.DataLoader(dataset, batch_size, shuffle=True)
#
# test_dataset = Data.TensorDataset(batch_x_test, batch_y_test)
# test_loader = Data.DataLoader(test_dataset, batch_size, shuffle=False)
# model = torch.load("../model/bilstm5.pkl")
# ##设置模型进行测试模式
# model.eval()
# with torch.no_grad():
#     for data, target in test_loader:
#         # print(data, target)
#         # if torch.cuda.is_available():
#         #     data = data
#         #     target = target
#         outputs, attention = model(data)
#         # print(outputs)
#         prediction = []
#         for i in outputs:
#             # print(i)
#             prediction.append(torch.max(i, 0)[1].item())
#         prediction = torch.tensor(prediction)
#         # print(prediction)
#         # loss = criterion(prediction, target)
#         # total_test_loss += loss.item()
#         # prediction = prediction.detach().cpu().numpy()
#         # label = target.detach().cpu().numpy()
#         # accuracy = np.sum(np.abs(prediction - label) < 0.1) / label.size
#         # rmse = np.sqrt(np.mean((prediction - label) ** 2))
#         # rmse_list.append(rmse)
#         # act_build_energy.append(target.cpu())
#         # pre_build_energy.append(prediction)
#         # target = target.cpu()
#         # prediction
#         if flag:
#             # print(f"act:{act_build_energy}")
#             # print(f"pre:{pre_build_energy}")
#             act_build_energy = target.reshape(-1)
#             pre_build_energy = prediction.reshape(-1)
#             flag = False
#         else:
#             # print(f"act:{act_build_energy}")
#             # print(f"pre:{pre_build_energy}")
#             act_build_energy = torch.cat([act_build_energy, target.reshape(-1)], 0)
#             pre_build_energy = torch.cat([pre_build_energy, prediction.reshape(-1)], 0)
# print(type(act_build_energy), type(pre_build_energy))
# # act_build_energy = act_build_energy + 1
# # pre_build_energy = pre_build_energy + 1
# print(act_build_energy)
# print(pre_build_energy)
# # act_build_energy, pre_build_energy = act_build_energy.numpy(), pre_build_energy.numpy()
# # for i in range(len(act_build_energy)):
# #     act_build_energy[i] = act_build_energy[i] % 4
# #     pre_build_energy[i] = pre_build_energy[i] % 4
# # accuracy, precision, recall, f1_score = evaluation(act_build_energy, pre_build_energy)
# accuracy, precision, recall, f1_score = evaluation(np.array(act_build_energy), np.array(pre_build_energy))
# print(f"accuracy:{accuracy},precision:{precision},recall:{recall},f1_score:{f1_score}")
# out_path = "../final_data/load_data_test_all_8_10.csv"
# w_csv = csv.writer(open(out_path, 'w', encoding='gb2312', newline=''))
# input_path_ele = "../final_data/load_data_test.csv"
# sum = 0
# w_csv.writerow(['DATE', 'LOAD', 'TEMP', 'CLASS', 'CLASS_PRE'])
# test_sum=1
# with open(input_path_ele, "r", encoding='gb2312') as f:
#     reader = csv.reader(f)
#     curr = 0
#     curr1 = 0
#     for row in reader:
#         # print(row)
#         if row[0] == 'DATE':
#             continue
#         date, time = row[0].split(" ")
#         if sum <= 21:
#             # print(row)
#             if time == '0:00' or time == '8:00' or time == '16:00':
#                 curr = int(test_seq_index[test_sum])
#                 row.append(str(curr))
#                 row.append(str(curr))
#                 sum += 1
#                 test_sum+=2
#                 print(row, sum)
#             else:
#                 row.append(curr)
#                 row.append(curr)
#         else:
#             if time == '0:00' or time == '8:00' or time == '16:00':
#                 curr = pre_build_energy[sum - 22].item()
#                 curr1 = act_build_energy[sum - 22].item()
#                 row.append(curr)
#                 row.append(curr1)
#                 sum += 1
#                 print(row, sum)
#             else:
#                 row.append(curr)
#                 row.append(curr1)
#         w_csv.writerow(row)
#         # ele_dict[float(row[1])] = row[0]
#         # print(type(row[1]))
# print(len(seq_index))
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
# from umap import UMAP
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from matplotlib import rcParams
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 32
# 加载Word2Vec模型
model = KeyedVectors.load("model/word2vec_W")
word_dict = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm',
             14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y',
             26: 'z'}
# 获取词向量和词汇表
word_vectors = model.wv.vectors
vocab = model.wv.index_to_key
print(word_vectors)
print(vocab)
for i in range(len(vocab)):
    int_vocab=int(vocab[i])
    if int_vocab>=11:
        vocab[i]=word_dict[int_vocab-10]
# 使用t-SNE将词向量降至二维
tsne = TSNE(n_components=2, random_state=42, perplexity=15)
tsne_results = tsne.fit_transform(word_vectors)
load_key = ['1', '2', '3', '4']
temp_key = ['6','7', '8', '9', '10']
week_key = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
hour_key = ['h', 'i', 'j']
fig, axs = plt.subplots(1, 1, figsize=(5, 5))
load = []
load_vocab = []
temp = []
temp_vocab = []
week = []
week_vocab = []
hour = []
hour_vocab = []
# print(vocab)

config = {
    "font.family": 'serif',
    "font.size": 24,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


def encircle(x, y, ax=None, **kw):
    if not ax: ax = plt.gca()
    p = np.c_[x, y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices, :], **kw)  # 画多边形
    ax.add_patch(poly)  # 将图形添加到图中


for i in range(len(vocab)):
    if vocab[i] in load_key:
        load.append(tsne_results[i])
        load_vocab.append(vocab[i])
    if vocab[i] in temp_key:
        temp.append(tsne_results[i])
        temp_vocab.append(vocab[i])
    if vocab[i] in week_key:
        week.append(tsne_results[i])
        week_vocab.append(vocab[i])
    if vocab[i] in hour_key:
        hour.append(tsne_results[i])
        hour_vocab.append(vocab[i])
load = np.array(load)
temp = np.array(temp)
week = np.array(week)
hour = np.array(hour)
print(load)
print(temp)
print(week)
print(hour)
# print(type(tsne_results))
axs.scatter(load[:, 0], load[:, 1], alpha=1, s=75, c='b')
axs.scatter(temp[:, 0], temp[:, 1], alpha=1, s=75, c='tomato')
axs.scatter(week[:, 0], week[:, 1], alpha=1, s=75, c='g')
axs.scatter(hour[:, 0], hour[:, 1], alpha=1, s=75, c='purple')
# axs.set_title('t-SNE Visualization')
# axs[1].scatter(umap_results[:,0], umap_results[:,1], alpha=0.5)
# axs[1].set_title('UMAP Visualization')
det_y = 1
del_x = 1
fontsize=24
for i, word in enumerate(load_vocab):
    axs.annotate(word, (load[i, 0] + del_x, load[i, 1] + det_y), color='b', fontsize=fontsize, weight='bold')
for i, word in enumerate(temp_vocab):
    axs.annotate(word, (temp[i, 0] + del_x, temp[i, 1] + det_y), color='tomato', fontsize=fontsize, weight='bold')
for i, word in enumerate(week_vocab):
    axs.annotate(word, (week[i, 0] + del_x, week[i, 1] + det_y), color='g', fontsize=fontsize, weight='bold')
for i, word in enumerate(hour_vocab):
    axs.annotate(word, (hour[i, 0] + del_x, hour[i, 1] + det_y), color='purple', fontsize=fontsize, weight='bold')
    # axs[1].annotate(word, (umap_results[i,0], umap_results[i,1]))
encircle(load[:, 0], load[:, 1], ax=axs, ec="b", fc="lightskyblue", alpha=0.2)
encircle(temp[:, 0], temp[:, 1], ax=axs, ec="r", fc="tomato", alpha=0.2)
encircle(week[:, 0], week[:, 1], ax=axs, ec="g", fc="g", alpha=0.2)
encircle(hour[:, 0], hour[:, 1], ax=axs, ec="purple", fc="purple", alpha=0.2)
plt.show()
# plt.savefig("show_word2vec.svg", dpi=300, format="svg")
