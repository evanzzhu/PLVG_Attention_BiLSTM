# -*- coding:utf-8 -*-
"""
作者:朱昊喆
日期:2024年12月18日
"""
# -*- coding:utf-8 -*-
"""
作者:朱昊喆
日期:2023年09月13日
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

##模型保存
# torch.save(model,"PATH")
##模型加载
word_model = KeyedVectors.load("model/word2vec_W")
dtype = torch.FloatTensor
embedding_dim = 21
n_hidden = 64
num_classes = 6  # 0 or 1
vocab_size = 21
flag = True
batch_size = 1

word_dict = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm',
             14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y',
             26: 'z'}
word_dict_reverse = {}
for i in range(26):
    word_dict_reverse[word_dict[i + 1]] = i + 1


class BiLSTM_Attention(nn.Module):
    def __init__(self):
        super(BiLSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(int(0.5 * embedding_dim), n_hidden, bidirectional=True)
        self.out = nn.Linear(n_hidden * 2, num_classes)
        # self.cnn = nn.Conv1d(input_size, out_channel, kernel_size=3, padding=1)
        self.cnn = nn.Conv1d(embedding_dim, int(0.5 * embedding_dim), kernel_size=3, padding=1)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, n_hidden * 2, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy()

    def forward(self, X):
        input = []
        for i in X:
            tmp = []
            for j in i:
                tmp.append(word_model.wv.get_vector(str(int(j))))
            input.append(tmp)
        input = torch.Tensor(input)
        # print(input)
        input = input.permute(0, 2, 1)
        # print(input_seq.shape)
        input = self.cnn(input)
        input = input.permute(0, 2, 1)
        input = input.permute(1, 0, 2)
        hidden_state = Variable(torch.zeros(1 * 2, len(X), n_hidden))
        cell_state = Variable(torch.zeros(1 * 2, len(X), n_hidden))
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2)
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output), attention


def evaluation(y_test, y_predict):
    print(classification_report(y_test, y_predict))
    accuracy = classification_report(y_test, y_predict, output_dict=True)['accuracy']
    s = classification_report(y_test, y_predict, output_dict=True)['weighted avg']
    precision = s['precision']
    recall = s['recall']
    f1_score = s['f1-score']
    return accuracy, precision, recall, f1_score  # , kappa


seq_index = []
test_seq_index = []
with open('label_train_dh.txt', 'r') as f:
    content = f.read()
    tmp_line = content.split('.')
    print(tmp_line)
    for i in tmp_line:
        j = i.split(' ')
        tmp = []
        for word in j:
            if word == '':
                break
            if word in word_dict_reverse.keys():
                seq_index.append(int(word_dict_reverse[word] + 10))
            else:
                seq_index.append(int(word))
print(seq_index[:5])
seq_length = len(seq_index)
window = 18
# 生成输入数据
batch_x = []
batch_y = []

for i in range(0, seq_length - window + 1, 4):
    if i + window + 1 >= seq_length:
        # y = word2index[' ']
        break
        pass
    else:
        y = seq_index[i + window + 1]
        # print(y)
    x = seq_index[i: i + window + 1]
    # print(x)

    batch_x.append(x)
    batch_y.append(y)
with open('label_test_dh.txt', 'r') as f:
    content = f.read()
    tmp_line = content.split('.')
    print(tmp_line)
    for i in tmp_line:
        j = i.split(' ')
        tmp = []
        for word in j:
            if word == '':
                break
            if word in word_dict_reverse.keys():
                test_seq_index.append(int(word_dict_reverse[word] + 10))
            else:
                test_seq_index.append(int(word))
seq_length = len(test_seq_index)
# window = 18
# 生成输入数据
test_batch_x = []
test_batch_y = []
for i in range(0, seq_length - window + 1, 4):
    if i + window + 1 >= seq_length:
        break
        pass
    else:
        y = test_seq_index[i + window + 1]
    x = test_seq_index[i: i + window + 1]

    test_batch_x.append(x)
    test_batch_y.append(y)
X_train = batch_x
y_train = batch_y
X_test = test_batch_x
y_test = test_batch_y
print('test_seq_index:', test_seq_index[:50])
print('trainX', X_train[:10])
print('trainY', y_train[:10])
print('testX', X_test[:10])
print('testY', y_test[:10])
# 训练数据
batch_x, batch_y = Variable(torch.LongTensor(X_train)), Variable(torch.LongTensor(y_train))
batch_x_test, batch_y_test = Variable(torch.LongTensor(X_test)), Variable(torch.LongTensor(y_test))
dataset = Data.TensorDataset(batch_x, batch_y)
loader = Data.DataLoader(dataset, batch_size, shuffle=True)

test_dataset = Data.TensorDataset(batch_x_test, batch_y_test)
test_loader = Data.DataLoader(test_dataset, batch_size, shuffle=False)
model = BiLSTM_Attention()
model.load_state_dict(torch.load("model/bilstm_best_W.pkl"))
##设置模型进行测试模式
model.eval()
with torch.no_grad():
    step = 0
    first_pre = 0
    second_pre = 0
    for data, target in test_loader:
        if step == 0:
            pass
        elif step == 1:
            data[0][window - 3] = first_pre
        elif step == 2:
            data[0][window - 7] = first_pre
            data[0][window - 3] = second_pre
        elif step == 3:
            step = 0
        outputs, attention = model(data)
        prediction = []
        for i in outputs:
            prediction.append(torch.max(i, 0)[1].item())
        prediction = torch.tensor(prediction)
        if step == 0:
            first_pre = prediction
        elif step == 1:
            second_pre = prediction
        elif step == 2:
            first_pre = 0
            second_pre = 0
        step += 1
        if flag:
            act_build_energy = target.reshape(-1)
            pre_build_energy = prediction.reshape(-1)
            flag = False
        else:
            act_build_energy = torch.cat([act_build_energy, target.reshape(-1)], 0)
            pre_build_energy = torch.cat([pre_build_energy, prediction.reshape(-1)], 0)
# print(type(act_build_energy), type(pre_build_energy))
# print(act_build_energy)
# print(len(pre_build_energy))
# print(pre_build_energy)
accuracy, precision, recall, f1_score = evaluation(np.array(act_build_energy), np.array(pre_build_energy))
print(f"accuracy:{accuracy},precision:{precision},recall:{recall},f1_score:{f1_score}")
out_path = "../final_data/load_data_test_all_8_10.csv"
w_csv = csv.writer(open(out_path, 'w', encoding='gb2312', newline=''))
input_path_ele = "../Data_pre/test_load.csv"
sum = 0
w_csv.writerow(['DATE', 'LOAD', 'TEMP', 'CLASS', 'CLASS_PRE'])
test_sum=3
with open(input_path_ele, "r", encoding='gb2312') as f:
    reader = csv.reader(f)
    curr = 0
    curr1 = 0
    for row in reader:
        if row[0] == 'Date':
            continue

        date, time = row[0].split(" ")
        if sum <= window-1:
            if time == '0:00' or time == '8:00' or time == '16:00':
                curr = int(test_seq_index[test_sum])
                row.append(str(curr))
                row.append(str(curr))
                sum += 1
                test_sum+=4
            else:
                row.append(curr)
                row.append(curr)
        else:
            if time == '0:00' or time == '8:00' or time == '16:00':
                curr = pre_build_energy[sum - window].item()
                curr1 = act_build_energy[sum - window].item()
                row.append(curr)
                row.append(curr1)
                sum += 1
            else:
                row.append(curr)
                row.append(curr1)
        w_csv.writerow(row)
print(len(seq_index))
