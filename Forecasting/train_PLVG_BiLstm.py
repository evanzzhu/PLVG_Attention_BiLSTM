# -*- coding:utf-8 -*-
"""
作者:朱昊喆
日期:2023年04月27日
"""
# GA优化lstm的遗传算法部分
# import LSTM as ga
import numpy as np
import pandas as pd
import matplotlib as plt
import os
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import random
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
import logging
import torch.nn.functional as F

# 不显示警告信息
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 设置遗传算法的参数
DNA_size = 1
DNA_size_max = 2  # 每条染色体的长度
POP_size = 20  # 种群数量
CROSS_RATE = 0.5  # 交叉率
MUTATION_RATE = 0.01  # 变异率
N_GENERATIONS = 10  # 迭代次数
logging.basicConfig(filename='log/BILSTM_parameter_none_8_10_dh_step1.log', level=logging.INFO)

df = pd.read_csv('../final_data/load_data_train_all_8_10_d.csv', encoding='gb2312')
df_test = pd.read_csv('../final_data/load_data_test_all_8_10_d.csv', encoding='gb2312')
MAX_LOAD = np.max(df['LOAD']) if np.max(df['LOAD']) > np.max(df_test['LOAD']) else np.max(df_test['LOAD'])
MAX_TEMP = np.max(df['TEMP']) if np.max(df['TEMP']) > np.max(df_test['TEMP']) else np.max(df_test['TEMP'])
MAX_CLASS = np.max(df['CLASS']) if np.max(df['CLASS']) > np.max(df_test['CLASS']) else np.max(df_test['CLASS'])

MIN_LOAD = np.min(df['LOAD']) if np.min(df['LOAD']) < np.min(df_test['LOAD']) else np.min(df_test['LOAD'])
MIN_TEMP = np.min(df['TEMP']) if np.min(df['TEMP']) < np.min(df_test['TEMP']) else np.min(df_test['TEMP'])
MIN_CLASS = np.min(df['CLASS']) if np.min(df['CLASS']) < np.min(df_test['CLASS']) else np.min(df_test['CLASS'])

print(f"MAX_LOAD:{MAX_LOAD},MIN_LOAD:{MIN_LOAD}")
print(f"MAX_TEMP:{MAX_TEMP},MIN_TEMP:{MIN_TEMP}")
print(f"MAX_CLASS:{MAX_CLASS},MIN_CLASS:{MIN_CLASS}")


def evaluation(y_test, y_predict):
    global MAX_LOAD, MIN_LOAD
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    y_pre_t = y_predict * (MAX_LOAD - MIN_LOAD) + MIN_LOAD
    y_te_t = y_test * (MAX_LOAD - MIN_LOAD) + MIN_LOAD
    mape = (abs(y_pre_t - y_te_t) / y_te_t).mean()
    print(y_predict, y_test)
    r_2 = r2_score(y_test, y_predict)
    return mae, rmse, mape, r_2


def load_data(file_name):
    global MAX, MIN
    global MAX_LOAD, MAX_TEMP, MAX_CLASS, MIN_LOAD, MIN_TEMP, MIN_CLASS
    df = pd.read_csv(file_name, encoding='gb2312')
    columns = df.columns
    df.fillna(df.mean(), inplace=True)
    MAX = np.max(df['LOAD'])
    MIN = np.min(df['LOAD'])
    print(f"MAX:{MAX},MIN:{MIN}")
    # (MAX, MIN)
    df['LOAD'] = (df['LOAD'] - MIN_LOAD) / (MAX_LOAD - MIN_LOAD)

    MAX = np.max(df['TEMP'])
    MIN = np.min(df['TEMP'])
    # (MAX, MIN)
    print(f"MAX:{MAX},MIN:{MIN}")
    df['TEMP'] = (df['TEMP'] - MIN_TEMP) / (MAX_TEMP - MIN_TEMP)

    MAX = np.max(df['CLASS'])
    MIN = np.min(df['CLASS'])
    print(f"MAX:{MAX},MIN:{MIN}")
    df['CLASS'] = (df['CLASS'] - MIN_CLASS) / (MAX_CLASS - MIN_CLASS)

    MAX = np.max(df['WEEK'])
    MIN = np.min(df['WEEK'])
    print(f"MAX:{MAX},MIN:{MIN}")
    df['WEEK'] = (df['WEEK'] - MIN) / (MAX - MIN)

    MAX = np.max(df['HOUR'])
    MIN = np.min(df['HOUR'])
    # (MAX, MIN)
    print(f"MAX:{MAX},MIN:{MIN}")
    df['HOUR'] = (df['HOUR'] - MIN) / (MAX - MIN)
    print(df)
    return df


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


day_len = 24
model_centroid = [[[185.60849151],
                   [168.42261458],
                   [160.63817468],
                   [158.22154006],
                   [158.29424686],
                   [154.77930703],
                   [145.29962121],
                   [127.58038238]]

    , [[89.23237274],
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
center = np.array(center)
center = (center - MIN_LOAD) / (MAX_LOAD - MIN_LOAD)
print(center)


def nn_seq(file_name, B, num):
    print('data processing...')
    data = load_data(file_name)
    data_list = data["DATE"]
    load = data[data.columns[1]]
    load = load.tolist()
    data = data.values.tolist()
    seq = []
    for i in range(0, len(data) - day_len - num, num):
        train_seq = []
        train_label = []
        for j in range(i, i + day_len):
            x = [load[j]]
            for c in range(2, 6):
                x.append(data[j][c])
            if j <= 7:
                # print(data[day_len + 1][6], j % 8)
                x.append(center[int(data[day_len + 1][6]) - 1][j % 8])
            elif 8 <= j <= 15:
                # print(data[day_len + 9][6], j % 8)
                x.append(center[int(data[day_len + 9][6]) - 1][j % 8])
            else:
                # print(data[day_len + 17][6], j % 8)
                x.append(center[int(data[day_len + 17][6]) - 1][j % 8])
            # print(x)
            train_seq.append(x)
        for j in range(i + day_len, i + day_len + num):
            train_label.append(load[j])
        train_seq = torch.FloatTensor(train_seq)
        train_label = torch.FloatTensor(train_label).view(-1)
        seq.append((train_seq, train_label))
    Dtr = seq
    train_len = int(len(Dtr) / B) * B
    Dtr = Dtr[:train_len]
    train = MyDataset(Dtr)
    Dtr = DataLoader(dataset=train, batch_size=B, shuffle=False, num_workers=0)
    return Dtr, data_list


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, out_channel):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 2
        self.batch_size = batch_size
        self.out_channel = out_channel
        self.lstm = nn.LSTM(self.out_channel, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        # self.attention = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=1,dropout=0.8)
        self.attention = nn.Linear(hidden_size, 3)
        self.cnn = nn.Conv1d(input_size, out_channel, kernel_size=3, padding=1)

    def att_dot_var(self, x):
        # b, s, input_size / b, s, hidden_size
        e = torch.bmm(x.permute(0, 2, 1), x)  # bis*bsi=bii
        attention = F.softmax(e, dim=-1)  # b i i
        out = torch.bmm(attention, x.permute(0, 2, 1))  # bii * bis ---> bis
        out = F.relu(out.permute(0, 2, 1))

        return out

    def att_dot_seq_len(self, x):
        # b, s, input_size / b, s, hidden_size
        x = self.attention(x)  # bsh--->bst
        e = torch.bmm(x, x.permute(0, 2, 1))  # bst*bts=bss
        attention = F.softmax(e, dim=-1)  # b s s
        out = torch.bmm(attention, x)  # bss * bst ---> bst
        out = F.relu(out)

        return out

    def forward(self, input_seq):
        # print(f"hidden_size:{self.hidden_size},num_layer:{self.num_layers}")
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # print(input_seq.size())
        seq_len = input_seq.shape[1]
        # input(batch_size, seq_len, input_size)
        # print(input_seq.shape)
        input_seq = input_seq.permute(0, 2, 1)
        # print(input_seq.shape)
        input_seq = self.cnn(input_seq)
        # print(input_seq.shape)
        input_seq = input_seq.permute(0, 2, 1)
        # print(input_seq.shape)
        input_seq = input_seq.view(self.batch_size, seq_len, self.out_channel)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        # print(output.shape)
        output = output.contiguous().view(self.batch_size, seq_len, self.num_directions, self.hidden_size)
        output = torch.mean(output, dim=2)
        output = self.att_dot_var(output)
        output = output[:, -1, :]
        pred = self.linear(output)
        # print('pred=', pred.shape)
        # pred = pred[:, -1, :]

        return pred

Dtr, train_date = nn_seq("../final_data/load_data_train_all_8_10_d.csv", 32, 24)
Dte, test_date = nn_seq("../final_data/load_data_test_all_8_10_d.csv", 32, 24)
# print(Dtr[:20])
loss_fn = nn.MSELoss(reduction='mean')
loss_fn = loss_fn.to(device)
min_rmse = 1


lstm_model = BiLSTM(6, 60, 1, 24, 32, 60)
lstm_model = lstm_model.to(device)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
echos = 140

lstm_model.train()
loss_hitory = []
act_build_energy = []
pre_build_energy = []
loss_hitory_np = []
flag = True
rmse_list = []
for echo in range(echos):
    # print(f'------------第{echo}次训练开始----------------')
    for data, target in Dtr:
        # print(data)
        # print(target)
        data = torch.tensor(data, requires_grad=True, dtype=torch.float32)
        target = torch.tensor(target, requires_grad=True, dtype=torch.float32)
        data = data.to(device)
        target = target.to(device)
        # print(data.shape)
        predictions = lstm_model(data)
        loss = loss_fn(predictions, target)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        loss_hitory.append(loss.data.cpu().numpy())
    print(f'------------第{echo}次训练开始----------------')
    logging.info(f'------------第{echo}次训练开始----------------')
    print(f'第{echo}次训练的损失为：{np.mean(loss_hitory)}')
    logging.info(f'第{echo}次训练的损失为：{np.mean(loss_hitory)}')
    loss_hitory_np.append(np.mean(loss_hitory))
lstm_model.eval()
total_test_loss = 0
total_accuracy = 0
with torch.no_grad():
    for data, target in Dte:
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        outputs = lstm_model(data)
        loss = loss_fn(outputs, target)
        total_test_loss += loss.item()
        prediction = outputs.detach().cpu().numpy()
        label = target.detach().cpu().numpy()
        accuracy = np.sum(np.abs(prediction - label) < 0.1) / label.size
        rmse = np.sqrt(np.mean((prediction - label) ** 2))
        rmse_list.append(rmse)
        if flag:
            act_build_energy = target.reshape(-1)
            pre_build_energy = outputs.reshape(-1)
            flag = False
        else:
            act_build_energy = torch.cat([act_build_energy, target.reshape(-1)], 0)
            pre_build_energy = torch.cat([pre_build_energy, outputs.reshape(-1)], 0)
print(type(act_build_energy), type(pre_build_energy))
mae, rmse, mape, r_2 = evaluation(act_build_energy.cpu().numpy(), pre_build_energy.cpu().numpy())
if min_rmse > rmse:
    torch.save(lstm_model.state_dict(), "model/PLVG_parameter_none_8_10_dh_step1.pkl")
    min_rmse = rmse
print(
    f"mae:{mae},rmse:{rmse * (MAX_LOAD - MIN_LOAD)},mape:{mape},r_2:{r_2},min_rmse:{min_rmse * (MAX_LOAD - MIN_LOAD)}")
logging.info(f"mae:{mae},rmse:{rmse},mape:{mape},r_2:{r_2},min_rmse:{min_rmse}")
