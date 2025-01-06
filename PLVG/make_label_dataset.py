# -*- coding:utf-8 -*-
"""
作者:朱昊喆
日期:2024年12月18日
"""
# -*- coding:utf-8 -*-
"""
作者:朱昊喆
日期:2023年08月10日
"""
import csv
from chinese_calendar import is_workday, is_holiday, get_holiday_detail, is_in_lieu
from datetime import datetime, timedelta


def get_date(data):
    pos = data.find("-")
    year = data[:pos]
    data = data[pos + 1:]
    pos = data.find("-")
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


file_content = ''
date_temp = {}
word_dict = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm',
             14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y',
             26: 'z'}
hour_dict = {0: 'h', 8: 'i', 16: 'j'}

dataset = 'test'
# type_name = 'load'
with open('../cluster/'+dataset+'_temp_full_dim2_8_dtw.csv', "r", encoding='gb2312') as f1:
    reader1 = csv.reader(f1)
    for row in reader1:
        if row[9] == 'load_label':
            continue
        date_temp[row[0]] = row[9]
with open('../cluster/'+dataset+'_load_full_dim2_8_dtw.csv', "r", encoding='gb2312') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[9] == 'load_label':
            continue
        print(row[9])
        if row[0] in date_temp.keys():
            print(date_temp[row[0]])
        else:
            print('error')
            exit(-1)
        year, month, day, hour, minute = get_date(row[0])
        dt = datetime.strptime(str(year) + '-' + str(month) + '-' + str(day), "%Y-%m-%d")
        file_content += str(int(date_temp[row[0]]) + 6)
        file_content += " "
        file_content += word_dict[
            int(datetime.date(datetime(year=int(year), month=int(month), day=int(day))).weekday() + 1)]
        file_content += " "
        file_content += hour_dict[int(hour)]
        file_content += " "
        file_content += str(int(row[9])+1)
        file_content += "."

#
file = open('label_'+dataset+'_full_dh.txt', 'w')
file.write(file_content)
file.close()
