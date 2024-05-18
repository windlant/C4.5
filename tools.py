# tools.py 工具函数，用于用户输入或者调试
import pandas as pd
import numpy as np


# 按回车键继续函数
def pause():
    input("按回车键继续...\n")


# 连续数据检测函数，本次测试中默认数字都是连续数据
def is_discrete(data, a):
    # print("data:", data, "\na:", a)
    value = data.iloc[0].at[a]
    # print("\ndata[a]:", value)
    try:
        v = float(value)
    except:
        return 1
    return 0


# 整数输入函数，确保用户输入的是范围内的整数
def int_get(low, up, input_str):
    while True:
        k = input(input_str)
        try:
            k = int(k)
        except:
            print("输入错误！请重新输入\n")
            continue
        if k < low or k > up:
            print("输入错误！请重新输入\n")
            continue
        return k


# 布尔数输入函数，确保用户输入的是布尔数
def bool_get(input_str):
    while True:
        k = input(input_str)
        try:
            k = int(k)
        except:
            print("输入错误！请重新输入\n")
            continue
        if k != 0 and k != 1:
            print("输入错误！请重新输入\n")
            continue
        return k


# 浮点输入函数，确保用户输入的是范围内的数
def float_get(low, up, input_str):
    while True:
        f = input(input_str)
        try:
            f = float(f)
        except:
            print("输入错误！请重新输入\n")
            continue
        if f < low or f > up:
            print("输入错误！请重新输入\n")
            continue
        return f


# 路径输入函数，确保用户输入的是正确的数据集路径
def dir_get(input_str):
    while True:
        d = input(input_str)
        try:
            f = open(d)
        except:
            print("输入路径无效！请重新输入\n")
            continue
        f.close()
        return d


# 临时写入日志函数
def log(log_str):
    f = open("./log.txt", "w")
    f.write(str(log_str))
    f.close()


# 打印数组形状函数
def print_shape(name_str, name):
    print(name_str + ".shape：", name.shape, "\n")


# 写入csv函数
def csv_write(target_dir, data):
    data.to_csv(target_dir, mode='w', index=True, header=True)