import pandas as pd
import numpy as np
from tools import *


# 计算信息熵
def ie_get(data):
    data_label = data.iloc[:, -1]
    label_class = data_label.value_counts()
    ent = 0
    for k in label_class.keys():
        pk = label_class[k] / len(data_label)
        ent += -pk * np.log2(pk)
    return ent


# 计算连续数据的最优划分点, 以及信息增益
def best_cut_point_get(data, a):
    value = data[a]
    value_array = value.values
    value_array = np.sort(value_array)
    v1 = value_array[:-1]
    v2 = value_array[1:]
    points = (v1 + v2) / 2
    ent_pre = ie_get(data)

    #  计算最优划分点
    best_point = 0
    best_gain = 0
    max_ig = 0
    n = data.shape[0]
    for p in points:
        data_left = data.loc[data[a] <= p]
        data_right = data.loc[data[a] > p]
        w1 = data_left.shape[0]/n
        w2 = data_right.shape[0]/n
        ent_aft = w1 * ie_get(data_left) + w2 * ie_get(data_right)
        gain = ent_pre - ent_aft
        if gain > max_ig:
            max_ig = gain
            best_point = p
            best_gain = gain
    return best_gain, best_point


# 计算给定数据特征a的信息增益，如果是连续数据，额外返回最优划分点
def ig_get(data, a):
    gain = 0
    point = 0
    dt = is_discrete(data, a)
    if dt == 0:
        gain, point = best_cut_point_get(data, a)
        return gain, point

    ent_pre = ie_get(data)
    feature_class = data[a].value_counts()
    ent_aft = 0
    for v in feature_class.keys():
        weight = feature_class[v] / data.shape[0]
        ent_v = ie_get(data.loc[data[a] == v])
        ent_aft += weight * ent_v
    gain = ent_pre - ent_aft
    return gain, point


#  计算信息增益率
def igr_get(data, a):
    IV = 0
    point = 0
    n = data.shape[0]
    feature_class = data[a].value_counts()
    for v in feature_class.keys():
        weight = feature_class[v] / n
        IV += -weight * np.log2(weight)
    gain, point = ig_get(data, a)
    if IV == 0:
        gain_ration = 0
    else:
        gain_ration = gain / IV
    return gain_ration, point


# 获取标签最多的那一类，用于特征都一样标签不一样的情况
def most_label_get(data):
    data_label = data.iloc[:, -1]
    label_sort = data_label.value_counts(sort=True)
    most_label = label_sort.keys()[0]
    return most_label


# 挑选最优特征，即在信息增益大于平均水平的特征中选取增益率最高的特征
def best_feature_get(data):
    features = data.columns[:-1]
    res = {}
    point = 0
    for a in features:
        gain, point = ig_get(data, a)
        gain_ration, point = igr_get(data, a)
        res[a] = (gain, gain_ration, point)
    res = sorted(res.items(), key=lambda x: x[1][0], reverse=True)  # 按信息增益排名
    res_avg = sum([x[1][0] for x in res]) / len(res)  # 信息增益平均水平
    good_res = [x for x in res if x[1][0] >= res_avg]  # 选取信息增益高于平均水平的特征
    result = sorted(good_res, key=lambda x: x[1][1], reverse=True)  # 将信息增益高的特征按照增益率进行排名
    best_a = result[0][0]
    point = result[0][1][2]
    return best_a, point  # 返回高信息增益中增益率最大的特征


# 按照属性对数据进行划分，返回每个属性划分后对应的新数据，并且要去掉本次划分用过的属性
def new_data_get(data, a, point):
    dt = is_discrete(data, a)
    if dt == 0:
        data_left = data[data[a] <= point]
        data_right = data[data[a] > point]
        new_data = [(0, data_left), (1, data_right)]
        new_data = [(n[0], n[1].drop([a], axis=1)) for n in new_data]
        return new_data
    attr = pd.unique(data[a])
    new_data = [(nd, data[data[a] == nd]) for nd in attr]
    new_data = [(n[0], n[1].drop([a], axis=1)) for n in new_data]
    return new_data


# 创建决策树
def tree_create(data, feature_info):
    data_label = data.iloc[:, -1]
    if len(data_label.value_counts()) == 1:  # 只有一类
        return data_label.values[0]
    # 所有数据的特征值一样，或者特征已划分完毕，选样本最多的类作为分类结果，注意all(空集) = True
    if all(len(data[i].value_counts()) == 1 for i in data.iloc[:, :-1].columns):
        return most_label_get(data)
    best_feature, point = best_feature_get(data)  # 根据信息增益得到的最优划分特征
    tree = {best_feature: {}, "point": point}  # 用字典形式存储决策树
    dt = is_discrete(data, best_feature)
    # 连续值的情况
    if dt == 0:
        new_data = new_data_get(data, best_feature, point)
        tree[best_feature][0] = tree_create(new_data[0][1], feature_info)
        tree[best_feature][1] = tree_create(new_data[1][1], feature_info)
        return tree

    # 离散的情况
    exist_vals = pd.unique(data[best_feature])  # 当前数据下最佳特征的取值
    if len(exist_vals) != len(feature_info[best_feature]):  # 如果特征的取值相比于原来的少了
        no_exist_attr = set(feature_info[best_feature]) - set(exist_vals)  # 少的那些特征
        for no_feat in no_exist_attr:
            tree[best_feature][no_feat] = most_label_get(data)  # 缺失的特征分类为当前类别最多的
    for item in new_data_get(data, best_feature, point):  # 根据特征值的不同递归创建决策树
        tree[best_feature][item[0]] = tree_create(item[1], feature_info)
    return tree


# 单项数据预测
def predict_one(tree, test_data):
    current_feature = list(tree.keys())[0]
    feature_dict = tree[current_feature]
    dt = is_discrete(test_data, current_feature)
    data_value = test_data.iloc[0].at[current_feature]
    point = tree["point"]
    if dt == 0:
        data_feature = 0 if data_value <= point else 1
        next_tree = feature_dict[data_feature]
    else:
        next_tree = feature_dict[data_value]
    if isinstance(next_tree, dict):  # 判断分支还是不是字典
        class_label = predict_one(next_tree, test_data)
    else:
        class_label = next_tree
    return class_label


# 多项数据预测，返回表格
def predict_all(tree, test_data):
    res_data = test_data.copy()
    for i in range(len(test_data)):
        res_data.iloc[i, -1] = predict_one(tree, test_data.iloc[[i]])
    return res_data


# 获取某次预测的准确率
def accuracy_get(test_data, res_data):
    n = len(test_data)
    if n == 0:
        return 0
    correct = res_data[res_data.iloc[:, -1] == test_data.iloc[:, -1]]
    accuracy = round(len(correct)/n, 2)
    return accuracy


# 获取某次预测的所有标签的精确率和召回率
def precision_recall_get(test_data, res_data):
    labels = test_data.iloc[:, -1].value_counts().keys()
    res = {}
    for label in labels:
        prediction = res_data[res_data.iloc[:, -1] == label]
        original = test_data[test_data.iloc[:, -1] == label]
        p_index = set(prediction.index)
        o_index = set(original.index)
        pos_true_num = len(p_index & o_index)
        if len(prediction) == 0:
            precision = round(0, 2)
        else:
            precision = round(pos_true_num/len(prediction), 2)
        recall = round(pos_true_num/len(original), 2)
        res[label] = (precision, recall)
    return res

