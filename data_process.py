import pandas as pd


#  数据格式为CSV，直接获取表格数据
def data_get(data_dir):
    data = pd.read_csv(data_dir)
    return data


# 获取每一个特征的所有种类，返回字典，特征名与特征种类列表一一对应
def feature_get(data):
    feature_info = dict([(ds, list(pd.unique(data[ds]))) for ds in data.iloc[:, :-1].columns])
    return feature_info


# 获取标签种类
def label_get(data):
    labels = data.iloc[:, -1].value_counts().keys()
    return labels


# 按比例获取训练、测试数据
def train_data_get(data, rate):
    labels = label_get(data)
    res_train = pd.DataFrame(data=None, columns=data.columns)
    res_test = pd.DataFrame(data=None, columns=data.columns)
    for label in labels:
        tmp_data = data[data.iloc[:, -1] == label]
        n = len(tmp_data)
        n_rated = int(n * rate)
        tmp_data2 = tmp_data.iloc[0:n_rated, :]
        tmp_data3 = tmp_data.iloc[n_rated:n, :]
        res_train = pd.concat([res_train, tmp_data2], axis=0, ignore_index=True)
        res_test = pd.concat([res_test, tmp_data3], axis=0, ignore_index=True)
    return res_train, res_test

