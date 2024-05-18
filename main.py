import numpy as np
from data_process import *
from C4_5 import *
from draw import *
import sys
default_data_dir = "./dataset/iris.csv"
default_res_dir = "./result.csv"
note = "分类算法作业 2331915 龚乙骁\n" \
    + "默认使用的是鸢尾花数据集iris.csv，该数据集有n=150个样本，每个样本有fn=4个属性\n" \
    + "如果需要使用新的数据集或者测试集请保证数据格式与iris.csv一致\n" \
    + "在./dataset中还有其他如./dataset/watermelon.csv的数据集\n" \
    + "决策树可视化基于graphviz实现，如果没有安装则无法可视化\n" \
    + "决策树为C4.5决策树\n" \
    + "可以转化为float的特征都将视为连续数据，其余则视为离散数据，连续数据采用最优二分\n"\
    + "用户可以按比例（每种label单独划分出一部分）抽取数据集中一部分作为训练集，其余部分作为测试集\n"


def main():
    print(note)
    sys.stdout.flush()
    data_dir = default_data_dir
    while True:
        # 数据集选择
        input_str = "当前数据集为：" + data_dir
        print(input_str)
        input_str = "是否更换数据集？输入1是，输入0否：\n"
        b = bool_get(input_str)
        if b:
            input_str = "请输入数据集路径：\n"
            data_dir = dir_get(input_str)

        data = data_get(data_dir)
        n = data.shape[0]
        if n <= 1:
            print("数据集样本数过小，请更换数据集！\n")
            continue

        # 请选择数据集作为训练集的比例
        rate = 0.5
        input_str = "请输入训练集比例 0.2<=rate<=0.8\n"
        rate = float_get(0.2, 0.8, input_str)
        train_data, test_data = train_data_get(data, rate)

        # 构建决策树
        print("开始构建决策树...\n")
        # 统计每个特征有哪些种类，以及标签种类
        feature_info = feature_get(data)  # 全局特征信息
        labels = label_get(data)
        decision_tree = tree_create(train_data, feature_info)
        print("决策树构建完成，如果要查看决策树结构，请确保电脑已安装graphviz")
        input_str = "是否查看决策树的结构？输入1是，输入0否\n"
        b = bool_get(input_str)
        if b == 1:
            plot_model(decision_tree, "decision_tree.gv")
        input_str = "是否更换测试集（默认为训练集之外的数据）？输入1是，输入0否\n"
        b = bool_get(input_str)
        if b == 1:
            input_str = "请输入测试集路径：\n"
            test_dir = dir_get(input_str)
            test_data = data_get(test_dir)
        print("开始进行测试...\n")
        res_data = predict_all(decision_tree, test_data)
        csv_write(default_res_dir, res_data)
        print("测试完毕，测试结果已写入" + default_res_dir + "\n")
        pause()
        accuracy = accuracy_get(test_data, res_data)
        print("预测结果准确率：", accuracy)
        res = precision_recall_get(test_data, res_data)
        for label in labels:
            print("种类" + label + " 精确率：", res[label][0], "，召回率：", res[label][1])
        pause()
        input_str = "是否退出程序？输入1退出，输入0继续：\n"
        b = bool_get(input_str)
        if b == 1:
            exit(1)


if __name__ == '__main__':
    main()
    # 读取数据
    # data_dir = default_data_dir
    # data = data_get(data_dir)
    # train_data, test_data = train_data_get(data, 0.5)
    # # 统计每个特征有哪些种类，以及标签种类
    # feature_info = feature_get(data)  # 全局特征信息
    # labels = label_get(data)
    #
    # # 创建决策树
    # decision_tree = tree_create(train_data, feature_info)
    # plot_model(decision_tree, "decision_tree.gv")
    #
    # res_data = predict_all(decision_tree, test_data)
    # csv_write(default_res_dir, res_data)
    # log(str(res_data))
    # accuracy = accuracy_get(test_data, res_data)
    # print("预测结果准确率：", accuracy)
    # res = precision_recall_get(test_data, res_data)
    # for label in labels:
    #     print("种类" + label + " 精确率：", res[label][0], "，召回率：", res[label][1])

    # print(decision_Tree)
    # 测试数据
    # test_data_1 = pd.DataFrame([['青绿', '蜷缩', '浊响', '稍糊', '凹陷', '硬滑']],
    #                            columns=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感'])
    # test_data_2 = pd.DataFrame([['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑']],
    #                            columns=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感'])
    # test_data_1 = pd.DataFrame([['青绿', '蜷缩', '浊响', '稍糊', '凹陷', '硬滑', 0.9, 0.6]],
    #                            columns=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率'])
    # test_data_2 = pd.DataFrame([['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.3, 0.2]],
    #                            columns=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率'])
    # test_data_1 = {'色泽': '青绿', '根蒂': '蜷缩', '敲声': '浊响', '纹理': '稍糊', '脐部': '凹陷', '触感': '硬滑'}
    # test_data_2 = {'色泽': '乌黑', '根蒂': '稍蜷', '敲声': '浊响', '纹理': '清晰', '脐部': '凹陷', '触感': '硬滑'}
    # result = predict_one(decision_tree, test_data_2)
    # print("分类结果为1", result)
    # print("\n")
    # result = predict_one(decision_tree, test_data_1)
    # print("分类结果为2", result)
