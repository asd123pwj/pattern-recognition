import numpy as np
import math


#   对letter数据集分类
def get_list_letter(char, features, path):
    with open(path, 'r') as letter_train:
        content_letter_train = letter_train.readlines()
    for line in content_letter_train:
        temp = line.split(',')
        temp[-1] = list(temp[-1])[0]
        char.append(temp[0])
        features.append((temp[1::]))


#   对optdigits数据集分类
def get_list_optdigits(dig, features, path):
    with open(path, 'r') as dig_train:
        content_dig_train = dig_train.readlines()
    for line in content_dig_train:
        temp = line.split(',')
        temp[-1] = list(temp[-1])[0]
        dig.append(temp[-1])
        features.append((temp[0:len(temp)-1:]))


#   对sat数据集分类
def get_list_sat(sat, features, path):
    with open(path, 'r') as sat_train:
        content_sat_train = sat_train.readlines()
    for line in content_sat_train:
        temp = line.split(' ')
        temp[-1] = list(temp[-1])[0]
        sat.append(temp[-1])
        features.append((temp[0:len(temp)-1:]))


#   对vowel数据集分类
def get_list_vowel(vowel, features, path):
    with open(path, 'r') as vowel_train:
        content_vowel_train = vowel_train.readlines()
    for line in content_vowel_train:
        temp = line.split()
        temp[-1] = list(temp[-1])[0]
        vowel.append(temp[-1])
        features.append((temp[3:len(temp)-1:]))


#   列表字符串元素转浮点型
def convert_float(str_list):
    for row in range(0, len(str_list)):
        for col in range(0, len(str_list[row])):
            str_list[row][col] = float(str_list[row][col])


#   列表字符串元素转整型
def convert_int(str_list):
    for row in range(0, len(str_list)):
        for col in range(0, len(str_list[row])):
            str_list[row][col] = int(str_list[row][col])


#   求期望
def get_expectation(features):
    expectation = [0] * len(features)
    for row in range(0, len(features)):
        for col in range(0, len(features[0])):
            expectation[row] += features[row][col]
    for pos in range(0, len(expectation)):
        expectation[pos] = expectation[pos] / len(features[0])
    return expectation


#   train_features_sort_by_class_with_same_feature:
#       将训练集的特征列表，按类保存成字典，即，键是类，值是特征列表的列表
#   sort_features{}: { class, features_list[] }, 保存每个类对应的特征列表的列表，字典
#   class: 类
#   features_list[]: [ feature_list[] ], 保存所有特征列表，每个元素均是列表
#   注意，这里字典里保存的不再是特征列表的列表，而是特征的列表，即原本按类来排序的特征列表的列表，被转置了。
#       这样方便后续的求协方差。
def train_features_sort_by_class_with_same_feature(train_features, train_class):
    sort_features = {}
    for pos in range(0, len(train_class)):
        if train_class[pos] not in sort_features:
            sort_features[train_class[pos]] = []
        sort_features[train_class[pos]].append(train_features[pos])
    for clas in sort_features:
        sort_features[clas] = np.array(sort_features[clas]).T
    return sort_features


#   get_param: 根据训练集特征与类，得到对应QDF的参数列表
#   param_list[] : [ class[], cov[][][], expectation[][] ], 保存class对应的协方差矩阵cov与期望expectation
#   class: 类别
#   cov[][]: 由训练集特征得到的协方差矩阵
#   expectation[]: 特征期望
def get_param(sort_features, train_class):
    param_list = [[], [], []]
    for clas in sort_features:
        param_list[0].append(clas)
        param_list[1].append(np.cov(sort_features[clas]))
        param_list[2].append(np.array(get_expectation(sort_features[clas])))
    param_list.append(get_char_probability(train_class))
    return param_list


#   得到训练集中各类出现的可能性，返回可能性字典，键是类，值是可能性
def get_char_probability(train_class):
    char_probability = {}
    for chr in train_class:
        if chr not in char_probability:
            char_probability[chr] = 0
        char_probability[chr] += 1
    for chr in char_probability:
        char_probability[chr] = char_probability[chr] / len(train_class)
    return char_probability


#   以method方法预测test_features可能的类别，返回其类别
def predict(test_features, param_list, method):
    max_probability = 0
    max_char = ''
    for pos in range(0, len(param_list[0])):
        clas = param_list[0][pos]
        cov = param_list[1][pos]
        expectation = param_list[2][pos]
        cov_det = np.linalg.det(cov)

        ele1 = test_features - expectation
        if cov_det == 0:
            # 协方差矩阵不可逆，使用伪逆
            ele2 = np.linalg.pinv(cov)
            cov_reversible = False
        else:
            # 协方差矩阵可逆
            ele2 = np.linalg.inv(cov)
            cov_reversible = True
        ele3 = ele1.T
        if method == 'QDF':
            if cov_reversible:
                ele4 = - math.log(cov_det)
            else:
                ele4 = math.log(param_list[3][clas]) * 2
        elif method == 'LDF':
            ele4 = math.log(param_list[3][clas]) * 2

        probability = np.matmul(ele1, ele2)
        probability = np.matmul(probability, ele3)
        probability = - probability + ele4
        if probability > max_probability or max_char == '':
            max_probability = probability
            max_char = clas
    return max_char


#   使用method方法对训练集进行训练，并测试测试集，返回每个测试特征可能的类别
def train_and_predict(train_features, train_class, test_features, method):
    predict_class = []
    sort_features = train_features_sort_by_class_with_same_feature(train_features, train_class)
    param = get_param(sort_features, train_class)
    for pos in range(0, len(test_features)):
        predict_class.append(predict(test_features[pos], param, method))
    return predict_class


#   分析方法精确度，返回精确度
def analysis_accuracy(judge_result, test_char):
    sum = 0
    right_num = 0
    for pos in range(0, len(judge_result)):
        sum += 1
        if judge_result[pos] == test_char[pos]:
            right_num += 1
    return right_num / sum


#   letter数据集初始化
letter_train_path = './dataset/letter.train'
letter_train_class = []
letter_train_features = []
letter_test_path = './dataset/letter.test'
letter_test_class = []
letter_test_features = []
get_list_letter(letter_train_class, letter_train_features, letter_train_path)
get_list_letter(letter_test_class, letter_test_features, letter_test_path)
convert_int(letter_train_features)
convert_int(letter_test_features)
#   letter数据集训练及测试
letter_LDF_predict = train_and_predict(letter_train_features, letter_train_class, letter_test_features, 'LDF')
letter_LDF_accuracy = analysis_accuracy(letter_LDF_predict, letter_test_class)
print('使用LDF对letter的', len(letter_train_features), '份数据学习后，对',
      len(letter_test_features), '份测试数据分类的准确率为：', letter_LDF_accuracy)
letter_QDF_predict = train_and_predict(letter_train_features, letter_train_class, letter_test_features, 'QDF')
letter_QDF_accuracy = analysis_accuracy(letter_QDF_predict, letter_test_class)
print('使用QDF对letter的', len(letter_train_features), '份数据学习后，对',
      len(letter_test_features), '份测试数据分类的准确率为：', letter_QDF_accuracy)

#   optdigits数据集初始化
optdigits_train_path = './dataset/optdigits.train'
optdigits_train_class = []
optdigits_train_features = []
optdigits_test_path = './dataset/optdigits.test'
optdigits_test_class = []
optdigits_test_features = []
get_list_optdigits(optdigits_train_class, optdigits_train_features, optdigits_train_path)
convert_int(optdigits_train_features)
get_list_optdigits(optdigits_test_class, optdigits_test_features, optdigits_test_path)
convert_int(optdigits_test_features)
#   optdigits数据集训练及测试
optdigits_LDF_predict = train_and_predict(optdigits_train_features, optdigits_train_class, optdigits_test_features, 'LDF')
optdigits_LDF_accuracy = analysis_accuracy(optdigits_LDF_predict, optdigits_test_class)
print('使用LDF对optdigits的', len(optdigits_train_features), '份数据学习后，对',
      len(optdigits_test_features), '份测试数据分类的准确率为：', optdigits_LDF_accuracy)
optdigits_QDF_predict = train_and_predict(optdigits_train_features, optdigits_train_class, optdigits_test_features, 'QDF')
optdigits_QDF_accuracy = analysis_accuracy(optdigits_QDF_predict, optdigits_test_class)
print('使用QDF对optdigits的', len(optdigits_train_features), '份数据学习后，对',
      len(optdigits_test_features), '份测试数据分类的准确率为：', optdigits_QDF_accuracy)

#   sat数据集初始化
sat_train_path = './dataset/sat.train'
sat_train_class = []
sat_train_features = []
sat_test_path = './dataset/sat.test'
sat_test_class = []
sat_test_features = []
get_list_sat(sat_train_class, sat_train_features, sat_train_path)
convert_int(sat_train_features)
get_list_sat(sat_test_class, sat_test_features, sat_test_path)
convert_int(sat_test_features)
#   sat数据集训练及测试
sat_LDF_predict = train_and_predict(sat_train_features, sat_train_class, sat_test_features, 'LDF')
sat_LDF_accuracy = analysis_accuracy(sat_LDF_predict, sat_test_class)
print('使用LDF对sat的', len(sat_train_features), '份数据学习后，对',
      len(sat_test_features), '份测试数据分类的准确率为：', sat_LDF_accuracy)
sat_QDF_predict = train_and_predict(sat_train_features, sat_train_class, sat_test_features, 'QDF')
sat_QDF_accuracy = analysis_accuracy(sat_QDF_predict, sat_test_class)
print('使用QDF对sat的', len(sat_train_features), '份数据学习后，对',
      len(sat_test_features), '份测试数据分类的准确率为：', sat_QDF_accuracy)

#   vowel数据集初始化
vowel_train_path = './dataset/vowel.train'
vowel_train_class = []
vowel_train_features = []
vowel_test_path = './dataset/vowel.test'
vowel_test_class = []
vowel_test_features = []
get_list_vowel(vowel_train_class, vowel_train_features, vowel_train_path)
convert_float(vowel_train_features)
get_list_vowel(vowel_test_class, vowel_test_features, vowel_test_path)
convert_float(vowel_test_features)
#   vowel数据集训练及测试
vowel_LDF_predict = train_and_predict(vowel_train_features, vowel_train_class, vowel_test_features, 'LDF')
vowel_LDF_accuracy = analysis_accuracy(vowel_LDF_predict, vowel_test_class)
print('使用LDF对vowel的', len(vowel_train_features), '份数据学习后，对',
      len(vowel_test_features), '份测试数据分类的准确率为：', vowel_LDF_accuracy)
vowel_QDF_predict = train_and_predict(vowel_train_features, vowel_train_class, vowel_test_features, 'QDF')
vowel_QDF_accuracy = analysis_accuracy(vowel_QDF_predict, vowel_test_class)
print('使用QDF对vowel的', len(vowel_train_features), '份数据学习后，对',
      len(vowel_test_features), '份测试数据分类的准确率为：', vowel_QDF_accuracy)
