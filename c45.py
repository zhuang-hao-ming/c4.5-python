#-*- coding:utf-8 -*-
import operator
import math
def create_test_set():
    '''
    '''
    data_set = [
        [0, 0, 0, 0, 'N'], 
        [0, 0, 0, 1, 'N'], 
        [1, 0, 0, 0, 'Y'], 
        [2, 1, 0, 0, 'Y'], 
        [2, 2, 1, 0, 'Y'], 
        [2, 2, 1, 1, 'N'], 
        [1, 2, 1, 1, 'Y']
        ]
    labels = ['outlook', 'temperature', 'humidity', 'windy']
    return (data_set, labels)

def get_majority_class(class_list):
    '''
        输入： class列表
        输出： 出现频数最大的class
        例子：对于输入['Y', 'Y', 'N],返回'Y'
    '''
    class_count = {}
    for item in class_list:
        if item in class_count:
            class_count[item] += 1
        else:
            class_count[item] = 1
    class_items = class_count.items()
    class_items = sorted(class_items, key=operator.itemgetter(1), reverse=True)
    return class_items[0][0]

def calculate_shannon_ent(data_set):
    '''
    输入： 样本集
    输出： 样本集的熵
    例子： 对于输入[[0],[0],[1],[1]],返回1
    '''
    num_of_samples = len(data_set)
    class_count = {}
    for sample in data_set:
        class_label = sample[-1]
        if class_label in class_count:
            class_count[class_label] += 1
        else:
            class_count[class_label] = 1
    ent = 0.0
    for key in class_count:
        prob = float(class_count[key]) / num_of_samples
        ent -= prob * math.log(prob, 2)
    return ent

def split_data_set(data_set, feature_idx, val):
    '''
    输入： 样本集， 特征索引， 特征值
    输出： 样本子集
    描述： 选择出样本集中特征索引所指的特征的值是给定特征值的样本，删除样本中的给定特征，返回样本子集
    例子： [[2, 'a'], [2, 'a'], [1, 'b'], [1, 'b']] ， 1， 2， [['a'], ['a']]
    '''
    sub_data_set = []
    for sample in data_set:
        if sample[feature_idx] == val:
            reduce_sample = sample[:feature_idx] + sample[feature_idx+1:]
            sub_data_set.append(reduce_sample)
    return sub_data_set


def choose_best_feature_for_split(data_set):
    '''
    输入： 样本集
    输出： 最好的划分特征索引
    '''
    num_features = len(data_set[0]) - 1
    base_entropy = calculate_shannon_ent(data_set)
    best_info_gain_ration = 0.0
    best_feature_idx = -1
    for i in range(num_features):
        feature_list = [ sample[i] for sample in data_set ]
        unique_values = set(feature_list)
        new_entropy = 0.0
        split_info = 0.0
        for val in unique_values:
            sub_set = split_data_set(data_set, i, val)
            prob = float(len(sub_set)) / len(data_set)
            new_entropy += prob * calculate_shannon_ent(sub_set)
            split_info += -prob * math.log(prob, 2)
        if split_info == 0:
            continue
        info_gain = base_entropy - new_entropy
        info_gain_ration = info_gain / split_info
        if info_gain_ration > best_info_gain_ration:
            best_info_gain_ration = info_gain_ration
            best_feature_idx = i
    return best_feature_idx








def create_tree(data_set, labels):
    '''

    '''
    class_list = [ item[-1] for item in data_set ]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return get_majority_class(class_list)
    best_feature_idx = choose_best_feature_for_split(data_set)
    best_feature_label = labels[best_feature_idx]
    del(labels[best_feature_idx])
    my_tree = {best_feature_label:{}}
    feature_vals = [ sample[best_feature_idx] for sample in data_set ]
    unique_vals = set(feature_vals)
    for val in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feature_label][val] = create_tree(split_data_set(data_set, best_feature_idx, val), sub_labels)
    return my_tree


def main():
    data_set, labels = create_test_set()
    decision_tree = create_tree(data_set, labels)
    print(decision_tree)

if __name__ == '__main__':
    main()
