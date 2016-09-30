#-*- coding:utf-8 -*-
import pytest

import c45

# 测试“测试数据集”是否正确创建
def test_create_data_set():
    (data_set, labels) = c45.create_test_set()
    assert len(data_set[0]) == 5
    assert len(labels) == 4


def test_get_majority_class():
    test_set = ['Y', 'Y', 'N']
    expect_val = 'Y'
    actual_val = c45.get_majority_class(test_set)
    assert expect_val == actual_val

# 测试熵计算是否正确
def test_calculate_shannon_ent():
    test_set = [
        [0],
        [0],
        [1],
        [1]
        ]

    expect_val = 1
    actual_val = c45.calculate_shannon_ent(test_set)
    assert expect_val == actual_val

# 测试取样本子集正确
def test_split_data_set():
    test_set = [
        [2, 'a'],
        [2, 'a'],
        [1, 'b'], 
        [1, 'b']]
    test_idx = 0
    test_val = 2
    except_val = [['a'], ['a']]
    actual_val = c45.split_data_set(test_set, test_idx, test_val)
    assert except_val == actual_val

# 测试选择最好的划分特征
def test_choose_best_feature_for_split():
    test_set = [
        [0, 0, 0, 0, 'N'], 
        [0, 0, 0, 1, 'N'], 
        [1, 0, 0, 0, 'Y'], 
        [2, 1, 0, 0, 'Y'], 
        [2, 2, 1, 0, 'Y'], 
        [2, 2, 1, 1, 'N'], 
        [1, 2, 1, 1, 'Y']]
    expect_val = 0    
    actual_val = c45.choose_best_feature_for_split(test_set)
    assert expect_val == actual_val

# 测试选择最好的划分特征
def test_choose_best_feature_for_split1():
    test_set = [                
        [ 1, 0, 0, 'Y'], 
        [ 2, 1, 0, 'Y'], 
        [ 2, 1, 1, 'N']]
    expect_val = 2   
    actual_val = c45.choose_best_feature_for_split(test_set)
    assert expect_val == actual_val

# 测试构建决策树
def test_create_tree():
    test_set, labels = c45.create_test_set()
    my_tree = c45.create_tree(test_set, labels)
    assert my_tree['outlook'][2]['windy'][0] == 'Y'