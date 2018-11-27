import numpy as np
import pandas as pd
import cv2

class Tree(object):
    def __init__(self,node_type,Class = None, feature = None):
        '''
        :param node_type: leaf and internel
        :param Class: leaf node's label
        :param feature: internel node's feature
        '''
        self.node_type = node_type
        self.dict = {}
        self.Class = Class
        self.feature = feature

    def add_tree(self,val,tree):
        self.dict[val] = tree

    def predict(self,features):
        #dfs深搜
        if self.node_type == 'leaf':
            return self.Class
        tree = self.dict[features[self.feature]]
        return tree.predict(features)

def cal_ent(y):
    #计算集合的信息熵
    if y.shape[0] == 0:
        return 0
    values = list(set(y))
    result = 0
    value_count = pd.value_counts(y)
    for value in values:
        p = value_count[value] / y.shape[0]
        result -= p*np.log2(p)
    return result
def cal_condition_ent(A,D):
    #计算条件熵,A为当前分割的feature，D为当前数据集
    values = list(set(A))
    result = 0
    for value in values:
        sub_D = D[A == value]#注意numpy数组==号的作用
        temp_ent = cal_ent(sub_D)
        p = sub_D.shape[0] / D.shape[0]
        result += p*temp_ent
    return result

def gain(A,D):
    return cal_ent(D) - cal_condition_ent(A,D)
class DecisionTree():
    def __init__(self,tree = None):
        self.tree = tree
    def _build_tree(self,X,Y,threshold,fea_index):
        # node type
        LEAF = 'leaf'
        INTERNAL = 'internal'
        #fea_index = np.arange(X.shape[1])#生成特征数量的索引
        labels = Y
        print(labels)
        #step1:D中所有实例属于一类，返回该节点的类,树为单节点树
        label_set = set(Y)
        if len(label_set) == 1:
            return Tree(LEAF,Class=labels[0])

        #step2如果特征集为0，则返回最大特征类
        max_label = pd.value_counts(labels).index[0]#取label中数量最大的
        if len(fea_index) == 0:
            return Tree(LEAF,Class=max_label)

        #step3 :选择信息增益最大的特征
        max_feat = 0
        max_gda = 0
        D = labels
        for i in fea_index:
            A = X[:,i]#取第i列，这一列代表这个特征的所有取值
            gda = gain(A,D)#求该特征的信息增益
            if max_gda < gda:
                max_gda,max_feat = gda , i

        #step4:如果信息增益小于阈值，就返回单节点树最大的类标记
        if max_gda < threshold:
            return max_label

        #step5:根据特征A的取值分割数据集并构建子节点
        sub_features = list(filter(lambda x: x != max_feat, fea_index))#filter为过滤，过滤掉不符合lamda表达式的feature
        tree = Tree(INTERNAL, feature=max_feat)#这棵树是internel节点，需要递归的建立子节点
        values = set(X[:,max_feat])#values是最大熵的全部特征取值集合
        for value in values:
            sub_D = D[X[:,max_feat] == value]#取分割后的D
            sub_x = X[X[:,max_feat] == value,:]#取X
            #sub_x = np.delete(sub_x,max_feat,1)
            sub_tree = self._build_tree(sub_x,sub_D,threshold,sub_features)
            tree.add_tree(value,sub_tree)
        return tree

    def train(self,x_data,y_label,threshold):
        fea = np.arange(x_data.shape[1])
        self.tree = self._build_tree(x_data,y_label,threshold,fea)

    def perdict(self,features):
        result = []
        for feature in features:
            temp = self.tree.perdict(feature)
            result.append(temp)
        return result


















