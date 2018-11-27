import numpy as np
import pandas as pd
import cv2
def cal_ent(y):
    #计算集合的信息熵
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
        sub_D = D[A == values]#注意numpy数组==号的作用
        temp_ent = cal_ent(sub_D)
        p = sub_D.shape[0] / D.shape[0]
        result += p*temp_ent
    return result

def gain(A,D):
    cal_ent(D) - cal_condition_ent(A,D)
class DecisionTree():
    def __init__(self,tree = None):
        self.tree = tree
    def _build_tree(self,X,Y,threshold):
        fea_index = np.arange(X.shape[1])#生成特征数量的索引
        labels = Y

        #step1:D中所有实例属于一类，返回该节点的类,树为单节点树
        if len(set(label)) == 1:
            return labels[0]

        #step2
        max_label = pd.value_counts(labels).index[0]#取label中数量最大的
        if len(fea_index) == 0:
            return max_label

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
        tree = {}
        sub_tree = {}
        #step5:根据特征A的取值分割数据集并构建子节点
        values = set(X[:,max_feat])#values是最大熵的全部特征取值集合
        for value in values:
            sub_D = D[X[:,max_feat] == value]#取分割后的D
            sub_x = X[X[:,max_feat] == value,:]#取X
            sub_x = np.delete(sub_x,max_feat,1)
            #递归的构建子树
            sub_tree[str(value)] = self._build_tree(sub_x,sub_D,threshold)
        #返回这个节点
        tree["特征"+str(max_feat)] = sub_tree
        return tree
    def predict(self,x,tree):














