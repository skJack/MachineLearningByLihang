import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from NaiveBayes import NaiveBayes
import time
class UnitTest():
    def Test1(self):
        start = time.clock()
        raw_data = pd.read_csv("../data/train.csv")
        data = raw_data.values
        x_data = data[:,1:]
        y_label = data[:,:1]
        #三分之一交叉验证
        train_data,test_data,train_label,test_label = train_test_split(x_data,y_label,test_size=0.33)
        end = time.clock()
        print("读取数据，处理数据结束，共花费{}秒".format(str(end-start)))
        c = NaiveBayes()
        start = time.clock()
        c.train(x_data,y_label)
        end = time.clock()
        print("训练朴素贝叶斯结束，共花费{}秒".format(str(end - start)))
        #验证
        start = time.clock()
        perdict_result = []
        for test in test_data:
            label = c.perdict(test)
            perdict_result.append(label)
        perdict_result = np.array(perdict_result)
        end = time.clock()
        print("预测结束，共花费{}秒".format(str(end - start)))
        score = accuracy_score(test_label,perdict_result)
        return score




if __name__ == '__main__':
    a = UnitTest()
    score = a.Test1()
    print("score is {0}".format(score))




