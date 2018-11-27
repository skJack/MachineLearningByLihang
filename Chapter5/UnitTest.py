import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTree
import cv2
import time
def binary(img):
    # 二值化
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY, cv_img)  # 将大于50的向量变为1，其余为0
    return cv_img

class UnitTest():
    def Test1(self):
        start = time.clock()
        raw_data = pd.read_csv("../data/train.csv").values
        x_data = raw_data[:,1:]
        y_label = raw_data[::,0]
        feature = []
        for img in x_data:
            feature.append(binary(img=img))
        feature = np.array(feature)
        end = time.clock()
        print("读取数据，处理数据结束，共花费{}秒".format(str(end - start)))
        #cross validation
        train_data, test_data, train_label, test_label = train_test_split(feature, y_label, test_size=0.33)
        start = time.clock()
        classifier = DecisionTree()
        classifier.train(train_data,train_label,0.1)
        end = time.clock()
        print("建立决策树完成，共花费{}秒".format(str(end - start)))
        start = time.clock()
        perdict = classifier.perdict(test_data)
        end = time.clock()
        print("预测结束，共花费{}秒".format(str(end - start)))
        score = accuracy_score(test_label, perdict)
        print(score)
if __name__ == '__main__':
    test = UnitTest()
    test.Test1()

