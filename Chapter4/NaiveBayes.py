import numpy as np
import cv2
class NaiveBayes():
    def __init__(self,X = None,Y = None,class_num = 10,feature_num = 784):
        self.prior_probability = np.zeros(class_num)
        self.conditional_probability = None
        self.X = X
        self.Y = Y
        self.class_num = 10
        self.feature_num = feature_num
    def binary(self,img):
        #二值化
        cv_img = img.astype(np.uint8)
        cv2.threshold(cv_img,50,1,cv2.THRESH_BINARY,cv_img)#将大于50的向量变为1，其余为0
        return cv_img
    def train(self,X,Y):
        sample_num = X.shape[0]
        #注意这里要二值化原来图像，不然无法统计
        #这里将所有特征范围压缩到2，即只有0和1
        self.conditional_probability = np.zeros((self.class_num,self.feature_num,2))
        for i in range(sample_num):
            current_img = self.binary(X[i])#二值化
            current_label = int(Y[i])
            self.prior_probability[current_label] += 1#计算先验概率
            for j in range(self.feature_num):
                self.conditional_probability[current_label][j][current_img[j]] +=1#计算条件概率
        #将数量转化为条件概率
        for i in range(self.class_num):
            for j in range(self.feature_num):
                num_1 = self.conditional_probability[i][j][1]
                num_0 = self.conditional_probability[i][j][0]
                self.conditional_probability[i][j][0] = num_0/(num_0+num_1)
                self.conditional_probability[i][j][1] = num_1/(num_1+num_0)


    def _cal_probability(self,img,label):
        # 计算当前图片的后验概率
        probability = self.prior_probability[label]
        for i in range(self.feature_num):
            probability *= self.conditional_probability[label][i][img[i]]
        return probability
    def perdict(self,test_img):
        #预测单张图片所属分类
        img = self.binary(test_img)
        max_pro = -1
        label = -1
        for i in range(self.class_num):
            temp_pro = self._cal_probability(img,i)
            if temp_pro > max_pro:
                max_pro = temp_pro
                label = i
        return label












