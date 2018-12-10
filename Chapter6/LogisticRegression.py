import numpy as np
import pandas as pd
import random
import math
class LogisticRegression():
    def __init__(self,learning_rate = 0.001,epoch = 2000):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.w = np.array([])
        self.cols = []

    def _gradient_descent(self,x,y):
        count = 0
        self.w = np.zeros([x.shape[1]])
        while count<self.epoch:
            index = random.randint(0,len(y)-1)
            current_x = x[index]
            current_y = y[index]
            count +=1
            wx = sum(self.w[i]*current_x[i] for i in range(x.shape[1]))
            exp_wx = math.exp(wx)
            for i in range(self.w.shape[0]):
                #(yj-h(xj))*xj
                self.w[i] -= self.learning_rate*(-current_y*current_x[i]
                                                 +current_x[i]*exp_wx/(1+exp_wx))
    def train(self,x,y):
        self._gradient_descent(x,y)
    def _perdict(self,x):
        wx = sum([self.w[j] * x[j] for j in range(self.w.shape[0])])
        try:
            exp_wx = math.exp(wx)
        except OverflowError:
            exp_wx = float('inf')
        perdict_score = []
        #for i in range(10):
            #current_per =
        predict1 = exp_wx / (1 + exp_wx)
        #sigmoid
        predict0 = 1 / (1 + exp_wx)

        if predict1 > predict0:
            return 1
        else:
            return 0
    def perdict(self,x_datas):
        labels = []
        for data in x_datas:
            score = self._perdict(data)
            labels.append(score)
        return labels




