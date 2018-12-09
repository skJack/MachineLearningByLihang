import numpy as np
import pandas as pd
class LogisticRegression():
    def __init__(self,learning_rate = 0.001,epoch = 2000):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.w = np.array([])
        self.cols = []

    def gradient_descent(self,x,y,rate = 0.0001,epoch = 2000):
        n = x.shape()