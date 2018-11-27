import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
class UnitTest():
    def Test1(self):
        raw_data = pd.read_csv("./data/train.csv").values
        x_data = raw_data[:,1:]
        y_label = raw_data[:,:1]
        #cross validation
        train_data, test_data, train_label, test_label = train_test_split(x_data, y_label, test_size=0.33)

