from Perception import *
import numpy as np

class TestPerception:
    def test1(self):
        data_matrix = np.loadtxt("data_2-1.txt")
        X = data_matrix[0:3, 0:2]#前三行的前两行元素
        Y = data_matrix[:, -1]#最后一列
        perception = Perception()
        perception.train(X,Y)
        perception.plot_result()
        #perception.print_para()
        #perception.plot()
if __name__=='__main__':
	a = TestPerception()
	a.test1()