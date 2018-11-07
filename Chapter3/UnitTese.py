import numpy as np
from Knn import *
import time
class UnitTest():
    def brute_force(self,x,query):
        min = 1000000
        nearest = None
        for i in range(x.shape[0]):
            if np.linalg.norm(x[i]-query)<min:
                min = np.linalg.norm(x[i]-query,ord=2)
                nearest = x[i]
        return nearest

    def test1(self):
        x = np.loadtxt("Input/data_3-1.txt")
        query = np.array([3,3])
        knn = KNN()
        knn.fit(x)
        knn.print_kdtree(knn.kdtree)
        print(knn.search(query))

    def test2(self):
        num = 10000
        d = 2
        x = np.random.randint(0,100,[num,d])#随机的100个二维点，范围为0到100
        query = np.array([50,50])
        knn = KNN()
        #build kdtree
        start = time.clock()
        knn.fit(x)
        end = time.clock()
        print("{0}个{1}维点，建立kd树所花的时间为：{2}".format(num,d,str(end-start)))

        #search kdtree
        print("最近点为：")
        start = time.clock()
        print(knn.search(query))
        end = time.clock()
        print("{0}个{1}维点，搜索最近邻所花费时间：{2}".format(num,d,str(end - start)))
        #brute force search
        start = time.clock()
        print("暴力搜索最近点为:")
        print(self.brute_force(x,query))
        end = time.clock()
        print("{0}个{1}维点，使用暴力搜索花费的时间为：{2}".format(num, d, str(end - start)))
if __name__ == '__main__':
    a = UnitTest()
    a.test2()

