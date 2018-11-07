
import numpy as np
from collections import namedtuple
tree_Node = namedtuple('tree_Node',['value','left_child' ,'right_child'])#相当于struct
class KNN():
    def __init__(self,k=2,p=2):
        #p为范数的形式，默认为2范数
        self.k = k
        self.p = p
        self.kdtree = None

    def _build_kdtree(self,X,depth = 0):
        d = X.shape[1]
        axis = depth % d#当前操纵的坐标
        x_temp = X[:,axis].argsort()#排序的索引
        X = X[x_temp]
        mid = X.shape[0] // 2
        try:
            X[mid]
        except IndexError:
            return None
        return tree_Node(
            value=X[mid],
            left_child = KNN._build_kdtree(self,X=X[:mid],depth = depth+1),
            right_child = KNN._build_kdtree(self,X=X[mid+1:],depth = depth+1)
        )

    def _cal_distance(self,x,y):
        return np.linalg.norm(x-y,ord=self.p)

    def _search_node(self,query,node = None,current_nearest = None,depth = 0):
        if node is None:
            return current_nearest
        d = query.shape[0]
        if current_nearest is None or self._cal_distance(query,node.value)<self._cal_distance(current_nearest,node.value):
            new_current = node.value
        else:
            new_current = current_nearest
        axis = depth % d
        if query[axis]<node.value[axis]:
            #左子树
            node = node.left_child
        else:
            node = node.right_child
        return self._search_node(query,node,new_current,depth+1)

    def fit(self,X):
        self.kdtree = self._build_kdtree(X,0)

    def search(self,query):
        result = self._search_node(query=query,node=self.kdtree)
        return result

    def print_kdtree(self, kdtree_node):  # 中序遍历
        if kdtree_node is None:
            return None
        self.print_kdtree(kdtree_node.left_child)
        print(kdtree_node.value)
        self.print_kdtree(kdtree_node.right_child)















