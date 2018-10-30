#!/usr/bin/python
import numpy as np
import random
import matplotlib.pyplot as plt
class Perception:
	def __init__(self,max_iter = 10,learning_rate = 1,x_sample = None,y_label = None):
		self.max_iter = max_iter
		self.learning_rate = learning_rate
		self.w = 0
		self.b = 0
		self.w_list = []
		self.b_list = []#存储所有的迭代w,b值，方便观察
		self.x_sample = x_sample
		self.y_label = y_label
	def train(self,x_data,y_label):
		self.x_sample = x_data
		self.y_label = y_label
		self.w = np.zeros(x_data.shape[1])
		for i in range(50):
			#random_index = random.randint(0,x_data.shape[1])#随机挑选样本
			random_index = i%(x_data.shape[1]+1)
			wx = np.sum(np.dot(self.w,x_data[random_index]))#需要改成内积数字计算wx的和，作为中间结果
			yx = np.dot(x_data[random_index],y_label[random_index])#计算yx 为二维向量
			y = y_label[random_index]*(wx+self.b)#当前预测值注意括号
			if y<=0:
				#未被正确分类，需要修改w,b
				self.w = self.w + self.learning_rate*yx
				self.b = self.b + y_label[random_index]
				self.w_list.append(self.w)
				self.b_list.append(self.b)
				Perception.print_para(self)
	def print_para(self):
		print("w = ",self.w)
		print("b = ",self.b)
	def plot_result(self):
		x1 = np.linspace(0, 6, 6)  # 在0到6中取6个点
		#画点
		for i in range(self.x_sample.shape[1] + 1):
			plt.text(self.x_sample[i][0], self.x_sample[i][1], str((self.x_sample[i][0], self.x_sample[i][1])), ha='right')
			if self.y_label[i]==1:
				plt.plot(self.x_sample[i][0], self.x_sample[i][1], 'ro',color = 'red')
			else:
				plt.plot(self.x_sample[i][0], self.x_sample[i][1], 'ro', color='blue')
		plt.xlim(-6, 6)
		plt.ylim(-6, 6)
		#画分类面
		for i in range(len(self.w_list)-1,0,-1):#注意倒序要减一
			if self.w_list[i][1] == 0:
				continue
			y_axis = -self.b_list[i] - self.w_list[i][0] * x1 / self.w_list[i][1]
			plt.plot(x1, y_axis)
		plt.show()










		