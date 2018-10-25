#!/usr/bin/python
		
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from perceptron import *
import numpy as np
import argparse
import logging
import unittest
class TestPerception:
	def test1(self):
		data_row = np.loadtxt("Input/data_2-1.txt")
		X = data_row[0:3, 0:2]#前三行的前两行元素
		Y = data_row[:, -1]#最后一列
		print(data_row)
		print(X)
		print(Y)
if __name__=='__main__':
	a = TestPerception()
	a.test1()
	
	