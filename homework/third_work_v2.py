import sympy
from sympy import diff
from sympy import *
import numpy as np
import random
from numpy.linalg import inv,norm
from Linesearch import *
from Mathcompute import *

def Augmentlagra(optiFun,equalConstra,inequalConstra,err,value):
	x1 = value
	err_k =999
	sigma = 1
	equal_lam = np.ones(len(equalConstra))
	inequal_lam = np.ones(inequalConstra)
	iter_num = 1
	while err_k > err:
		print('第' + str(iter_num) + '次迭代的误差为:', err_k)
		acon_1 = active(x1[0],inequal_lam,sigma)
		# 开始使用非精确搜索的BFGS算法进行求该增广函数对应的优化函数
		gra_k = np.array([list(Gradient_1(x1[0],acon_1))])
		H = np.eye(x1.shape[1])
		d_k = -1*gra_k.dot(H)
		alpha_k = Inexactsearch_1(d_k,x1,inequal_lam,sigma)
		x2 = x1 + alpha_k*d_k
		acon_2 = active(x2[0],inequal_lam,sigma)
		gra_k1 = Gradient_1(x2[0],acon_2)
		errBfgs = norm(gra_k1)
		while errBfgs > err:
			del_x = x2 - x1
			del_y = np.array([list(gra_k1)]) - np.array([list(gra_k)])
			v = del_x.T/(del_x.dot(del_y.T)) - H.dot(del_y.T)/(del_y.dot(H)).dot(del_y.T)
			part_1 = del_x.T.dot(del_x)/del_x.dot(del_y.T)
			part_2 = (H.dot(del_y.T)).dot(del_y.dot(H.T))/(del_y.dot(H)).dot(del_y.T)
			H = H +  part_1 - part_2 + ((del_y.dot(H)).dot(del_y.T))*v.dot(v.T)
			x1 = x2
			acon_1 = acon_2
			# 先判断此点处属于分段函数哪一段
			# 构造初始点处的优化函数并开始计算
			gra_k = gra_k1
			d_k = -1*gra_k.dot(H)
			alpha_k = Inexactsearch_1(d_k,x1,inequal_lam,sigma)
			x2 = x1 + alpha_k*d_k
			acon_2 = active(x2[0],inequal_lam,sigma)
			gra_k1 = Gradient_1(x2[0],acon_2)
			errBfgs = norm(gra_k1)
		x1 = x2
		acon_1 = active(x1[0],inequal_lam,sigma)
		# 先判断此点处属于分段函数哪一段
		err_k = err_au(x1[0],acon_1)
		iter_num = iter_num + 1
	print('第' + str(iter_num) + '次迭代的误差为:', err_k)
	return x1