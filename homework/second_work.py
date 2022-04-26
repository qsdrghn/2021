import sympy
from sympy import diff
import numpy as np
import random
from numpy.linalg import inv,norm
from Linesearch import *
from Mathcompute import *

# 使用非精确搜索的最速下降法，传入参数为求最优值的函数，精度误差，初始点
def Steepestdes(optiFun,err,value):
    x1 = value
    step_gra = Gradient(optiFun,x1)
    err_k = norm(step_gra)
    iter_num = 1
    while err_k > err:
        print('第'+str(iter_num)+'次迭代的误差为:',err_k)
        # 确定下降方向，进行非精确搜索，确定步长alpha的值
        d_k = -1 * step_gra
        alpha_k = Inexactsearch(optiFun,d_k,x1)
        # 更新新的x值
        x1 = x1 - alpha_k * step_gra
        step_gra = Gradient(optiFun,x1)
        err_k = norm(step_gra)
        iter_num = iter_num + 1
    print('第'+str(iter_num)+'次迭代的误差为:',err_k)
    return x1

def Zunewton(optiFun,err,value):
    x1 = value
    err_k = norm(Gradient(optiFun,x1))
    iter_num = 1
    while err_k > err:
        # 确定下降方向，进行精确搜索，确定步长alpha的值
        print('第' + str(iter_num) + '次迭代的误差为:', err_k)
        gra_k = Gradient(optiFun,x1)
        hess_k = Hessian(optiFun,x1)
        d_k = -1*gra_k.dot(inv(hess_k))
        alpha_k = Inexactsearch(optiFun,d_k,x1)
        # 更新新的x值和误差
        x1 = x1 + alpha_k * d_k
        err_k = norm(Gradient(optiFun,x1))
        iter_num = iter_num + 1
    print('第' + str(iter_num) + '次迭代的误差为:', err_k)
    return x1

# 传入参数为求最优值的函数，精度误差，初始点及决定是DFP还是BFGS的rho,其意义见书上BFGS部分的公式
def Quasinewton(optiFun,err,value,rho=0):
    # 先以单位矩阵作为第一次迭代的H进行计算
    x1 = value
    gra_k = Gradient(optiFun,x1)
    H = np.eye(value.shape[1])
    d_k = -1*gra_k.dot(H)
    # 根据下降方向确定步长，并得到更新后的x值及误差
    alpha_k = Inexactsearch(optiFun,d_k,x1)
    x2 = x1 + alpha_k*d_k
    err_k = norm(Gradient(optiFun,x2))
    iter_num = 1
    # 开始进行迭代，从而更新对称正定矩阵H和x值
    while err_k > err:
        print('第' + str(iter_num) + '次迭代的误差为:', err_k)
        del_x = x2 - x1
        del_y = Gradient(optiFun,x2) - Gradient(optiFun,x1)
        v = del_x.T/(del_x.dot(del_y.T)) - H.dot(del_y.T)/(del_y.dot(H)).dot(del_y.T)
        part_1 = del_x.T.dot(del_x)/del_x.dot(del_y.T)
        part_2 = (H.dot(del_y.T)).dot(del_y.dot(H.T))/(del_y.dot(H)).dot(del_y.T)
        H = H +  part_1 - part_2 + rho*((del_y.dot(H)).dot(del_y.T))*v.dot(v.T)
        x1 = x2
        gra_k = Gradient(optiFun,x1)
        d_k = -1 * gra_k.dot(H)
        alpha_k = Inexactsearch(optiFun,d_k,x1)
        x2 = x1 + alpha_k*d_k
        err_k = norm(Gradient(optiFun,x2))
        iter_num = iter_num + 1
    print('第' + str(iter_num) + '次迭代的误差为:', err_k)
    return x2

def Conjugategra(optiFun,err,value):
	# 先以初始点的梯度反方向为下降方向进行计算
    x1 = value
    d_k = -1*Gradient(optiFun,x1)
    # 根据下降方向确定步长，并得到更新后的x值及误差
    alpha_k = Inexactsearch(optiFun,d_k,x1)
    x2 = x1 + alpha_k*d_k
    err_k = norm(Gradient(optiFun,x2))
    # 开始进行迭代，从而更新下降方向和x值
    while err_k > err:
        gra_1 = Gradient(optiFun,x1)
        gra_2 = Gradient(optiFun,x2)
        beta = gra_2.dot(gra_2.T)/gra_1.dot(gra_1.T)
        d_k = -1*gra_2+beta*d_k
        x1 = x2
        alpha_k = Inexactsearch(optiFun,d_k,x1)
        x2 = x1 + alpha_k*d_k
        err_k = norm(Gradient(optiFun,x2))
    return x2

if __name__ == '__main__':
    # 定义初始点和精度误差
    value = np.array([[3, -1, 0, 1]])
    err = 0.0001
    # 定义搜索函数，这里直接定义所用变量和题目的函数
    x_1 = sympy.symbols('x_1')
    x_2 = sympy.symbols('x_2')
    x_3 = sympy.symbols('x_3')
    x_4 = sympy.symbols('x_4')
    optiFun = (x_1 + 10 * x_2) ** 2 + 5 * (x_3 - x_4) ** 2 + (x_2 - 2 * x_3) ** 4 + 10 * (x_1 - x_4) ** 4
    #Steepestdes(optiFun, err, value)
    #Zunewton(optiFun, err, value)
    Quasinewton(optiFun, err, value, 0)
    #Conjugategra(optiFun, err, value)