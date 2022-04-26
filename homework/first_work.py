import sympy
from sympy import diff
import numpy as np
import random
from numpy.linalg import inv,norm
from Linesearch import *
from Mathcompute import *

# 学号：22111076  姓名：卢常建


# 此为一维精确搜索的最速下降法，传入的参数为求最优值的函数，精度误差，初始点
def Steepestdes(err,value,G,b):
    x1 = value
    x_symbol = sympy.symbols('x_1:'+str(value.shape[1]+1))
    step_gra = G.dot(x1.T) + b
    err_k = norm(step_gra)
    iter_num = 1
    while err_k > err:
        print('第'+str(iter_num)+'次迭代的误差为:',err_k)
        # 确定下降方向，进行一维精确搜索，确定alpha的值
        d = -1*step_gra
        alpha_k = -1*step_gra.T.dot(d)/(d.T.dot(G.dot(d)))
        alpha_k = alpha_k[0][0]
        # 更新新的x值
        x1 = x1 + alpha_k * d.T
        step_gra = G.dot(x1.T) + b
        err_k = norm(step_gra)
        iter_num = iter_num + 1
    print('第'+str(iter_num)+'次迭代的误差为:',err_k)
    return x1

# 此为一维精确搜索的阻尼牛顿法，传入参数为求最优值的函数，精度误差，初始点
def Zunewton(err,value,G,b):
    x1 = value
    step_gra = G.dot(x1.T) + b
    err_k = norm(step_gra)
    iter_num = 1
    while err_k > err:
        print('第'+str(iter_num)+'次迭代的误差为:',err_k)
        # 进行一维精确搜索，确定alpha的值
        hess_k = G
        d = -1*inv(hess_k).dot(step_gra)
        alpha_k = -1*step_gra.T.dot(d)/(d.T.dot(G.dot(d)))
        alpha_k = alpha_k[0][0]
        # 更新新的x值
        x1 = x1 + alpha_k * d.T
        step_gra = G.dot(x1.T) + b
        err_k = norm(step_gra)
        iter_num = iter_num + 1
    print('第'+str(iter_num)+'次迭代的误差为:',err_k)
    return x1

# 一维精确搜索的拟牛顿法
# 传入参数为求最优值的函数，精度误差，初始点及决定是DFP还是BFGS的rho,其意义见书上BFGS部分的公式
def Quasinewton(err,value,G,b,rho):
    # rho决定是DFP还是BFGS
    x1 = value
    gra_k = G.dot(x1.T) + b
    H = np.eye(value.shape[1])
    d= -1*H.dot(gra_k)
    alpha_k = -1*gra_k.T.dot(d)/(d.T.dot(G.dot(d)))
    alpha_k = alpha_k[0][0]
    x2 = x1 + alpha_k*d.T
    err_k = norm(G.dot(x2.T) + b)
    iter_num = 1
    while err_k > err:
        print('第'+str(iter_num)+'次迭代的误差为:',err_k)
        del_x = x2 - x1
        del_y = (G.dot(x2.T) - G.dot(x1.T)).T
        v = del_x.T/(del_x.dot(del_y.T)) - H.dot(del_y.T)/(del_y.dot(H)).dot(del_y.T)
        part_1 = del_x.T.dot(del_x)/del_x.dot(del_y.T)
        part_2 = (H.dot(del_y.T)).dot(del_y.dot(H.T))/(del_y.dot(H)).dot(del_y.T)
        H = H +  part_1 - part_2 + rho*((del_y.dot(H)).dot(del_y.T))*v.dot(v.T)
        x1 = x2
        gra_k = G.dot(x1.T) + b
        d= -1*H.dot(gra_k)
        alpha_k = -1*gra_k.T.dot(d)/(d.T.dot(G.dot(d)))
        x2 = x1 + alpha_k*d.T
        err_k = norm(G.dot(x2.T) + b)
        iter_num = iter_num + 1
    print('第'+str(iter_num)+'次迭代的误差为:',err_k)
    return x2

# 此为一维精确搜索的共轭梯度法，传入的参数为最优值的函数，精度误差，初始点
def Conjugategra(err,value,G,b):
    x1 = value
    gra_k = G.dot(x1.T) + b
    d = -1*gra_k
    alpha_k = -1*gra_k.T.dot(d)/(d.T.dot(G.dot(d)))
    alpha_k = alpha_k[0][0]
    x2 = x1 + alpha_k*d.T
    err_k = norm(G.dot(x2.T) + b)
    iter_num = 1
    while err_k > err:
        print('第'+str(iter_num)+'次迭代的误差为:',err_k)
        gra_1 = G.dot(x1.T) + b
        gra_2 = G.dot(x2.T) + b
        beta = gra_2.T.dot(gra_2)/gra_1.T.dot(gra_1)
        d = -1*gra_2+beta*d
        x1 = x2
        alpha_k = -1*gra_k.T.dot(d)/(d.T.dot(G.dot(d)))
        alpha_k = alpha_k[0][0]
        x2 = x1 + alpha_k*d.T
        err_k = norm(G.dot(x2.T) + b)
    print('第'+str(iter_num)+'次迭代的误差为:',err_k)
    return x2

if __name__ == '__main__':
    n = 276
    a = np.random.randint(1,10,[n,1])
    G = np.dot(a,a.T) + random.randint(1,2)*np.eye(n)
    b = 0.5*np.dot(G,np.ones([n,1]))
    x_array = np.array(sympy.symbols('x_1:'+str(n+1)))
    fun_array = 0.5*(x_array.dot(G)).dot(x_array.T)+np.dot(b.T,x_array.T)
    fun=fun_array[0]
    Steepestdes(fun,0.01,np.zeros([1,n]),G,b)
    Zunewton(0.01, value, G, b)
    Quasinewton(0.01, value, G, b, 1)
    Conjugategra(0.01, value, G, b)
