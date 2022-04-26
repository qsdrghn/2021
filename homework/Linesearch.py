import sympy
from Mathcompute import *
from sympy import diff
import numpy as np
import random
from numpy.linalg import inv,norm


def Accuracysearch(optiFun, searchErr):
    # 精确搜索————进退算法确定区间加0.618确定搜索值,传入的函数以alpha为变量
    # 虽然进退算法并不能一定找到单峰区间，但对于全是凸函数的作业题足够了
    alpha_zero = 0
    alpha = sympy.symbols('alpha')
    alpha1 = alpha_zero
    h = 10
    alpha2 = alpha_zero + h
    if optiFun.subs(alpha, alpha2) > optiFun.subs(alpha, alpha_zero):
        h = -1 * h
    alpha1 = alpha_zero + h
    while optiFun.subs(alpha, alpha1) <= optiFun.subs(alpha, alpha_zero):
        h = 2 * h
        alpha2 = alpha_zero
        alpha_zero = alpha1
        alpha1 = alpha_zero + h
    left = min(alpha1, alpha2)
    right = max(alpha1, alpha2)
    # 0.618算法（可以替换成其他的，不推荐抛物线法，此法需要找到合适的三个点）
    lam = left + 0.382 * (right - left)
    mu = left + 0.618 * (right - left)
    while (right - left) > searchErr:
        if optiFun.subs(alpha, left) > optiFun.subs(alpha, right):
            left = lam
            lam = mu
            mu = left + 0.618 * (right - left)
        else:
            right = mu
            mu = lam
            lam = left + 0.382 * (right - left)
    return (left + right) / 2

# 非精确搜索，使用Wolfe准则
def Inexactsearch(optiFun,d,value):
    c_1 = random.uniform(0, 1)
    c_2 = random.uniform(c_1, 1)
    alpha = 1
    n = -9999999999999
    m = 0
    x1 = value
    x_symbol = sympy.symbols('x_1:' + str(value.shape[1] + 1))
    x1_value = dict(zip(x_symbol, x1[0]))
    x2 = x1 + alpha * d
    x2_value = dict(zip(x_symbol, x2[0]))
    # 非精确搜索的两个条件
    conditon_1 = (optiFun.subs(x2_value) - optiFun.subs(x1_value)) <= c_1 * alpha * np.dot(Gradient(optiFun, x1), d.T)
    conditon_2 = np.dot(Gradient(optiFun, x2), d.T) >= c_2 * np.dot(Gradient(optiFun, x1), d.T)
    # 在不满足条件的情况下继续搜索步长
    while not (conditon_1 and conditon_2):
        if conditon_1:
            m = alpha
            alpha = min(2 * alpha, (alpha + n) / 2)
        else:
            n = alpha
            alpha = (alpha + m) / 2
        x2 = x1 + alpha * d
        x2_value = dict(zip(x_symbol, x2[0]))
        conditon_1 = (optiFun.subs(x2_value) - optiFun.subs(x1_value)) <= c_1 * alpha * np.dot(Gradient(optiFun, x1),
                                                                                               d.T)
        conditon_2 = np.dot(Gradient(optiFun, x2), d.T) >= c_2 * np.dot(Gradient(optiFun, x1), d.T)
    return alpha


# 专门用于第三题第二问的情况的非精确搜索函数，使用Wolfe准则
# 因为不再使用符号变量定义目标函数和约束集合，因此不再传入所优化的函数
# 改成传入判断不等式是否作为惩罚项的λ和σ
def Inexactsearch_1(d,value,lam,sigma):
    d = d[0]
    c_1 = 0.5
    c_2 = random.uniform(c_1, 1)
    alpha = 1
    n = -9999999999999
    m = 0
    x1 = value[0]
    x2 = x1 + alpha * d
    # 非精确搜索的两个条件
    acon = active(x1,lam,sigma)
    conditon_11 = fun(x2)+active_1(x2,acon).dot(lam)+(active_2(x2,acon).dot(np.ones(4)))*sigma/2- fun(x1)-active_1(x1,acon).dot(lam)-(active_2(x1,acon).dot(np.ones(4)))*sigma/2
    conditon_12 = c_1 * alpha * Gradient_1(x1,acon,lam,sigma).dot(d)
    conditon_1 = conditon_11 <= conditon_12
    conditon_2 = Gradient_1(x2,acon,lam,sigma).dot(d) >= c_2 * Gradient_1(x1,acon,lam,sigma).dot(d)
    # 在不满足条件的情况下继续搜索步长
    while not (conditon_1 and conditon_2):
        print(conditon_11,conditon_12,conditon_1,conditon_2)
        if conditon_1:
            m = alpha
            alpha = min(2 * alpha, (alpha + n) / 2)
        else:
            n = alpha
            alpha = (alpha + m) / 2
        x2 = x1 + alpha * d
        acon_2 = active(x2,lam,sigma)
        conditon_11 = fun(x2)+active_1(x2,acon).dot(lam)+(active_2(x2,acon).dot(np.ones(4)))*sigma/2- fun(x1)-active_1(x1,acon).dot(lam)-(active_2(x1,acon).dot(np.ones(4)))*sigma/2
        conditon_12 = c_1 * alpha * Gradient_1(x1,acon,lam,sigma).dot(d)
        conditon_1 = conditon_11 <= conditon_12
        conditon_2 = Gradient_1(x2,acon,lam,sigma).dot(d) >= c_2 * Gradient_1(x1,acon,lam,sigma).dot(d)
    return alpha
