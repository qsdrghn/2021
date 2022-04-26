import sympy
from sympy import diff
import numpy as np

# 求任意函数某点的梯度，默认传入的函数是x_1,x_2这种格式变量构成的函数
# value代表传入的该点处的x值
def Gradient(fun,value):
    x_symbol = sympy.symbols('x_1:'+str(value.shape[1]+1))
    x_value = dict(zip(x_symbol,value[0]))
    gra = np.array([[diff(fun,i).subs(x_value) for i in x_symbol]]).astype(np.float64)
    return gra

# 求任意函数某点的HESSIAN矩阵，传入参数特点同上述求梯度的函数
def Hessian(fun,value):
	x_symbol = sympy.symbols('x_1:'+str(value.shape[1]+1))
	x_value = dict(zip(x_symbol,value[0]))
	return np.array(sympy.hessian(fun,x_symbol).subs(x_value)).astype(np.float64)


# 用于第三题第二小问的梯度计算，直接给出导数公式
# 并且仅计算不等式约束的梯度
def Gradient_1(value,acon):
    gra_1 = [2*x[0] - 6,E**(x[0]-3)-x[1],-1,0]
    gra_2 = [4,3 - x[0],0,-1]
    fun_gra = np.array([2*x[0]-16,2*x[1]-10])
    return fun_gra + ((np.array([gra_1,gra_2])*(acon_1)).dot(np.ones(4)).astype(np.float64))


# 定义优化函数fun和各个约束，均为标准形式的左半部分
# 传入的参数x为变量的值构成的列表
fun = lambda x: x[0]**2+x[1]**2-16*x[0]-10*x[1]
h1 = lambda x: x[0]**2 - 6*x[0] + 4*x[1] - 11
h2 = lambda x: 3*x[1] + E**(x[0]-3) - x[0]*x[1] - 1
h3 = lambda x: -1*x[0]
h4 = lambda x: -1*x[1]

# 判断不等式约束中起作用的约束，返回一个列表，索引对应着约束的索引
# 0代表不起作用，1代表起作用
def active(x,lams,sigma):
    acon = []
    for h,lam in zip([h1(x),h2(x),h3(x),h4(x)],lams):
        if (lam+sigma*h) > 0:
            acon.append(1)
        else:
            acon.append(0)
    return np.array(acon)

# 计算增广拉格朗日函数关于约束的乘子部分
def active_1(x,acon):
    return (np.array([h1(x),h2(x),h3(x),h4(x)])*acon).astype(np.float64)

# 计算增广拉格朗日函数关于约束的惩罚项部分
def active_2(x,acon):
    return (np.array([h1(x)**2,h2(x)**2,h3(x)**2,h4(x)**2])*(acon)).astype(np.float64)

# 计算增广拉格朗日函数在某点的梯度
def Gradient_1(x,acon,lam,sigma):
    gra_1 = [2*x[0] - 6,E**(x[0]-3)-x[1],-1,0]
    gra_2 = [4,3 - x[0],0,-1]
    fun_gra = np.array([2*x[0]-16,2*x[1]-10])
    return fun_gra + ((np.array([gra_1,gra_2])*acon).dot(lam)).astype(np.float64) + sigma*((np.array([gra_1,gra_2])*active_1(x,acon)).dot(np.ones(4))).astype(np.float64)

# 计算一次迭代的误差
def err_au(x,acon,lams,sigma):
    err_k = 0
    for h,lam in zip([h1(x),h2(x),h3(x),h4(x)],lams):
        if (lam+sigma*h) > 0:
            err_k = err_k + h**2
        else:
            err_k = err_k + (lam/sigma)**2
    return err_k
    
# 更新增广拉格朗日函数的乘子
def updatelam(x,lams,sigma,acon):
    return (sigma*np.array([h1(x),h2(x),h3(x),h4(x)])+lams)*(acon).astype(np.float64)