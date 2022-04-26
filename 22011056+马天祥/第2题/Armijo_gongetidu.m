%Armijo_共轭梯度法（迭代不出结果）
function [x_star] = Armijo_gongetidu(x0,eps)
xk=x0;
gradk=f(xk);
res=norm(gradk);
a=0;
n=length(xk);
dk1=ones(n,1);
k=0;
alpha_bar = 1;
rho = 0.5;
c = 0.5;
fprintf('At the %d-th iteration, the residual is ------- %f\n',k,res);
while res>eps
    k=k+1;
    dk=-gradk+a*dk1;%方向
    
    slope = gradk'*dk;
    fun0 = obj(xk);
    m = 0;
    x_new = xk + alpha_bar*rho^m*dk;
    fun1 = obj(x_new);
    while fun1 - fun0 > c*alpha_bar*rho^m*slope 
            m = m + 1;
            x_new = xk + alpha_bar*rho^m*dk;
            fun1 = obj(x_new);
    end
    alphak = alpha_bar * rho^m;
   
    deltak=alphak*dk;
    xk=xk+deltak;
    gradk1=gradk;
    gradk=f(xk);
    a=(gradk'*gradk)/(gradk1'*gradk1);
    res=norm(gradk);
    dk1=dk;
    fprintf('At the %d-th iteration, the residual is ------- %f\n',k,res);
end
x_star=xk;
end

