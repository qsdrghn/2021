%Armijo_最速下降法(迭代11539次，结束）
function [x_star] = Armijo_zuisuxiajiangfa(x0,eps)
xk=x0;
gradk=f(xk);
res=norm(gradk);
k=0;
alpha_bar = 1;
rho = 0.5;
c = 0.5;
fprintf('At the %d-th iteration, the residual is ------- %f\n',k,res);
while res>eps 
    k=k+1;
    dk=-gradk;
    
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
    
    xk=xk+alphak*dk;
    gradk=f(xk);
    res=norm(gradk);
   fprintf('At the %d-th iteration, the residual is ------- %f\n',k,res);
end
x_star=xk;
end

