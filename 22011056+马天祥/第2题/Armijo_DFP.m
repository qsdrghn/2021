% Armijo 线搜索  的DFP算法
function [x_star] = Armijo_DFP(x0,eps)
xk=x0;
gradk=f(xk);
n=length(xk);
Hk=eye(n);
res=norm(gradk);
alpha_bar = 1;
rho = 0.5;
c = 0.5;
k=0;
while res>eps
    k=k+1;
    dk=-Hk*gradk;
    %Armijo line search
    %alphak=1;
    %slope=gradk'*dk;
   % fk=obj(xk);
   % fnew=obj(xk+alphak*dk);
   % iterk=0;
   % while fnew>fk+c*alphak*slope 
    %     iterk=iterk+1;
   %      alphak=0.5*alphak;
   %      fnew=obj(xk+alphak*dk);     
   % end
    slope = gradk'*dk;
    fun0 = obj(xk);
    m = 0;
    x_new = xk + alpha_bar*rho^m*dk;
    fun1 = obj(x_new);
    while fun1 - fun0 > c*alpha_bar*rho^m*slope & m <= 200
            m = m + 1;
            x_new = xk + alpha_bar*rho^m*dk;
            fun1 = obj(x_new);
    end
    alphak = alpha_bar * rho^m;
    
    
    
    deltak=alphak*dk;
    xk=xk+deltak;
    grad0=gradk;
    gradk=f(xk);
    gk=gradk-grad0;
   % muk=1+(gk'*Hk*gk)/(deltak'*gk);
    %Hk=Hk+(muk*(deltak*deltak')-Hk*gk*deltak'-deltak*gk'*Hk)/(deltak'*gk);
   
    Hk=Hk+(deltak*deltak')/(deltak'*gk)-Hk*gk*(Hk*gk)'/(gk'*Hk*gk);
    
    res=norm(gradk);
    fprintf('At the %d-th iteration, the residual is ------- %f\n',k,res);
end
x_star=xk;
end

