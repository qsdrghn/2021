%BFGS
function [x_star] = BFGS(G,c,x0,eps)
xk=x0;
gradk=G*xk+c;
n=length(xk);
Hk=eye(n);
res=norm(gradk);
k=0;
fprintf('At the %d-th iteration, the residual is ------- %f\n',k,res);
while res>eps
    k=k+1;
    dk=-Hk*gradk;%方向
    alphak=-(gradk'*dk)/(dk'*G*dk);%沿着方向走多少
    deltak=alphak*dk;
    xk=xk+deltak;
    grad0=gradk;
    gradk=G*xk+c;
    gk=gradk-grad0;
    muk=1+(gk'*Hk*gk)/(deltak'*gk);
    Hk=Hk+(muk*(deltak*deltak')-Hk*gk*deltak'-deltak*gk'*Hk)/(deltak'*gk);
    res=norm(gradk);
    fprintf('At the %d-th iteration, the residual is ------- %f\n',k,res);
end
x_star=xk;
end

