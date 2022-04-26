%共轭梯度法
function [x_star] = gongetidu(G,c,x0,eps)
xk=x0;
gradk=G*xk+c;
res=norm(gradk);
a=0;
n=length(xk);
dk1=ones(n,1);
k=0;
fprintf('At the %d-th iteration, the residual is ------- %f\n',k,res);
while res>eps
    k=k+1;
    dk=-gradk+a*dk1;%方向
    alphak=-(gradk'*dk)/(dk'*G*dk);%沿着方向走多少
    deltak=alphak*dk;
    xk=xk+deltak;
    gradk1=gradk;
    gradk=G*xk+c;
    a=(gradk'*gradk)/(gradk1'*gradk1);
    res=norm(gradk);
    dk1=dk;
    fprintf('At the %d-th iteration, the residual is ------- %f\n',k,res);
end
x_star=xk;
end

