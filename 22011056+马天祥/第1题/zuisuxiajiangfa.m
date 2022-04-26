%æœ??ä¸‹é™æ³?
function [x_star] = zuisuxiajiangfa(G,c,x0,eps)
xk=x0;
gradk=G*xk+c;
res=norm(gradk);
k=0;
fprintf('At the %d-th iteration, the residual is ------- %f\n',k,res);
while res>eps
    k=k+1;
    dk=-gradk;
    alphak=-(gradk'*dk)/(dk'*G*dk);
    xk=xk+alphak*dk;
    gradk=G*xk+c;
    res=norm(gradk);
   fprintf('At the %d-th iteration, the residual is ------- %f\n',k,res);
end
x_star=xk;
return xk
end
