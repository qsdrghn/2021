x0=[3;-1;0;1];
eps=1e-4;
%x_finish1=Armijo_DFP(x0,eps)
x_finish2=Armijo_newton(x0,eps)
%x_finish3=Armijo_gongetidu(x0,eps)
%x_finish4=Armijo_zuisuxiajiangfa(x0,eps)