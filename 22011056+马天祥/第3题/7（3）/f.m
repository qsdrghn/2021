function [f,grad] = f(x)
%求x点处函数值与梯度
global M
f=4*x(1)-x(2)^2-12;
grad=[4;-2*x(2)];
h1=25-x(1)^2-x(2)^2;
f=f+M*h1^2;
grad=grad+[-2*x(1)*2*h1*M;-2*x(2)*2*h1*M];
g1=10*x(1) - x(1)^2 + 10*x(2) - x(2)^2 - 34;
g2=x(1);
g3=x(2);
if g1<0
    f=f+M*g1^2;
    grad=grad+[10-2*x(1);10-2*x(2)].*(2*g1*M);
end
if g2<0
    f=f+M*g2^2;
    grad=grad+[1;0].*(2*g2*M);
end
if g3<0
    f=f+M*g3^2;
    grad=grad+[0;1].*(2*g3*M);
end
end

