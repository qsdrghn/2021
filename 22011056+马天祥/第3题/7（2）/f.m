function [f,grad] = f(x)
%求x点处函数值与梯度
global M
f=x(1)^2+x(2)^2-16*x(1)-10*x(2);
grad=[2*x(1)-16;2*x(2)-10];
g1=x(1)^2-6*x(1)+4*x(2)-11;
g2=
%f=-x(1)*x(2);
%grad=[-x(2);-x(1)];
%g1=-x(1)-x(2)^2+1;
%g2=x(1)+x(2);
if g1<0
    f=f+M*g1^2;
    grad=grad+[-2*g1;-4*g1*x(2)].*M;
end
if g2<0
    f=f+M*g2^2;
    grad=grad+[2*g2;2*g2].*M;
end
end



