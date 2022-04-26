n =256;
a=unidrnd(10,n,1);
G=a*a'+unidrnd(2)*eye(n);
b=-0.5*G*ones(n,1);
x0=zeros(n,1);
eps=0.0001;
a1=zuisuxiajiangfa(G,b,x0,eps);
%a2=newton(G,b,x0,eps);
%a3=BFGS(G,b,x0,eps);
%a4=gongetidu(G,b,x0,eps);














