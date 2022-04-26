function [x_star] = SUMT(x)
k=1;
c=5;
global M
M=1;
while 1
   [x,exit]=BGFS_sumt(x);
   if exit==1
       x_star=x;
       break
   end
   M=c*M; 
   k=k+1;
end
sprintf('迭代次数：k=%d',k)
a=f2(x);
sprintf('最小值：a=%d',a)
end


function[f]=f2(x)
f=-x(1)*x(2);

end

%SUMT([0;-1])