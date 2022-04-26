function[x_new]=armoji(x_old,s,f_old,grad_old)
m=0;
rho=0.5;
c=0.8;
while m<20
    [f_new,~]=f(x_old+rho^m*s);
    if f_new-f_old<=c*rho^m*(grad_old'*s)        
        break
    else
        m=m+1;
    end  
end
mk=m;   
l=rho^mk;  
x_new=x_old+s.*l;
end

