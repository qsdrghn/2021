function[x_star,exit]= BGFS_sumt(x)
% 使用BGFS方法确定xk+1,并判断xk+1能否作为最优点
global M
exit=0;
k=0;
Hk=eye(2,2);
while 1
    [fk,gk]=f(x);
    if norm(gk)<0.00001
        if (fk-f2(x))/M<0.00001
            exit=1;
        end
        break
    end
    if k==0
        Hk=eye(2,2);
    else
        x_deta=x-x_old;
        gk_deta=gk-gk_old;
        vk=x_deta./(x_deta'*gk_deta)-Hk*gk_deta./(gk'*Hk*gk);
        Hk=Hk+(gk_deta'*Hk*gk_deta)*vk*vk'-Hk*(gk_deta*gk_deta')*Hk./(gk_deta'*Hk*gk_deta)+x_deta*x_deta'./(x_deta'*gk_deta);
    end
    s=-Hk*gk;
    x_old=x;
    gk_old=gk;
    x=armoji(x,s,fk,gk);
    if k==1
        k=0;
    else
        k=k+1;
    end
end
x_star=x;
end

function[f]=f2(x)
f=-x(1)*x(2);

end





