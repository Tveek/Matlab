function theta=Gradient_descent(X,Y)  
    theta0=0;  
    theta1=0;      
    t0=0;  
    t1=0;  
    while(1)
        for i=1:1:100 %100个点  
            t0=t0+(theta0+theta1*X(i,1)-Y(i,1))*1; %x0=1
            t1=t1+(theta0+theta1*X(i,1)-Y(i,1))*X(i,1);  
        end  
        old_theta0=theta0;  
        old_theta1=theta1;  
        theta0=theta0-0.000001*t0; %0.000001表示学习率  
        theta1=theta1-0.000001*t1;  
        t0=0;  
        t1=0;  
        if(sqrt((old_theta0-theta0)^2+(old_theta1-theta1)^2)<0.0001) % 这里是判断收敛的条件，当然可以有其他方法来做  0.000001
            break;  
        end 
        theta(1,1)=theta0;
        theta(2,1)=theta1;
    end 