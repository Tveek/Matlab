X=[0:1:99]';
Y=[0:1:99]';
X=X+rand(100,1)*20;
plot(X,Y,'.');
hold on;
temp1=normal_equation(X,Y);
temp2=Gradient_descent(X,Y);
x=0:5:150;%¸ø¶¨x·¶Î§
k1=temp1(2,1);
b1=temp1(1,1)
k2=temp2(2,1);
b2=temp2(1,1)
plot(x,k1*x+b1)%»æÍ¼
plot(x,k2*x+b2)%»æÍ¼