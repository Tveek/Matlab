x=[0:1:99]';
y=[0:1:99]';
z=[0:1:99]';
plot3(x,y,z,'y');
hold on;
x=x+rand(100,1)*10;%x方向增加噪声
y=y+rand(100,1)*10;
z=z+rand(100,1)*10;
alpha=normal_equation(z,x);
beta=normal_equation(z,y);
alpha_0=alpha(1,1);
alpha_1=alpha(2,1);
beta_0=beta(1,1);
beta_1=beta(2,1);
x1=alpha_1*z+alpha_0;
y1=beta_1*z+beta_0;
z1=z;
plot3(x1,y1,z1,'r',x,y,z,'.')