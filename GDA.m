% Gaussian Discriminate Analysis
clc; clf;
clear all

% 随机产生2类高斯分布样本
mu = [2 3];
SIGMA = [1 0; 0 2];
x0 = mvnrnd(mu,SIGMA,500);%Multivariate normal random numbers
y0 = zeros(length(x0),1);
plot(x0(:,1),x0(:,2),'k+', 'LineWidth',1.5, 'MarkerSize', 7);
hold on;
mu = [7 8];
SIGMA = [ 1 0; 0 2];
x1 = mvnrnd(mu,SIGMA,200);
y1 = ones(length(x1),1);
plot(x1(:,1),x1(:,2),'ro', 'LineWidth',1.5, 'MarkerSize', 7)

x = [x0;x1];
y = [y0;y1];
m = length(x);

% 计算参数: \phi,\u0,\u1,\Sigma
phi = (1/m)*sum(y==1);
u0 = mean(x0,1);
u1 = mean(x1,1);
x0_sub_u0 = x0 - u0(ones(length(x0),1), :);
x1_sub_u1 = x1 - u1(ones(length(x1),1), :);
x_sub_u = [x0_sub_u0; x1_sub_u1];
sigma = (1/m)*(x_sub_u'*x_sub_u);

%% Plot Result
% 画分界线,Ax+By=C
u0 = u0';
u1 = u1';
a=sigma'*u1-sigma'*u0;  
b=u1'*sigma'-u0'*sigma';  
c=u1'*sigma'*u1-u0'*sigma'*u0;  
A=a(1)+b(1);
B=a(2)+b(2);  
C=c;  
x=-2:10;  
y=-(A.*x-C)/B;  
hold on;  
plot(x,y,'LineWidth',2);

% 画等高线
alpha = 0:pi/30:2*pi;%一个向量，范围是【0，2pei】，每隔pi/30取一个数
R = 3.3;
cx = u0(1)+R*cos(alpha);
cy = u0(2)+R*sin(alpha);
plot(cx,cy,'b-');
hold on;
cx = u1(1)+R*cos(alpha);
cy = u1(2)+R*sin(alpha);
plot(cx,cy,'b-');

% 加注释
title('Gaussian Discriminate Analysis(GDA)');
xlabel('Feature Dimension (One)');
ylabel('Feature Dimension (Two)');
legend('Class 1', 'Class 2', 'Discriminate Boundary');