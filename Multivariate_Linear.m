% Exercise 3: Multivariate Linear Regression
% 多变量线性回归
%
% 载入数据并作预处理
x = load('data\ex3x.dat');
y = load('data\ex3y.dat');
m = length(x);
x = [ones(m, 1), x];  %第一列全为1，加入偏置项
x_unscaled = x; %保存为归一化的x，后面方程解要用到
sigma = std(x);  % 如果为vector，则返回标准差，如果是矩阵，返回每一列即每个维的标准差
mu = mean(x);  % 如果为vector，返回平均值，如果为矩阵，返回每一列的平均值
x(:,2) = (x(:,2) - mu(2))./ sigma(2);  % 归一化：减去平均值再除标准差,x(:2)表示第二列
x(:,3) = (x(:,3) - mu(3))./ sigma(3);
%
% 为画图做准备
figure;
plotstyle = {'b', 'r', 'g', 'k', 'b--', 'r--'}; %不同的学习率用不同的画线风格
%
% 梯度下降
MAX_ITR = 100;
alpha = [0.01, 0.03, 0.1, 0.3, 1, 1.3];%学习率
theta_grad_descent = zeros(size(x(1,:)));
n = length(alpha);%六组不同的学习率
for i = 1:n
    theta = zeros(size(x(1,:)))'; % size(x(1,:))返回1*n向量，n为每个样本的维数，转置后为n*1的0向量
    J = zeros(100, 1);
    for num_iterations = 1:MAX_ITR
        J(num_iterations) = (0.5/m).* sum((y-x*theta).^2);  % 损失函数,
        theta = theta - alpha(i)*(1/m).*x'*(x*theta-y);% .* 是矩阵中对应位置变量相乘，组成新得矩阵
    end
    plot(0:49, J(1:50), char(plotstyle(i)), 'LineWidth', 2);%画出损失函数的图像
    hold on;
    
    %通过实验发现alpha为1时损失最小，这里记录下这时的theta
    if alpha(i) == 1
        theta_grad_descent = theta;
    end
end
legend('0.01', '0.03', '0.1', '0.3', '1', '1.3');
xlabel('Number of iterations');
ylabel('Cost L');

% 预测
theta_grad_descent;

% 预测房子面积为1650，房间数为3的房价
price_grad_desc = dot(theta_grad_descent, [1, (1650-mu(2))/sigma(2), (3-mu(3))/sigma(3)])

% Normal equations
theta = inv(x_unscaled'*x_unscaled)*x_unscaled'*y

price_normal = dot(theta, [1, 1650, 3])