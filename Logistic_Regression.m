% Exercise 4: Logistic Regression and Newton's Method
% 逻辑斯蒂回归和牛顿方法
% 这里是采用logistic regression模型和newton算法对数据进行二类分类（0和1）的
% 数据是40个被大学接收的学生和40个被拒绝的学生的两门成绩
% 从中学到分类模型然后给出一名学生的两门成绩，预测被大学接收的概率


% 初始化数据
x = load('data\ex4x.dat');
y = load('data\ex4y.dat');
m = length(x);   %m训练样本数
x = [ones(m,1), x];  %加入偏置项1到第一列
n = size(x, 2);  % n为特征数，这里是3

% find找出满足条件的行号索引
pos = find(y == 1); 
neg = find(y == 0);
% 画图显示

plot(x(pos, 2), x(pos, 3), '+'); hold on
plot(x(neg, 2), x(neg, 3), 'o'); hold on
xlabel('Exam 1 score')
ylabel('Exam 2 score')
% 定义sigmod 函数

g = @(z) 1.0 ./ (1.0 + exp(-z));
% 用法: 要找出2的sigmod值，调用g(2)
% 如果传入矩阵或向量，则计算矩阵或向量中每个元素的sigmod
% 牛顿法
MAX_ITR = 20;   % 最大迭代次数
J = zeros(MAX_ITR, 1); % 保存每次迭代后损失
theta = zeros(n, 1); % 初始化训练参数


for i = 1:MAX_ITR
    h = g(x*theta);   % 计算假设函数,得到一个列向量，每行为那个样本属于1的概率
    J(i) = (1/m) * sum(-y.*log(h) - (1-y).*log(1-h)); % 计算损失函数
    % 计算梯度和海森矩阵
    grad = (1/m) .* x' * (h - y); %计算J对theta的一阶导
    % 自己想到的实现
    H = (1/m) .* x' * (repmat(h .* (1-h), 1, n) .* x);  %计算海森矩阵，即J对theta的二阶导
    theta = theta - H\grad;
    % Solution中的实现
    % H = (1/m).*x'*diag(h)*diag(1-h)*x;
    % theta = theta - H\grad; % 左除跟inv(H)*grad一样
end
% Display theta
theta

% prediction
fprintf(['Probability that a student with a escore of exam 1 20 and 80 on exam 2 \n' ...
    'will not be admitted to college is %f\n'], 1 - g([1 20 80]*theta));



% 画出牛顿方法结果
% 决策边界：theta(1)*1+theta(2)*x2 + theta(3)*x3=0
% 两点确定一点直线，这里选择x2维度上的两点，
plot_x = [min(x(:,2))-2, max(x(:,2))+2];
% 算出对应的x3值
plot_y = (-1./theta(3)).*(theta(2).*plot_x+theta(1));
% 画直线
plot(plot_x, plot_y)
legend('Admitted', 'Not admitted', 'Decision Boundary')
hold off

% 画出J
figure
plot(0:MAX_ITR-1, J, 'o--', 'MarkerFaceColor', 'r', 'MarkerSize', 8)
xlabel('Iteration'); ylabel('J')
J

% 跟梯度下降法对比
MAX_ITR = 20;   % 最大迭代次数
J = zeros(MAX_ITR, 1); % 保存每次迭代后损失
theta = zeros(n, 1); % 初始化训练参数
alpha = 0.01;
for i = 1:MAX_ITR
    h = g(x*theta);   % 计算假设函数,得到一个列向量，每行为那个样本属于1的概率
    J(i) = (1/m) * sum(-y.*log(h) - (1-y).*log(1-h)); % 计算损失函数
    % 计算梯度
    grad = (1/m) .* x' * (h - y); %计算J对theta的一阶导
    theta = theta - alpha*grad;
end
% Display theta
theta
hold on
plot(0:MAX_ITR-1, J, '*--', 'MarkerFaceColor', 'r', 'MarkerSize', 8)
legend('newton', 'gradient descent')
