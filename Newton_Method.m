function theta=Newton_Method(x,y)

[m,n]=size(x);

x = [ones(m,1), x];  %加入偏置项1到第一列

MAX_ITR = 20;   % 最大迭代次数
J = zeros(MAX_ITR, 1); % 保存每次迭代后损失
theta = zeros(n+1, 1); % 初始化训练参数


for i = 1:MAX_ITR   
    
    h=x*theta;
%     J(i) = (1/2) .* sum((h-y).^2); % 计算损失函数
%     % 计算梯度和海森矩阵
     grad = x' * (h - y); %计算J对theta的一阶导
%     % 自己想到的实现
     H = x'*x;  %计算海森矩阵，即J对theta的二阶导
%     theta = theta - inv(H)*grad;
    % Solution中的实现
    %H = x'*diag(h)*diag(1-h)*x;
    %theta = theta - inv(H)*grad;
    theta = theta - H\grad; % 左除跟inv(H)*grad一样
end
end