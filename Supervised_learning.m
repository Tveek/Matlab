x = load('data\ex2x.dat'); % x是男生年龄,是一个列向量
y = load('data\ex2y.dat'); % y是男生身高，也是一个列向量

figure % 打开新的绘画窗口
plot(x, y, 'o');
ylabel('Height in meters');
xlabel('Age in years');
m = length(y);  % m存储样本数目
x = [ones(m, 1), x]; % 给x添加第一列全为1,这样x的每一行有2个数，表示每个男生的特征，第二列才是男生年龄
theta = zeros(2, 1); % theta为权重参数，一个2维列向量，初始化为0
alpha = 0.07; %步长
MAX_ITR = 1500; %最多迭代次数
ERROR = 1e-10;
% Batch gradient decent
for i=1:MAX_ITR
    % Jtheta = 0.07/m*sum(x*theta-y).^2); 计算代价
    grad = 1/m*x'*(x*theta-y); %计算梯度
    prev_theta = theta;
    theta = theta - alpha*grad; 
    if abs(prev_theta-theta)<ERROR
        break
    end
    fprintf('%d\n',i);%记录运行了多少次
end
%显示所求得到的theta;
theta; 
%输入升高，推测年龄
[1 3.5]*theta;
[1 7]*theta;
hold on % Plot new data without clearing old plot
plot(x(:,2), x*theta, '-'); % remember that x is now a matrix with 2 columns
                            % and the second column contains the time info
legend('Training data', 'Linear regression');