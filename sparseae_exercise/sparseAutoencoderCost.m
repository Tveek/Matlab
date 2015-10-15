function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 
% %--------------vectorial implementation with repmat function--------------------
% %矩阵向量化形式实现,速度比不用向量快得多
% Jocst = 0; %平方误差
% Jweight = 0; %规则项惩罚
% Jsparse = 0; %稀疏性惩罚
% [n, m] = size(data); %m为样本数，这里是10000，n为样本维数，这里是64
% 
% %feedforward前向算法计算隐含层和输出层的每个节点的z值（线性组合值）和a值（激活值）
% %data每一列是一个样本，
% z2 = W1*data + repmat(b1,1,m); %W1*data的每一列是每个样本的经过权重W1到隐含层的线性组合值,repmat把列向量b1扩充成m列b1组成的矩阵
% a2 = sigmoid(z2);
% z3 = W2*a2 + repmat(b2,1,m);
% a3 = sigmoid(z3);
% 
% %计算预测结果与理想结果的平均误差
% Jcost = (0.5/m)*sum(sum((a3-data).^2));
% %计算权重惩罚项
% Jweight = (1/2)*(sum(sum(W1.^2))+sum(sum(W2.^2)));
% %计算稀疏性惩罚项
% rho_hat = (1/m)*sum(a2,2);
% Jsparse = sum(sparsityParam.*log(sparsityParam./rho_hat)+(1-sparsityParam).*log((1-sparsityParam)./(1-rho_hat)));
% 
% %计算总损失函数
% cost = Jcost + lambda*Jweight + beta*Jsparse;
% 
% %反向传播求误差值
% delta3 = -(data-a3).*fprime(a3); %每一列是一个样本对应的误差
% sterm = beta*(-sparsityParam./rho_hat+(1-sparsityParam)./(1-rho_hat)); 
% delta2 = (W2'*delta3 + repmat(sterm,1,m)).*fprime(a2);
% 
% %计算梯度
% W2grad = delta3*a2';
% W1grad = delta2*data';
% W2grad = W2grad/m + lambda*W2;
% W1grad = W1grad/m + lambda*W1;
% b2grad = sum(delta3,2)/m; %因为对b的偏导是个向量，这里要把delta3的每一列加起来
% b1grad = sum(delta2,2)/m;

% %-------------non-vectorial implementation---------------------
% %对每个样本进行计算, 
% [n m] = size(data);
% a2 = zeros(hiddenSize,m);
% a3 = zeros(visibleSize,m);
% Jcost = 0;    %平方误差项
% rho_hat = zeros(hiddenSize,1);   %隐含层每个节点的平均激活度
% Jweight = 0;  %权重衰减项   
% Jsparse = 0;   % 稀疏项代价
% 
% for i=1:m
%     %feedforward向前转播
%     z2(:,i) = W1*data(:,i)+b1;
%     a2(:,i) = sigmoid(z2(:,i));
%     z3(:,i) = W2*a2(:,i)+b2;
%     a3(:,i) = sigmoid(z3(:,i));
%     Jcost = Jcost+sum((a3(:,i)-data(:,i)).*(a3(:,i)-data(:,i)));
%     rho_hat = rho_hat+a2(:,i);  %累加样本隐含层的激活度
% end
% 
% rho_hat = rho_hat/m; %计算平均激活度
% Jsparse = sum(sparsityParam*log(sparsityParam./rho_hat) + (1-sparsityParam)*log((1-sparsityParam)./(1-rho_hat))); %计算稀疏代价
% Jweight = sum(W1(:).*W1(:))+sum(W2(:).*W2(:));%计算权重衰减项
% cost = Jcost/2/m + Jweight/2*lambda + beta*Jsparse; %计算总代价
% 
% for i=1:m
%     %backpropogation向后传播
%     delta3 = -(data(:,i)-a3(:,i)).*fprime(a3(:,i));
%     delta2 = (W2'*delta3 +beta*(-sparsityParam./rho_hat+(1-sparsityParam)./(1-rho_hat))).*fprime(a2(:,i));
% 
%     W2grad = W2grad + delta3*a2(:,i)';
%     W1grad = W1grad + delta2*data(:,i)';
%     b2grad = b2grad + delta3;
%     b1grad = b1grad + delta2;
% end
% %计算梯度
% W1grad = W1grad/m + lambda*W1;
% W2grad = W2grad/m + lambda*W2;
% b1grad = b1grad/m;
% b2grad = b2grad/m;




%--------------vectorial implementation with bsxfun function--------------------
a1 = sigmoid(bsxfun(@plus,W1 * data,b1)); %hidden层输出  
a2 = sigmoid(bsxfun(@plus,W2 * a1,b2)); %输出层输出  
p = mean(a1,2); %隐藏神经元的平均活跃度  
sparsity = sparsityParam .* log(sparsityParam ./ p) + (1 - sparsityParam) .* log((1 - sparsityParam) ./ (1.-p)); %惩罚因子  
cost = sum(sum((a2 - data).^2)) / 2 / size(data,2) + lambda / 2 * (sum(sum(W1.^2)) + sum(sum(W2.^2))) + beta * sum(sparsity); %代价函数  
delt2 = (a2 - data) .* a2 .* (1 - a2); %输出层残差  
delt1 = (W2' * delt2 + beta .* repmat((-sparsityParam./p + (1-sparsityParam)./(1.-p)),1,size(data,2))) .* a1 .* (1 - a1); %hidden层残差  
W2grad = delt2 * a1' ./ size(data,2) + lambda * W2;   
W1grad = delt1 * data' ./ size(data,2) + lambda * W1;  
b2grad = sum(delt2,2) ./ size(data,2);  
b1grad = sum(delt1,2) ./ size(data,2);  




%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end


%      Implementation of derivation of f(z) 
% f(z) = sigmoid(z) = 1./(1+exp(-z))
% a = 1./(1+exp(-z))
% delta(f) = a.*(1-a)
function dz = fprime(a)
    dz = a.*(1-a);
end


%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

