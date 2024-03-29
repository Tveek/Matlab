% Exercise 6: Naive Bayes
% 朴素贝叶斯练习：垃圾邮件分类
% Multinomial Naive Bayes model

numTrainDocs = 700;  % 文档数量
numTokens = 2500;  % 词库

 %读取训练集特征文件，每行有3个元素，表示文档id、词id和词频
 %dlmread读取以ascii码分割的文件中的数字
M = dlmread('ex6DataPrepared\train-features.txt', ''); 

%根据训练集特征构建稀疏矩阵，矩阵大小为numTrainDocs*numTokens
%S = sparse(i,j,s,m,n), i,j为向量，S(i(k),j(k))=s(k),S大小为m*n
%表示形式为:(i(k) j(k)) s(k)
spmatrix = sparse(M(:,1), M(:,2), M(:,3), numTrainDocs, numTokens); 


%例子：
% B=sparse([1,2,3],[1,2,3],[0,1,2],4,4,4)
% 
% B =
%    (2,2)        1
% 
%    (3,3)        2
% 
% 其中i=[1,2,3]，稀疏矩阵的行位置；j=[1,2,3]，稀疏矩阵的列位置；s=[0,1,2]，稀疏矩阵元素值。 其位置为一一对应。
%s(1,1)=0,s(2,2)=1,s(3,3)=2;


% 根据稀疏矩阵还原完整矩阵,最终得到以文档ID为行号，词ID为列号，词频为值的矩阵
train_matrix = full(spmatrix);

%读取文档类别
train_labels = dlmread('ex6DataPrepared\train-labels.txt');

% 训练过程
spam_index = find(train_labels);   %记录垃圾邮件索引
nonspam_index = find(train_labels==0);  %记录非垃圾邮件索引

proc_spam = length(spam_index)/numTrainDocs; % 计算垃圾邮件概率
% 分别计算垃圾和非垃圾邮件中每个单词出现的次数
% train_matrix(spam_index,:)找出属于垃圾邮件的文档向量
% sum(train_matrix(spam_index, :))对每一列的所有行累加起来，即统计该词在垃圾邮件中出现的总次数
wc_spam = sum(train_matrix(spam_index, :));
wc_nonspam = sum(train_matrix(nonspam_index, :));
% 分别计算tokens在垃圾邮件和非垃圾邮件中出现的概率
prob_tokens_spam = (wc_spam + 1) ./ (sum(wc_spam) + numTokens);
prob_tokens_nonspam = (wc_nonspam + 1) ./ (sum(wc_nonspam) + numTokens);

% 测试
test_labels = dlmread('ex6DataPrepared\test-labels.txt');

M = dlmread('ex6DataPrepared\test-features.txt', '');

% 构建稀疏矩阵，这么貌似有个问题，如果测试文档中不含有第2500个词的话，构建出的稀疏矩阵列数不是numTokens
% 下面的test_matrix乘于训练出来的(log(spam_wc_proc))'就肯定会出错
spmatrix = sparse(M(:,1), M(:,2), M(:,3));
test_matrix = full(spmatrix);

% 分别计算test_matrix的每一行即每篇文档属于垃圾邮件和非垃圾邮件的概率,最终还是以词语出现的频次*词语概率（通过先前数据训练得到） 
% logp(x|y=1) + logp(y=1)
test_spam_proc = test_matrix * (log(prob_tokens_spam))' + log(proc_spam);
% logp(x|y=0) + logp(y=0)
test_nonspam_proc = test_matrix * (log(prob_tokens_nonspam))' + log(1-proc_spam);
% 预测
test_spam = test_spam_proc > test_nonspam_proc; 
% 计算分类准确率
accuracy = sum(test_spam==test_labels) / length(test_labels);
fprintf('Accuracy:%f\n', accuracy);