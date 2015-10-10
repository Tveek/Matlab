% Exercise 6: Naive Bayes
% ���ر�Ҷ˹��ϰ�������ʼ�����
% Multinomial Naive Bayes model

numTrainDocs = 700;  % �ĵ�����
numTokens = 2500;  % �ʿ�

 %��ȡѵ���������ļ���ÿ����3��Ԫ�أ���ʾ�ĵ�id����id�ʹ�Ƶ
 %dlmread��ȡ��ascii��ָ���ļ��е�����
M = dlmread('ex6DataPrepared\train-features.txt', ''); 

%����ѵ������������ϡ����󣬾����СΪnumTrainDocs*numTokens
%S = sparse(i,j,s,m,n), i,jΪ������S(i(k),j(k))=s(k),S��СΪm*n
%��ʾ��ʽΪ:(i(k) j(k)) s(k)
spmatrix = sparse(M(:,1), M(:,2), M(:,3), numTrainDocs, numTokens); 


%���ӣ�
% B=sparse([1,2,3],[1,2,3],[0,1,2],4,4,4)
% 
% B =
%    (2,2)        1
% 
%    (3,3)        2
% 
% ����i=[1,2,3]��ϡ��������λ�ã�j=[1,2,3]��ϡ��������λ�ã�s=[0,1,2]��ϡ�����Ԫ��ֵ�� ��λ��Ϊһһ��Ӧ��
%s(1,1)=0,s(2,2)=1,s(3,3)=2;


% ����ϡ�����ԭ��������,���յõ����ĵ�IDΪ�кţ���IDΪ�кţ���ƵΪֵ�ľ���
train_matrix = full(spmatrix);

%��ȡ�ĵ����
train_labels = dlmread('ex6DataPrepared\train-labels.txt');

% ѵ������
spam_index = find(train_labels);   %��¼�����ʼ�����
nonspam_index = find(train_labels==0);  %��¼�������ʼ�����

proc_spam = length(spam_index)/numTrainDocs; % ���������ʼ�����
% �ֱ���������ͷ������ʼ���ÿ�����ʳ��ֵĴ���
% train_matrix(spam_index,:)�ҳ����������ʼ����ĵ�����
% sum(train_matrix(spam_index, :))��ÿһ�е��������ۼ���������ͳ�Ƹô��������ʼ��г��ֵ��ܴ���
wc_spam = sum(train_matrix(spam_index, :));
wc_nonspam = sum(train_matrix(nonspam_index, :));
% �ֱ����tokens�������ʼ��ͷ������ʼ��г��ֵĸ���
prob_tokens_spam = (wc_spam + 1) ./ (sum(wc_spam) + numTokens);
prob_tokens_nonspam = (wc_nonspam + 1) ./ (sum(wc_nonspam) + numTokens);

% ����
test_labels = dlmread('ex6DataPrepared\test-labels.txt');

M = dlmread('ex6DataPrepared\test-features.txt', '');

% ����ϡ�������ôò���и����⣬��������ĵ��в����е�2500���ʵĻ�����������ϡ�������������numTokens
% �����test_matrix����ѵ��������(log(spam_wc_proc))'�Ϳ϶������
spmatrix = sparse(M(:,1), M(:,2), M(:,3));
test_matrix = full(spmatrix);

% �ֱ����test_matrix��ÿһ�м�ÿƪ�ĵ����������ʼ��ͷ������ʼ��ĸ���,���ջ����Դ�����ֵ�Ƶ��*������ʣ�ͨ����ǰ����ѵ���õ��� 
% logp(x|y=1) + logp(y=1)
test_spam_proc = test_matrix * (log(prob_tokens_spam))' + log(proc_spam);
% logp(x|y=0) + logp(y=0)
test_nonspam_proc = test_matrix * (log(prob_tokens_nonspam))' + log(1-proc_spam);
% Ԥ��
test_spam = test_spam_proc > test_nonspam_proc; 
% �������׼ȷ��
accuracy = sum(test_spam==test_labels) / length(test_labels);
fprintf('Accuracy:%f\n', accuracy);