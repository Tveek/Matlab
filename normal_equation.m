function theta=normal_equation(X,Y)
c=ones(100,1);%产生100行全为1的列向量
X=[c X];%一个向量增加一列
theta=(inv(X'*X))*X'*Y;
end