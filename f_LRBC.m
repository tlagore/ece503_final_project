function f = f_LRBC(w,X)
P = size(X,2);
f = sum(log(1+exp(-X'*w)))/P;
% [num_rows, num_samples] = size(D);
%     
% X = D(1:num_rows-1,:);
% X = [X; ones(1, num_samples)];
% y = D(num_rows,:);
% 
% P = num_samples;
% 
% Xy = X.*y;
% 
% f = sum(log(1+exp(-Xy'*w)))/P;