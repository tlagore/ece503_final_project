% f_elw.m
% Tyrone Lagore V00995698
% regularized log cost gradient
function f = f_elw(wh, D, mu) 
    % e = exp(1);
    
    [num_rows, num_samples] = size(D);
    
    X = D(1:num_rows-1,:);
    X = [X; ones(1, num_samples)];
    y = D(num_rows,:);
    
    P = num_samples;
    
    Xy = X.*y;
    regularizer = (mu/2) * (wh'*wh);
    % f = sum((log(1 + e.^(-Xy'*wh)) / P)) + regularizer;
    
    f = sum(log(1+exp(-Xy'*wh)))/P + regularizer;
end