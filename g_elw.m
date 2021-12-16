% g_elw.m
% Tyrone Lagore V00995698
% regularized log cost gradient
function g = g_elw(wh, D, mu) 
    e = exp(1);
    
    [num_rows, num_samples] = size(D);
    
    X = D(1:num_rows-1,:);
    X = [X; ones(1, num_samples)];
    y = D(num_rows,:);
    
    P = num_samples;
    
    regularizer = mu*wh;
    Xy = X.*y;
    g = regularizer - (Xy*(1./(1+e.^(Xy'*wh)))/P);   
end