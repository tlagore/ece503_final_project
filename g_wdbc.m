%g_wdbc.m
% modified costmax function gradient
function g = g_wdbc(w, D, mu) 
    e = exp(1);
    
    X = D(1:30,:);
    X = [X; ones(1, length(X))];
    y = D(31,:);
    P = length(X);
    
    Xy = X.*y;
    g = Xy * (1./(1+e.^(Xy'*w)));
    
    % g = zeros(length(w),1);

    %for p= 1:P
    %    numerator = (y(p)*X(:,p));
    %    denominator = (1+e^(y(p)*(w'*(X(:,p)))));
    %    g = g + (numerator/denominator);
    %end
    
    regularizer = mu*w;
    g = regularizer - (1/P)*(g);
end