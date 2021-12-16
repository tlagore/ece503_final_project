%f_wdbc.m
% modified costmax function
function f = f_wdbc(w, D, mu) 
    e = exp(1);
    
    [num_rows, num_samples] = size(D);
    
    X = D(1:num_rows-1,:);
    X = [X; ones(1, num_samples)];
    y = D(num_rows,:);
    
    P = num_samples;

    partsum = 0;
    for p= 1:P
        partsum = partsum + log(1 + e^(-y(p)*(w'*X(:,p))));
    end
    
    f = ((1/P) * partsum) + mu/2*(norm(w,2)^2);
end