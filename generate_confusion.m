%generate_confusion.m
% generate confusion matrix and accuracy for a given D = dataset, Ws = parameters,
% y = true labels, and K = number of classifiers
function [confusion, accuracy] = generate_confusion(D, Ws, y, K)
    disp(size(D))
    [~,ind_pre] =  max((D'*Ws)');
    confusion = zeros(K,K); 
    for j = 1:K 
        ind_j = find(y == j); 
        for i = 1:K 
            ind_pre_i = find(ind_pre == i); 
            confusion(i,j) = length(intersect(ind_j,ind_pre_i)); 
        end 
    end
    
    accuracy = (trace(confusion) / length(D)) * 100;
end

