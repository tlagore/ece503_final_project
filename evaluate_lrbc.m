%evaluate_lrbc.m
% given 3 sets of class data (in order), 
% and their tuned parameters, perform the predictions
% and display confusion matrix and accuracy
% for the predictions.
function [confusion, accuracy] = evaluate_lrbc(X, y, ws)
    total_samples = length(X);
    
    disp(size(X))
    disp(total_samples)
    % prepare dataset for predictions
    Xh = [X; ones(1, total_samples)];
    
    % prepare predictions
    
    est = Xh'*ws;

    guess = sign(est);
    
    guess(guess==-1) = 2;

    confusion = zeros(2,2);

    i = 1;
    
    % generate confusion matrix by incrementing
    % (predicted(i), actual(i)) for each sample
    while i <= length(guess)
        if y(i) == -1
            actual = 2;
        else
            actual = 1;
        end
        
        try
            confusion(guess(i), actual) = confusion(guess(i), actual)+1;
        catch ME
            sprintf("exception setting value. guess(i)=%d and y(i) = %d", guess(i), y(i))
            rethrow(ME)
        end 
                
        i = i + 1;
    end
    
    % calculate accuracy
    diagonal = trace(confusion);        
    accuracy = 100*(diagonal/total_samples);
end