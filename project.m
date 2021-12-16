[Xtr, y_tr, Xte, y_te] = lr_train_and_predict('heart_cleaned_filled.csv');

function [Xtr, y_tr, Xte, y_te] =  lr_train_and_predict(file_name)
    [Xtr, y_tr, Xte, y_te, num_features] = load_and_prepare_lr_data(file_name);

    tic
    mu = 0.01;
    w0 = zeros(num_features+1, 1);
    [xs, ~, ~] = bfgs_ML('f_elw', 'g_elw', w0, 1e-6, [Xtr; y_tr], mu);
    disp("Trained ws_1 with confusion matrix")
    disp("Confusion against test data:")
    [confusion, accuracy] = evaluate_lrbc(Xte, y_te, xs);
    disp(confusion);
    fprintf("bfgs_ML, mu=%f: Accuracy of %%%.6f on test data\n\n", mu, accuracy)
    disp(toc)
    
    tic
    w0 = zeros(num_features+1, 1);
    [xs, ~, ~] = bfgs('f_elw', 'g_elw', w0, 1e-6, [Xtr; y_tr], mu);
    disp("Trained ws_1 with confusion matrix")
    disp("Confusion against test data:")
    [confusion, accuracy] = evaluate_lrbc(Xte, y_te, xs);
    disp(confusion);
    fprintf("bfgs, mu=%f: Accuracy of %%%.6f on test data\n\n", mu, accuracy)
    disp(toc)
    
    tic
    mu = 0.1;
    w0 = zeros(num_features+1, 1);
    [xs, ~, ~] = cg('f_elw', 'g_elw', w0, 1e-6, [Xtr; y_tr], mu);
    disp("Trained ws_1 with confusion matrix")

    disp("Confusion against test data:")
    [confusion, accuracy] = evaluate_lrbc(Xte, y_te, xs);
    disp(confusion);
    fprintf("cg, mu=%f: Accuracy of %%%.6f on test data\n\n", mu, accuracy)
    disp(toc)

    tic
    mu = 0.1;
    [xs, confusion] = LRBC_newton(Xtr, y_tr, 1);
    disp("Trained ws_1 with confusion matrix")
    disp(confusion)

    disp("Confusion against test data:")
    [confusion, accuracy] = evaluate_lrbc(Xte, y_te, xs);
    disp(confusion);
    fprintf("newton, iterations=1: Accuracy of %%%.6f on test data\n\n", mu, accuracy)
    disp(toc)
end

function [Xtr, y_tr, Xte, y_te, num_features] = load_and_prepare_lr_data(file_name)
    X = readmatrix(file_name);

    [num_rows, total_samples] = size(X);
    num_features = num_rows - 1;

    train_size = floor(total_samples*0.8);

    Xtr = X(1:num_features,1:train_size);
    y_tr = X(num_features+1,1:train_size);

    Xte = X(1:num_features,train_size+1:total_samples);
    y_te = X(num_features+1,train_size+1:total_samples);

    % conform our labels to logistic regression
    % -1 instead of 0
    y_tr(y_tr == 0) = -1;
    y_te(y_te == 0) = -1;
end