function g = g_LRBC(w,X)
[~,P] = size(X);
q1 = exp(X'*w);
q = 1./(1+q1);
g = -(X*q)/P;