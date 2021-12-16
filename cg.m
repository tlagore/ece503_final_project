% To implement CG algorithm with Polak-Ribiere-Polyak-(plus)'s beta.
% Example:
% [xs,fs,k] = cg('f_rosen','g_rosen',[-1;-1],1e-6);
function [xs,fs,k] = cg(fname,gname,x0,epsi,D,mu)
format compact
format long
n = length(x0);
k = 0;
xk = x0;
gk = feval(gname,xk,D,mu);
dk = -gk;
er = norm(gk);
while er >= epsi
    ak = bt_lsearch2019(xk,dk,fname,gname,D,mu);
    xk_new = xk + ak*dk;  
    gk_new = feval(gname,xk_new,D,mu);
    gmk = gk_new - gk;
    bk = max((gk_new'*gmk)/(gk'*gk),0);
    dk = -gk_new + bk*dk;
    xk = xk_new;
    gk = gk_new;
    er = norm(gk);
    k = k + 1;
end
% disp('solution:')
xs = xk;
disp('objective function at solution point:')
fs = feval(fname,xs,D,mu);
format short
disp('number of iterations at convergence:')
k