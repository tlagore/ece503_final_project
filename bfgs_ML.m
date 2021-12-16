% To implement memoryless BFGS algorithm
% Example: [xs,fs,k] = bfgs_ML('f_rosen','g_rosen',[0;2],1e-6);
function [xs,fs,k] = bfgs_ML(fname,gname,x0,epsi,D,mu)
format compact
format long
k = 1;
xk = x0;
fk = feval(fname,xk,D,mu);
gk = feval(gname,xk,D,mu);
dk = -gk;
ak = bt_lsearch2019(xk,dk,fname,gname,D,mu);
dtk = -ak*gk;
xk_new = xk + dtk;
fk_new = feval(fname,xk_new,D,mu);
dfk = abs(fk - fk_new);
er = max(dfk,norm(dtk));
while er >= epsi
      gk_new = feval(gname,xk_new,D,mu);
      gmk = gk_new - gk;
      gk = gk_new;
      rk = 1/(dtk'*gmk);
      if rk <= 0
         dk = -gk;
      else
         tk = dtk'*gk;
         qk = gk - (rk*tk)*gmk;
         bk = rk*(gmk'*qk - tk);
         dk = bk*dtk - qk;
      end
      fk = fk_new;
      xk = xk_new;
      ak = bt_lsearch2019(xk,dk,fname,gname,D,mu);
      dtk = ak*dk;
      xk_new = xk + dtk;
      fk_new = feval(fname,xk_new,D,mu);
      dfk = abs(fk - fk_new);
      er = max(dfk,norm(dtk));
      k = k + 1;
end
disp('solution:')
xs = xk_new
disp('objective function at solution point:')
fs = feval(fname,xs,D,mu)
format short
disp('number of iterations at convergence:')
k