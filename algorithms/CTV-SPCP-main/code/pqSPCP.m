function [L,E]=pqSPCP(D,L_int,p,q,lambda1,lambda2,mu1,mu2)
% mu1>1, mu1=2;
% mu2=1/2,mu2=1;
% initial
normD = norm(D,'fro');
%[u,s,v]=svd(D,'econ');
% rk=6;
% L = u(:,1:rk)*s(1:rk,1:rk)*v(:,1:rk)';
L = L_int;
L_old = L;
E = D-L;
E_old = E;
maxIter =100;
eps = 1e-6;
ratio =1.1;
tol = 1e-6;
for i=1:maxIter
    Error = L_old+E_old-D;
    [~,s,~]=svd(L_old,'econ');
    diags = diag(s);
    w_l = p./(diags+eps).^(1-p);
    [u,s,v]=svd(L_old-Error/mu1,'econ');
    L = u*diag(softhre_l(diag(s),lambda1,w_l))*v';
    w_e = q./(abs(E_old)+eps).^(1-q);
    E = softhre_s(E_old-Error/mu2,lambda2,w_e);
    relL = norm(L-L_old,'fro')/normD;
    relE = norm(E-E_old,'fro')/normD;
    
    if mod(i,10)==0
        fprintf('===========iter = %d, relative L =%.8f, E = %.8f========\n',i,relL, relE);
    end
    if relL < tol && relE < tol
        fprintf('===========It has converged, iter=%d\n',i);
        break;
    else
        L_old = L;
        E_old = E;
        mu1 = ratio*mu1;
        mu2 = ratio*mu2;
    end
end
end


function out=softhre_s(a,tau,w)
out = sign(a).* max( abs(a) - tau*w, 0);
end
function out=softhre_l(a,tau,w)
out = max( a - tau*w, 0);
end

